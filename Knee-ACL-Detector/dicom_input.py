"""Read a ZIP of classic single-frame sagittal MR images in memory."""
import io
import zipfile
from pathlib import PurePosixPath

import numpy as np
import pydicom
from pydicom.uid import MRImageStorage

from model import check_shape


def read_series(data):
    slices = []
    geometry = None
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        candidates = [i for i in archive.infolist() if not i.is_dir()
                      and not any(p.startswith((".", "__MACOSX")) for p in PurePosixPath(i.filename).parts)
                      and PurePosixPath(i.filename).suffix.lower() in ("", ".dcm", ".ima")
                      and PurePosixPath(i.filename).name.upper() != "DICOMDIR"]
        if not 8 <= len(candidates) <= 128 or sum(i.file_size for i in candidates) > 96 * 1024**2:
            raise ValueError("Choose a ZIP containing 8–128 slices from one sagittal series.")
        for entry in candidates:
            if entry.file_size > 5 * 1024**2:
                raise ValueError("Each DICOM file must be one supported single-frame slice.")
            ds = pydicom.dcmread(io.BytesIO(archive.read(entry)))
            if (str(ds.SOPClassUID) != str(MRImageStorage) or ds.Modality != "MR"
                or ds.PhotometricInterpretation != "MONOCHROME2" or int(ds.SamplesPerPixel) != 1
                or ds.file_meta.TransferSyntaxUID.is_compressed or int(getattr(ds,"NumberOfFrames",1)) != 1):
                raise ValueError("Use uncompressed, single-frame grayscale DICOM MR images.")
            if any(v in ("LOCALIZER","SCOUT") for v in getattr(ds,"ImageType",[])):
                raise ValueError("Localizer images cannot be analyzed.")
            rows, cols = int(ds.Rows), int(ds.Columns)
            check_shape((len(candidates), rows, cols))
            if int(ds.BitsAllocated) not in (8,16):
                raise ValueError("Use an 8-bit or 16-bit MR image series.")
            spacing = np.asarray(ds.PixelSpacing, dtype=float)
            orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
            location = np.asarray(ds.ImagePositionPatient, dtype=float)
            if (orientation.shape != (6,) or location.shape != (3,) or spacing.shape != (2,)
                or not np.isfinite(np.r_[orientation,location,spacing]).all() or np.any(spacing <= 0)):
                raise ValueError("Missing or invalid scan geometry.")
            horizontal, vertical = orientation[:3], orientation[3:]
            normal = np.cross(horizontal, vertical)
            if (abs(normal[0]) < .94 or abs(horizontal[1]) < .94 or abs(vertical[2]) < .94
                or not .85 <= spacing[0]/spacing[1] <= 1.18):
                raise ValueError("Select a sagittal series with standard in-plane orientation and near-square pixels.")
            identity = (str(ds.StudyInstanceUID),str(ds.SeriesInstanceUID),rows,cols)
            if geometry is None:
                geometry = identity, orientation, spacing, normal
            first, orient0, spacing0, normal0 = geometry
            if identity != first or not np.allclose(orientation,orient0,atol=.01) or not np.allclose(spacing,spacing0,rtol=.01):
                raise ValueError("ZIP must contain one consistent series from one exam.")
            slope, intercept = float(getattr(ds,"RescaleSlope",1)), float(getattr(ds,"RescaleIntercept",0))
            if not np.isfinite([slope,intercept]).all() or slope <= 0:
                raise ValueError("Unsupported image intensity scaling.")
            image = ds.pixel_array.astype(np.float32)*slope + intercept
            if horizontal[1] < 0:
                image = image[:,::-1]
            if vertical[2] > 0:
                image = image[::-1,:]
            slices.append((float(location @ normal0), image))
    slices.sort(key=lambda item:item[0])
    gaps = np.diff([s[0] for s in slices])
    if np.min(gaps) < .05 or not np.allclose(gaps,np.median(gaps),rtol=.15,atol=.1):
        raise ValueError("The DICOM series contains repeated or unevenly spaced slices.")
    return np.stack([s[1] for s in slices])
