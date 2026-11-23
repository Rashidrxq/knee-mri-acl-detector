"""Prepare one local research volume or DICOM series for preview and prediction."""
import hashlib
import io
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import PurePosixPath

import numpy as np
from pydicom.errors import InvalidDicomError

from dicom_input import DicomSeriesError, decode_dicom_members, dicom_members, visible_members
from model import MAX_BYTES, dataset_scan


class ScanInputError(ValueError):
    """An input problem with a message suitable for the upload screen."""


@dataclass
class Scan:
    volume: np.ndarray
    source: str
    digest: str | None = None


def research_scan(data):
    try:
        volume = dataset_scan(data)
    except ValueError as exc:
        raise ScanInputError("This .pck could not be read. Choose an original KneeMRI scan from Kaggle.") from exc
    return Scan(volume, "KneeMRI", hashlib.sha256(data).hexdigest())


def load_scan(data, name):
    if len(data) > MAX_BYTES:
        raise ScanInputError("Choose one scan of 64 MiB or less.")
    extension = PurePosixPath(name).suffix.lower()
    if extension == ".pck":
        return research_scan(data)
    if extension != ".zip":
        raise ScanInputError("Choose a KneeMRI .pck file or a .zip archive.")

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = visible_members(archive)
            research = [entry for entry in entries if entry.filename.lower().endswith(".pck")]
            dicom = dicom_members(entries)
            if len(research) > 1:
                raise ScanInputError("This ZIP contains several KneeMRI scans. Upload one .pck file or its single-scan ZIP.")
            if research and dicom:
                raise ScanInputError("This ZIP mixes a KneeMRI scan with possible DICOM slices. Upload one format at a time.")
            if any(entry.filename.lower().endswith(".zip") for entry in entries):
                raise ScanInputError("This ZIP contains another ZIP. Open the outer archive and upload a single scan.")
            selected = set(research + dicom)
            sidecars = {".txt", ".md", ".json", ".csv"}
            if selected and any(
                entry not in selected and PurePosixPath(entry.filename).suffix.lower() not in sidecars
                and PurePosixPath(entry.filename).name.upper() not in {"README", "LICENSE"}
                for entry in entries
            ):
                raise ScanInputError("This ZIP mixes scan files with an unsupported file type. Upload one KneeMRI scan or one DICOM series.")
            if research:
                entry = research[0]
                if entry.file_size > MAX_BYTES:
                    raise ScanInputError("The .pck inside this ZIP exceeds 64 MiB. Choose a smaller scan.")
                return research_scan(archive.read(entry))
            if not dicom:
                raise ScanInputError("No supported scan found. Choose a ZIP with one KneeMRI .pck or one sagittal DICOM series.")
            try:
                volume = decode_dicom_members(archive, dicom)
            except DicomSeriesError as exc:
                raise ScanInputError(str(exc)) from exc
            except (ValueError, KeyError, AttributeError, TypeError, IndexError, InvalidDicomError) as exc:
                raise ScanInputError("Cannot read these DICOM slices. Export one uncompressed, single-frame sagittal MR series with complete geometry.") from exc
            return Scan(volume, "DICOM")
    except (zipfile.BadZipFile, EOFError, OSError, RuntimeError, zlib.error) as exc:
        raise ScanInputError("Cannot read this ZIP. Download it again or export an unencrypted archive.") from exc


def middle_slice(volume):
    image = np.asarray(volume[len(volume) // 2], dtype=np.float32)
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        low, high = image.min(), image.max()
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - low) * (255 / (high - low)), 0, 255).astype(np.uint8)
