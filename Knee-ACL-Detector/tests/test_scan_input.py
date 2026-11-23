import hashlib
import io
import json
import unittest
import zipfile

import numpy as np
from pydicom import dcmwrite
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from model import ROOT, predict
from scan_input import ScanInputError, load_scan, middle_slice


def archive(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as output:
        for name, data in files:
            output.writestr(name, data)
    return buffer.getvalue()


def dicom_slices(mix_series=False):
    study_uid, series_uid = generate_uid(), generate_uid()
    files = []
    # Reverse archive order to exercise ordering by physical position.
    for i in reversed(range(8)):
        uid = generate_uid()
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = uid
        ds = FileDataset(None, {}, file_meta=meta, preamble=bytes(128))
        ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, uid
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = generate_uid() if mix_series and i == 0 else series_uid
        ds.Modality = "MR"
        ds.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
        ds.ImagePositionPatient = [-i * 3, 0, 0]
        ds.PixelSpacing = [1, 1]
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.Rows, ds.Columns = 96, 96
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.PixelData = (np.arange(96 * 96, dtype="<u2") + i).tobytes()
        buf = io.BytesIO()
        dcmwrite(buf, ds, enforce_file_format=True)
        files.append((f"series/{i}", buf.getvalue()))
    return files


class ScanInputTests(unittest.TestCase):
    def test_raw_and_zipped_examples_match(self):
        examples = json.loads((ROOT / "examples" / "index.json").read_text())
        for case in examples.values():
            with self.subTest(file=case["file"]):
                raw = (ROOT / "examples" / case["file"]).read_bytes()
                source = load_scan(raw, case["file"])
                wrapped = archive([("vol01/scan.PCK", raw), ("__MACOSX/._scan.pck", b"metadata")])
                uploaded = load_scan(wrapped, "download.ZIP")
                self.assertEqual(uploaded.source, "KneeMRI")
                self.assertEqual(uploaded.digest, case["sha256"])
                self.assertEqual(uploaded.digest, hashlib.sha256(raw).hexdigest())
                np.testing.assert_array_equal(source.volume, uploaded.volume)
                np.testing.assert_array_equal(middle_slice(source.volume), middle_slice(uploaded.volume))
                self.assertEqual(predict(source.volume), predict(uploaded.volume))

    def test_preview_uses_middle_slice_and_preserves_input(self):
        volume = np.zeros((5, 3, 3), dtype=np.uint16)
        volume[2, 1, 1] = 200
        original = volume.copy()
        image = middle_slice(volume)
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image[1, 1], 255)
        self.assertEqual(image[0, 0], 0)
        np.testing.assert_array_equal(volume, original)

    def test_rejects_ambiguous_or_unsupported_uploads(self):
        for entries, reason in [
            ([("a.pck", b""), ("b.pck", b"")], "several KneeMRI"),
            ([("a.pck", b""), ("slice.dcm", b"")], "mixes"),
            ([("a.pck", b""), ("photo.jpg", b"")], "unsupported file type"),
            ([("inner.zip", b"")], "another ZIP"),
            ([("notes.txt", b"hello")], "No supported scan"),
            ([("broken.pck", b"")], ".pck could not be read"),
        ]:
            with self.subTest(reason=reason), self.assertRaisesRegex(ScanInputError, reason):
                load_scan(archive(entries), "download.zip")
        with self.assertRaisesRegex(ScanInputError, "Cannot read this ZIP"):
            load_scan(b"no archive", "scan.zip")

    def test_dicom_order_and_mixed_series(self):
        scan = load_scan(archive(dicom_slices()), "scan.zip")
        self.assertEqual(scan.source, "DICOM")
        self.assertIsNone(scan.digest)
        np.testing.assert_array_equal(scan.volume[:, 0, 0], np.arange(8))
        with self.assertRaisesRegex(ScanInputError, "one consistent series"):
            load_scan(archive(dicom_slices(mix_series=True)), "scan.zip")
