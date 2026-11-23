"""Inference for sagittal, proton-density fat-suppressed knee series."""
import json
import math
import struct
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 64 * 1024 * 1024


def check_shape(shape):
    if len(shape) != 3:
        raise ValueError("Select a complete sagittal image series.")
    depth, height, width = shape
    if not (8 <= depth <= 128 and 96 <= height <= 1024 and 96 <= width <= 1024
            and math.prod(shape) <= 25_000_000):
        raise ValueError("Unsupported series dimensions or slice count.")


def dataset_scan(data):
    """Read the original numeric KneeMRI layout without executing pickle instructions."""
    if len(data) > MAX_BYTES or not data.startswith(b"\x80\x02"):
        raise ValueError("Choose an original KneeMRI .pck volume.")
    # The released files use a short protocol-2 ndarray header, followed by
    # a BINUNICODE payload containing little-endian uint16 pixel bytes in Latin-1.
    import pickletools
    previous = deque(maxlen=3)
    shape = None
    header = data[:768]
    try:
        for op, arg, pos in pickletools.genops(header):
            if op.name == "TUPLE3" and len(previous) == 3 and all(type(x) is int for x in previous):
                shape = tuple(previous)
            if op.name == "NEWFALSE" and shape and b'u2' in header[:pos] and b'<' in header[:pos]:
                if data[pos+1:pos+4] != b'h\x03X':
                    break
                check_shape(shape)
                length = struct.unpack_from("<I", data, pos + 4)[0]
                start = pos + 8
                expected = math.prod(shape) * 2
                if not expected <= length <= expected * 2 or start + length > len(data):
                    break
                raw = data[start:start+length].decode("utf-8").encode("latin1")
                if len(raw) != expected:
                    break
                return np.frombuffer(raw, dtype="<u2").reshape(shape)
            previous.append(arg if op.name.startswith("BININT") else None)
    except (ValueError, UnicodeError, IndexError, struct.error):
        pass
    raise ValueError("Unsupported KneeMRI volume encoding.")


def features(volume):
    check_shape(volume.shape)
    # Mirror-paired depth summaries make inference independent of file slice order.
    n = len(volume)
    half = [round(f * (n - 1)) for f in (.2, .3, .38, .44, .48)]
    indices = half + [n - 1 - i for i in half]
    small = []
    for i in indices:
        image = Image.fromarray(np.asarray(volume[i], dtype=np.float32))
        small.append(np.asarray(image.resize((80, 80), Image.Resampling.BILINEAR)))
    small = np.stack(small)
    lo, hi = np.percentile(small, (2, 99.5))
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("The scan does not contain enough usable image contrast.")
    small = np.clip((small - lo) / (hi - lo), 0, 1)
    vectors = []
    for im in small:
        crop = im[8:72, 8:72]
        dy, dx = np.gradient(crop)
        magnitude = np.hypot(dx, dy)
        angle = np.remainder(np.arctan2(dy, dx), np.pi)
        bins = np.minimum((angle * (8 / np.pi)).astype(int), 7)
        def cells(a):
            return a.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3).reshape(8, 8, 64)
        hist = np.stack([cells(np.where(bins == b, magnitude, 0)).sum(-1)
                         for b in range(8)], axis=-1)
        hist /= np.sqrt((hist**2).sum(-1, keepdims=True) + .04)
        regions = cells(crop)
        vectors.append(np.concatenate((hist.ravel(), regions.mean(-1).ravel(),
                                       regions.std(-1).ravel())))
    pair = (np.stack(vectors[:5]) + np.stack(vectors[5:])) / 2
    return np.concatenate([pair[:2].mean(0), pair[2:].mean(0)]).astype(np.float32)


def predict(volume):
    model = json.loads((ROOT / "weights.json").read_text())
    vector = features(volume)
    score = float(vector @ np.array(model["coef"]) + model["intercept"])
    return score >= model["decision_threshold"]
