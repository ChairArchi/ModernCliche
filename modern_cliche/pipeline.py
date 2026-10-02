from __future__ import annotations

from io import BytesIO
from hashlib import sha256
import json
import zipfile
import numpy as np
from PIL import Image

from .data import Sample, decode_image, deduplicate
from .segmentation import extract_mask, normalise_mask


def prepare(samples: list[Sample], method="u2net", threshold=.5, size=96, overrides=None, progress=None):
    samples, duplicates = deduplicate(samples)
    accepted, raw_masks, masks, quality, failures = [], [], [], [], []
    overrides = overrides or {}
    for index, sample in enumerate(samples):
        try:
            if sample.id in overrides:
                raw = np.asarray(overrides[sample.id], dtype=bool)
                if raw.shape != (sample.image.height, sample.image.width):
                    raise ValueError("Manual mask dimensions do not match the source image.")
                info = dict(method="manual-mask", needs_review=False, foreground_fraction=float(raw.mean()))
            else:
                raw, info = extract_mask(sample.image, method, threshold)
            mask = normalise_mask(raw, size)
            accepted.append(sample)
            raw_masks.append(raw)
            masks.append(mask)
            quality.append(info)
        except Exception as exc:
            failures.append({"id": sample.id, "title": sample.metadata.get("title", ""), "error": str(exc)})
        if progress:
            progress((index + 1) / max(1, len(samples)))
    return dict(samples=accepted, raw_masks=raw_masks, masks=masks, quality=quality,
                failures=failures, duplicates=duplicates,
                settings=dict(method=method, threshold=threshold, size=size, duplicate_hamming=3))


def fingerprint(prepared: dict, indices: list[int]) -> str:
    digest = sha256()
    for index in indices:
        digest.update(prepared["samples"][index].id.encode())
        digest.update(prepared["masks"][index].tobytes())
        digest.update(prepared["raw_masks"][index].tobytes())
    return digest.hexdigest()


def subset(prepared: dict, indices: list[int]) -> dict:
    return {**prepared, **{key: [prepared[key][index] for index in indices]
                          for key in ("samples", "raw_masks", "masks", "quality", "depth_results") if key in prepared}}


def load_experiment(raw: bytes) -> tuple[list[Sample], dict, dict]:
    if len(raw) > 220_000_000:
        raise ValueError("Experiment ZIP must be 220 MB or smaller.")
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        if sum(info.file_size for info in archive.infolist()) > 550_000_000:
            raise ValueError("Expanded experiment size exceeds the safety limit.")
        manifest = json.loads(archive.read("run.json"))
        with np.load(BytesIO(archive.read("analysis.npz")), allow_pickle=False) as arrays:
            manifest["_saved_features"] = arrays["features"].copy()
        entries = manifest.get("samples", [])
        if not 1 <= len(entries) <= 240:
            raise ValueError("An experiment must contain between 1 and 240 images.")
        samples, overrides = [], {}
        for entry in entries:
            sid = entry["id"]
            if not isinstance(sid, str) or not sid.isalnum() or len(sid) > 64:
                raise ValueError("Invalid image ID in experiment archive.")
            decoded = decode_image(archive.read(f"images/{sid}.png"), entry.get("metadata", {}))
            sample = Sample(sid, decoded.image, decoded.metadata)
            samples.append(sample)
            with Image.open(BytesIO(archive.read(f"masks/raw/{sid}.png"))) as image:
                overrides[sid] = np.asarray(image.convert("L")) > 127
        if manifest.get("depth_estimation"):
            from .depth import decode_depth
            manifest["_saved_depth"] = {}
            for sample in samples:
                depth = decode_depth(archive.read(f"depth/{sample.id}.npz"))
                if depth["metadata"].get("model_id") != manifest["depth_estimation"]["model_id"]:
                    raise ValueError("Experiment contains mixed depth models.")
                manifest["_saved_depth"][sample.id] = depth
        if manifest.get("selected_depth"):
            from .depth import decode_depth
            manifest["_selected_depth"] = decode_depth(archive.read("selected_depth.npz"))
    return samples, overrides, manifest
