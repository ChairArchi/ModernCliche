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
                    raise ValueError("수동 마스크와 원본 이미지 크기가 달라.")
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
    return digest.hexdigest()


def subset(prepared: dict, indices: list[int]) -> dict:
    return {**prepared, **{key: [prepared[key][index] for index in indices]
                          for key in ("samples", "raw_masks", "masks", "quality")}}


def load_experiment(raw: bytes) -> tuple[list[Sample], dict, dict]:
    if len(raw) > 80_000_000:
        raise ValueError("실험 ZIP은 80MB 이하로 올려줘.")
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        if sum(info.file_size for info in archive.infolist()) > 180_000_000:
            raise ValueError("압축 해제 크기가 너무 커.")
        manifest = json.loads(archive.read("run.json"))
        with np.load(BytesIO(archive.read("analysis.npz")), allow_pickle=False) as arrays:
            manifest["_saved_features"] = arrays["features"].copy()
        entries = manifest.get("samples", [])
        if not 1 <= len(entries) <= 80:
            raise ValueError("실험에는 1~80장의 이미지가 필요해.")
        samples, overrides = [], {}
        for entry in entries:
            sid = entry["id"]
            if not isinstance(sid, str) or not sid.isalnum() or len(sid) > 64:
                raise ValueError("잘못된 이미지 ID야.")
            decoded = decode_image(archive.read(f"images/{sid}.png"), entry.get("metadata", {}))
            sample = Sample(sid, decoded.image, decoded.metadata)
            samples.append(sample)
            with Image.open(BytesIO(archive.read(f"masks/raw/{sid}.png"))) as image:
                overrides[sid] = np.asarray(image.convert("L")) > 127
    return samples, overrides, manifest
