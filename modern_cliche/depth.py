"""Embedded monocular inference. Original image context is kept during inference.

MoGe-2 provides camera-space points; Depth Anything provides relative disparity.
Neither method observes the back of an object. No silhouette-radius fallback.
"""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
import zipfile
import numpy as np
from PIL import Image

MODELS = {
    "moge-small": ("Ruicheng/moge-2-vits-normal", "26b477f41595707c5db6770294c0d1721e8ed4ed"),
    "moge-base": ("Ruicheng/moge-2-vitb-normal", "ca5f0e07ff01d3e5a364c1d954ed12ee1814b368"),
    "moge-large": ("Ruicheng/moge-2-vitl-normal", "cb0e8bbd6b1e243589717c78e750b1ba4c093acf"),
    "depth-anything": ("depth-anything/Depth-Anything-V2-Small-hf", "5426e4f0f36572d16453bbda7a8389317b1bef99"),
}
DEPTH_NAMES = ["depth_extent_ratio", "depth_spread_ratio", "depth_gradient", "depth_edge_fraction",
               "surface_tilt", "valid_depth_fraction"]
DEPTH_LABELS = ["깊이 / 가로·세로 범위",
                "깊이 분포 폭 / 중앙 깊이", "상대 깊이 기울기", "깊이 불연속 비율",
                "표면 기울기", "유효 깊이 비율"]
_LOCK = threading.RLock()


def moge_available():
    return importlib.util.find_spec("moge") is not None


def default_model():
    return "moge-base" if moge_available() else "depth-anything"


def backproject(depth, intrinsics):
    h, w = depth.shape
    y, x = np.mgrid[:h, :w]
    u, v = (x + .5) / w, (y + .5) / h
    return np.stack([(u - intrinsics[0, 2]) / intrinsics[0, 0] * depth,
                     (v - intrinsics[1, 2]) / intrinsics[1, 1] * depth, depth], axis=-1).astype(np.float32)


@lru_cache(maxsize=1)
def load_depth_model(kind, device):
    import torch
    name, revision = MODELS[kind]
    torch.set_num_threads(min(4, os.cpu_count() or 2))
    if kind.startswith("moge"):
        if not moge_available():
            raise RuntimeError("MoGe 설치가 필요합니다: python -m pip install -r requirements-moge.txt")
        from moge.model.v2 import MoGeModel
        model = MoGeModel.from_pretrained(name, revision=revision).to(device).eval()
        model.enable_pytorch_native_sdpa()
        return None, model
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(name, revision=revision, use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained(name, revision=revision).to(device).eval()
    return processor, model


def depth_features(result):
    valid, depth, points = result["valid"], result["depth"], result["points"]
    if valid.sum() < 16:
        raise ValueError("유효한 대상 깊이가 부족합니다. 대상 마스크와 원본 이미지를 검토하십시오.")
    q = np.quantile(depth[valid], [.05, .5, .95])
    extent = np.quantile(points[valid], .95, axis=0) - np.quantile(points[valid], .05, axis=0)
    differences, slopes = [], []
    for axis in (0, 1):
        a, b = (slice(None, -1), slice(1, None))
        d0, d1 = (depth[a, :], depth[b, :]) if axis == 0 else (depth[:, a], depth[:, b])
        v0, v1 = (valid[a, :], valid[b, :]) if axis == 0 else (valid[:, a], valid[:, b])
        pair = v0 & v1
        delta = np.abs(d1 - d0)[pair] / np.maximum(1e-8, np.minimum(d0, d1)[pair])
        differences.extend(delta.tolist())
        p0, p1 = (points[a, :], points[b, :]) if axis == 0 else (points[:, a], points[:, b])
        vector = p1[pair] - p0[pair]
        slopes.extend((np.abs(vector[:, 2]) / np.maximum(1e-8, np.linalg.norm(vector, axis=1))).tolist())
    return np.array([extent[2] / max(1e-8, max(extent[:2])),
                     (q[2] - q[0]) / max(1e-8, q[1]), np.median(differences) if differences else 0.,
                     np.mean(np.asarray(differences) > .03) if differences else 0.,
                     np.median(slopes) if slopes else 0., valid.sum() / max(1, result["foreground"].sum())])


def infer_depth(image, mask, kind=None, max_size=768, resolution_level=9, device="auto", cache=True):
    """Return finite arrays in source-image coordinates (possibly downsampled).

    Disk caching keys include RGB, reviewed mask, model revision and all settings.
    Cache/model failures propagate; partial results never enter a mixed cohort.
    """
    import torch
    kind = kind or default_model()
    if kind not in MODELS or not 128 <= max_size <= 768 or not 0 <= resolution_level <= 9:
        raise ValueError("깊이 모델 또는 추론 해상도 설정이 올바르지 않습니다.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Transparent cutouts have no scene context; hidden RGB bytes must not become a scene.
    rgba = image.convert("RGBA")
    rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, (127, 127, 127, 255)), rgba).convert("RGB")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (rgb.height, rgb.width) or mask.sum() < 16:
        raise ValueError("깊이 분석에는 원본 크기의 검토된 대상 마스크가 필요합니다.")
    settings = dict(model=kind, model_id=MODELS[kind][0], revision=MODELS[kind][1],
                    max_size=int(max_size), resolution_level=int(resolution_level), device=device, algorithm=2)
    key = sha256(rgb.tobytes() + mask.tobytes() + json.dumps(settings, sort_keys=True).encode()).hexdigest()
    root = Path(os.environ.get("MODERN_CLICHE_DEPTH_CACHE", Path.home() / ".cache" / "modern_cliche" / "depth"))
    path = root / f"{key}.npz"
    if cache and path.exists():
        try:
            return decode_depth(path.read_bytes())
        except (ValueError, OSError, KeyError, EOFError, zipfile.BadZipFile):
            path.unlink(missing_ok=True)
    started = time.perf_counter()
    rgb.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    foreground = np.asarray(Image.fromarray(mask).resize(rgb.size, Image.Resampling.NEAREST), dtype=bool)
    with _LOCK, torch.inference_mode():
        processor, model = load_depth_model(kind, device)
        if kind.startswith("moge"):
            tensor = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float().div(255).to(device)
            output = model.infer(tensor, resolution_level=resolution_level, use_fp16=device.startswith("cuda"))
            arrays = {k: v.detach().cpu().numpy() for k, v in output.items()}
            depth, points, intrinsics = arrays["depth"], arrays["points"], arrays["intrinsics"]
            valid = arrays["mask"] & foreground
            camera = "model-estimated"
            scale = "model-estimated-metres-unverified"
            extra = {"normal": np.where(valid[..., None], arrays["normal"], 0).astype(np.float32)} if "normal" in arrays else {}
        else:
            inputs = processor(images=rgb, return_tensors="pt").to(device)
            output = model(**inputs)
            disparity = processor.post_process_depth_estimation(output, target_sizes=[(rgb.height, rgb.width)])[0]["predicted_depth"].cpu().numpy()
            lo, hi = np.quantile(disparity[foreground], [.01, .99])
            # Relative disparity has no calibrated inverse-depth scale/offset.
            depth = 1. / (1. + np.clip((disparity - lo) / max(1e-8, hi - lo), 0., 1.))
            f = .5 / np.tan(np.deg2rad(60.) / 2)
            intrinsics = np.array([[f, 0, .5], [0, f * rgb.width / rgb.height, .5], [0, 0, 1]], np.float32)
            points = backproject(depth, intrinsics)
            valid = foreground.copy()
            camera = "assumed-horizontal-fov-60-degrees"
            scale = "relative-disparity-with-assumed-inverse-depth-offset"
            extra = {"disparity": disparity.astype(np.float32)}
    valid &= np.isfinite(depth) & (depth > 0) & np.isfinite(points).all(axis=-1)
    depth = np.where(valid, depth, 0).astype(np.float32)
    points = np.where(valid[..., None], points, 0).astype(np.float32)
    result = dict(depth=depth, points=points, valid=valid, foreground=foreground,
                  rgb=np.asarray(rgb), intrinsics=intrinsics.astype(np.float32),
                  metadata={**settings, "cache_key": key, "camera": camera, "scale": scale,
                            "seconds": time.perf_counter() - started, "hidden_surface": "unobserved"})
    result.update(extra)
    result["features"] = depth_features(result)
    if cache:
        root.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_bytes(encode_depth(result))
        temporary.replace(path)
    return result


def encode_depth(result):
    buffer = BytesIO()
    np.savez_compressed(buffer, **{k: v for k, v in result.items() if k != "metadata"},
                        metadata=np.array(json.dumps(result["metadata"])))
    return buffer.getvalue()


def decode_depth(raw):
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > 32_000_000:
            raise ValueError("저장된 깊이 배열 크기가 제한을 초과했습니다.")
    with np.load(BytesIO(raw), allow_pickle=False) as arrays:
        result = {k: arrays[k].copy() for k in arrays.files if k != "metadata"}
        result["metadata"] = json.loads(str(arrays["metadata"]))
    h, w = result["depth"].shape
    if max(h, w) > 768 or result["points"].shape != (h, w, 3) or result["valid"].shape != (h, w) or result["intrinsics"].shape != (3, 3):
        raise ValueError("저장된 깊이 데이터의 크기가 일치하지 않습니다.")
    valid = result["valid"]
    if valid.dtype != bool or not np.isfinite(result["points"][valid]).all() or not np.isfinite(result["depth"][valid]).all():
        raise ValueError("저장된 깊이 데이터에 유효하지 않은 좌표가 포함되어 있습니다.")
    return result


def infer_cohort(prepared, kind=None, max_size=768, resolution_level=9, device="auto", progress=None):
    results = []
    for i, (sample, mask) in enumerate(zip(prepared["samples"], prepared["raw_masks"])):
        try:
            results.append(infer_depth(sample.image, mask, kind, max_size, resolution_level, device))
        except Exception as exc:
            raise RuntimeError(f"깊이 추정 실패 · {sample.metadata.get('title', sample.id)}: {exc}") from exc
        if progress:
            progress((i + 1) / len(prepared["samples"]))
    return results


def depth_preview(result):
    valid, depth = result["valid"], result["depth"]
    lo, hi = np.quantile(depth[valid], [.02, .98])
    normal = np.clip((depth - lo) / max(1e-8, hi - lo), 0, 1)
    from matplotlib import colormaps
    rgb = (colormaps["viridis"](normal)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 245
    return Image.fromarray(rgb)
