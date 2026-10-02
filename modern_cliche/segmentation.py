from __future__ import annotations

from functools import lru_cache
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import morphology


@lru_cache(maxsize=1)
def foreground_session():
    from rembg import new_session
    return new_session("u2netp", providers=["CPUExecutionProvider"], num_threads=2)


def extract_mask(image: Image.Image, method: str = "u2net", threshold: float = 0.5) -> tuple[np.ndarray, dict]:
    rgba = np.asarray(image.convert("RGBA"))
    alpha = rgba[..., 3]
    used = method
    if alpha.min() < 250:
        mask = alpha > 127
        used = "uploaded-alpha"
    elif method == "u2net":
        from rembg import remove
        result = remove(image.convert("RGB"), session=foreground_session(), only_mask=True)
        mask = np.asarray(result) > threshold * 255
    elif method == "binary":
        # Explicit option for existing black silhouettes, never a hidden photo fallback.
        mask = np.asarray(image.convert("L")) < threshold * 255
    elif method == "grabcut":
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("GrabCut에 필요한 OpenCV를 불러오지 못했습니다. U²-Net 또는 Binary 추출을 사용하십시오.") from exc
        rgb = rgba[..., :3].copy()
        h, w = rgb.shape[:2]
        gc = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        gc[h // 8:7 * h // 8, w // 8:7 * w // 8] = cv2.GC_PR_FGD
        gc[:2, :] = gc[-2:, :] = cv2.GC_BGD
        gc[:, :2] = gc[:, -2:] = cv2.GC_BGD
        cv2.setRNGSeed(17)
        cv2.grabCut(rgb, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
        mask = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD))
    else:
        raise ValueError("지원하지 않는 배경 분리 방법입니다.")
    mask = morphology.remove_small_objects(mask.astype(bool), max_size=max(8, int(mask.size * .0005)))
    labels, count = ndi.label(mask)
    if not count:
        raise ValueError("대상 영역을 추출하지 못했습니다. 분리 방법 또는 마스크를 변경하십시오.")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = labels == sizes.argmax()
    fraction = float(largest.mean())
    if fraction < .005 or fraction > .95:
        raise ValueError(f"추출 영역 비율 {fraction:.1%}: 빈 영역 또는 배경 전체일 가능성이 있습니다.")
    border = np.concatenate([largest[0], largest[-1], largest[:, 0], largest[:, -1]])
    return largest, dict(method=used, foreground_fraction=fraction,
                         discarded_fraction=float((mask.sum() - largest.sum()) / max(1, mask.sum())),
                         border_contact=float(border.mean()), original_components=int(count),
                         needs_review=bool(border.mean() > .08 or largest.sum() < mask.sum() * .8))


def normalise_mask(mask: np.ndarray, size: int = 96) -> np.ndarray:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("빈 마스크는 분석할 수 없습니다.")
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = (size - 12) / max(crop.shape)
    dimensions = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
    resized = np.asarray(Image.fromarray(crop).resize(dimensions, Image.Resampling.NEAREST)).astype(bool)
    canvas = np.zeros((size, size), dtype=bool)
    y, x = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    # Translation/scale normalisation preserves pose, orientation and aspect ratio.
    cy, cx = ndi.center_of_mass(canvas)
    canvas = ndi.shift(canvas.astype(np.uint8), ((size - 1) / 2 - cy, (size - 1) / 2 - cx),
                       order=0, mode="constant").astype(bool)
    canvas[:2] = canvas[-2:] = False
    canvas[:, :2] = canvas[:, -2:] = False
    return canvas


def signed_distance(mask: np.ndarray) -> np.ndarray:
    return (ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)).astype(np.float32)


def overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=float)
    tint = np.array([240, 74, 35])
    rgb[mask] = .55 * rgb[mask] + .45 * tint
    rgb[~mask] *= .42
    return Image.fromarray(rgb.astype(np.uint8))
