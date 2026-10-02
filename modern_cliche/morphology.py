from __future__ import annotations

import numpy as np
import warnings
from PIL import Image
from scipy import ndimage as ndi
from skimage.transform import resize


def _fit_to_canvas(values: np.ndarray, valid: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Crop to the visible object and fit it to a shared square canvas."""
    ys, xs = np.where(valid)
    if len(xs) < 16:
        raise ValueError("Not enough valid surface points for canonicalisation.")
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = values[y0:y1, x0:x1]
    crop_valid = valid[y0:y1, x0:x1]
    inner = max(16, size - 16)
    scale = inner / max(crop.shape[:2])
    h = max(1, int(round(crop.shape[0] * scale)))
    w = max(1, int(round(crop.shape[1] * scale)))
    resized = resize(crop, (h, w), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
    resized_valid = resize(crop_valid.astype(np.float32), (h, w), order=0, preserve_range=True, anti_aliasing=False) > .5
    canvas = np.full((size, size), np.nan, dtype=np.float32)
    canvas_valid = np.zeros((size, size), dtype=bool)
    oy, ox = (size - h) // 2, (size - w) // 2
    canvas[oy:oy + h, ox:ox + w] = resized
    canvas_valid[oy:oy + h, ox:ox + w] = resized_valid
    canvas[~canvas_valid] = np.nan
    return canvas, canvas_valid


def canonical_depth(result: dict, size: int = 160) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(result["points"], dtype=np.float32)
    valid = np.asarray(result["valid"], dtype=bool)
    if points.shape[:2] != valid.shape or points.shape[-1] != 3:
        raise ValueError("Point map and valid mask do not match.")
    p = points[valid]
    if len(p) < 16:
        raise ValueError("Not enough valid point samples.")
    q05, q50, q95 = np.quantile(p, [.05, .5, .95], axis=0)
    xy_span = max(float(q95[0] - q05[0]), float(q95[1] - q05[1]), 1e-6)
    relative = (points[..., 2] - float(q50[2])) / xy_span
    relative = np.clip(relative, -1.5, 1.5)
    return _fit_to_canvas(relative, valid, size)


def aggregate_depth_results(results: list[dict], size: int = 160, coverage_threshold: float = .45) -> dict:
    if not 2 <= len(results) <= 24:
        raise ValueError("Statistical reconstruction needs 2 to 24 specimens.")
    if not 64 <= size <= 384:
        raise ValueError("Canonical surface resolution must be between 64 and 384.")
    if not .1 <= coverage_threshold <= .95:
        raise ValueError("Coverage threshold must be between 0.10 and 0.95.")

    fields, masks = [], []
    for result in results:
        field, mask = canonical_depth(result, size)
        fields.append(field)
        masks.append(mask)
    stack = np.stack(fields)
    support = np.stack(masks)
    count = support.sum(axis=0)
    coverage = count.astype(np.float32) / len(results)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(stack, axis=0)
        q25 = np.nanpercentile(stack, 25, axis=0)
        q75 = np.nanpercentile(stack, 75, axis=0)
    variation = (q75 - q25).astype(np.float32)
    valid = (coverage >= coverage_threshold) & np.isfinite(median)
    closed = ndi.binary_closing(valid, structure=np.ones((3, 3), dtype=bool))
    valid = closed & (coverage >= max(.1, coverage_threshold * .75)) & np.isfinite(median)
    if valid.sum() < 64:
        raise ValueError("The selected specimens do not share enough visible surface. Try a lower coverage threshold or a tighter cluster.")

    y, x = np.mgrid[:size, :size]
    x = (x + .5 - size / 2) / size * 2
    y = (y + .5 - size / 2) / size * 2
    relief = np.where(np.isfinite(median), median, 0.).astype(np.float32)
    if valid.any():
        relief -= float(np.median(relief[valid]))
    depth = (2.0 + relief).astype(np.float32)
    points = np.stack([x * depth, y * depth, depth], axis=-1).astype(np.float32)
    points[~valid] = 0
    depth = np.where(valid, depth, 0).astype(np.float32)

    variation = np.where(np.isfinite(variation), variation, 0).astype(np.float32)
    scaled_var = np.zeros_like(variation)
    finite_var = variation[valid & np.isfinite(variation)]
    if finite_var.size:
        hi = max(float(np.quantile(finite_var, .95)), 1e-6)
        scaled_var = np.clip(variation / hi, 0, 1)
    grey = (205 - 55 * scaled_var).astype(np.uint8)
    rgb = np.repeat(grey[..., None], 3, axis=2)
    rgb[~valid] = 245

    return dict(
        depth=depth,
        points=points,
        valid=valid,
        foreground=valid.copy(),
        rgb=rgb,
        intrinsics=np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]], dtype=np.float32),
        coverage=coverage,
        variation=variation,
        metadata=dict(
            model="statistical-visible-surface",
            model_id="modern-cliche/canonical-median-point-map",
            revision="1",
            scale="canonical-relative",
            camera="cluster-canonicalised-camera-facing-pose",
            specimens=len(results),
            coverage_threshold=float(coverage_threshold),
            hidden_surface="unobserved",
            aggregation="median-depth-with-IQR-variation",
        ),
    )


def coverage_preview(result: dict) -> Image.Image:
    coverage = np.asarray(result["coverage"], dtype=float)
    grey = (255 - np.clip(coverage, 0, 1) * 210).astype(np.uint8)
    return Image.fromarray(grey, mode="L")


def variation_preview(result: dict) -> Image.Image:
    variation = np.asarray(result["variation"], dtype=float)
    valid = np.asarray(result["valid"], dtype=bool)
    output = np.full(variation.shape, 245, dtype=np.uint8)
    values = variation[valid]
    if values.size:
        hi = max(float(np.quantile(values, .95)), 1e-8)
        output[valid] = (30 + 200 * np.clip(variation[valid] / hi, 0, 1)).astype(np.uint8)
    return Image.fromarray(output, mode="L")
