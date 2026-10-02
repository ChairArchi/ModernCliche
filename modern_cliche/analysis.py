from __future__ import annotations

from functools import lru_cache
import importlib.util
import warnings
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from skimage.measure import regionprops, perimeter, euler_number
from skimage.morphology import medial_axis

from .segmentation import signed_distance

FEATURE_NAMES = [
    "aspect_ratio", "occupancy", "solidity", "eccentricity", "horizontal_symmetry",
    "vertical_symmetry", "compactness", "holes", "skeleton_endpoints",
    "branch_regions", "mean_radius", "radius_variation",
]
FEATURE_LABELS = [
    "width / height", "foreground occupancy", "convex solidity", "elongation",
    "left-right symmetry", "top-bottom symmetry", "boundary complexity", "holes",
    "skeleton endpoints", "skeleton branch regions", "mean internal radius", "radius variation",
]


def field_validation(fields: np.ndarray, seed=17) -> dict:
    n = len(fields)
    if n < 8:
        return dict(available=False, reason="At least 8 images are required for holdout reconstruction.")
    flat = fields.reshape(n, -1)
    rng, rows = np.random.default_rng(seed), []
    with threadpool_limits(limits=2):
        for repeat in range(5):
            order = rng.permutation(n)
            split = max(4, round(n * .75))
            train, test = order[:split], order[split:]
            pca = PCA(n_components=min(6, len(train) - 1), svd_solver="full").fit(flat[train])
            rebuilt = pca.inverse_transform(pca.transform(flat[test]))
            baseline = np.broadcast_to(pca.mean_, rebuilt.shape)
            truth = flat[test]
            iou = lambda pred: np.count_nonzero((pred > 0) & (truth > 0), axis=1) / np.maximum(
                1, np.count_nonzero((pred > 0) | (truth > 0), axis=1)
            )
            rows.append(dict(
                repeat=repeat + 1,
                train_n=len(train),
                test_n=len(test),
                field_rmse=float(np.sqrt(np.mean((rebuilt - truth) ** 2))),
                mean_baseline_rmse=float(np.sqrt(np.mean((baseline - truth) ** 2))),
                mask_iou=float(iou(rebuilt).mean()),
                mean_baseline_iou=float(iou(baseline).mean()),
            ))
    return dict(
        available=True,
        repeats=rows,
        mean_iou=float(np.mean([row["mask_iou"] for row in rows])),
        mean_baseline_iou=float(np.mean([row["mean_baseline_iou"] for row in rows])),
        interpretation="Within-sample silhouette reconstruction only; not a measure of recognition or 3D accuracy.",
    )


def shape_features(mask: np.ndarray) -> np.ndarray:
    props = regionprops(mask.astype(np.uint8))[0]
    y0, x0, y1, x1 = props.bbox
    skeleton, distance = medial_axis(mask, return_distance=True, rng=17)
    degree = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) - skeleton
    endpoints = np.count_nonzero(skeleton & (degree == 1))
    _, branches = ndi.label(skeleton & (degree >= 3))
    radii = distance[skeleton]
    symmetry = lambda other: np.count_nonzero(mask & other) / max(1, np.count_nonzero(mask | other))
    return np.array([
        (x1 - x0) / max(1, y1 - y0), mask.mean(), props.solidity, props.eccentricity,
        symmetry(np.fliplr(mask)), symmetry(np.flipud(mask)),
        perimeter(mask) ** 2 / max(1, 4 * np.pi * mask.sum()),
        max(0, 1 - euler_number(mask, connectivity=2)), endpoints, branches,
        np.mean(radii) / mask.shape[0], np.std(radii) / max(.1, np.mean(radii)),
    ], dtype=float)


def fit_field_model(masks: list[np.ndarray]) -> dict:
    fields = np.stack([signed_distance(mask) for mask in masks])
    flat = fields.reshape(len(fields), -1)
    if len(fields) < 2 or np.max(np.std(flat, axis=0)) < 1e-6:
        return dict(
            mean=flat.mean(axis=0).reshape(masks[0].shape),
            components=np.empty((0, *masks[0].shape)),
            scores=np.empty((len(fields), 0)), std=np.empty(0), explained=np.empty(0),
            retained_variance=0.0, fields=fields,
        )
    pca = PCA(n_components=min(len(fields) - 1, 12), svd_solver="full")
    scores = pca.fit_transform(flat)
    valid = pca.explained_variance_ > 1e-8
    return dict(
        mean=pca.mean_.reshape(masks[0].shape),
        components=pca.components_[valid].reshape(-1, *masks[0].shape),
        scores=scores[:, valid], std=np.sqrt(pca.explained_variance_[valid]),
        explained=pca.explained_variance_ratio_[valid],
        retained_variance=float(pca.explained_variance_ratio_[valid].sum()), fields=fields,
    )


@lru_cache(maxsize=2)
def load_visual_model(kind: str):
    import torch
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    if kind == "dino":
        name = "facebook/dinov2-small"
        return AutoImageProcessor.from_pretrained(name, use_fast=False), AutoModel.from_pretrained(name).eval()
    name = "openai/clip-vit-base-patch32"
    return CLIPProcessor.from_pretrained(name, use_fast=False), CLIPModel.from_pretrained(name).eval()


def ml_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


def visual_embeddings(images, masks, kind="dino", query=""):
    import torch
    from PIL import Image
    processor, model = load_visual_model(kind)
    torch.set_num_threads(2)
    crops = []
    for image, mask in zip(images, masks):
        rgb = np.asarray(image.convert("RGB")).copy()
        rgb[~mask] = 127
        ys, xs = np.where(mask)
        crops.append(Image.fromarray(rgb).crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)))
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(crops), 4):
            batch = processor(images=crops[start:start + 4], return_tensors="pt")
            if kind == "dino":
                vector = model(**batch).last_hidden_state[:, 0]
            else:
                vector = model.get_image_features(**batch)
            vectors.append(vector.detach().cpu().numpy())
        matrix = np.concatenate(vectors)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
        if kind == "clip" and query:
            tokens = processor(text=[query], return_tensors="pt", padding=True)
            text = model.get_text_features(**tokens).detach().cpu().numpy()[0]
            text /= max(1e-8, np.linalg.norm(text))
            return matrix, matrix @ text
    return matrix, None


def _manual_kmeans(matrix: np.ndarray, seed: int, k: int) -> dict:
    n = len(matrix)
    with threadpool_limits(limits=2):
        km = KMeans(k, n_init=16, random_state=seed).fit(matrix)
    labels = km.labels_
    if len(np.unique(labels)) != k or min(np.bincount(labels)) < 2:
        raise ValueError("The requested cluster count does not produce usable groups.")
    silhouette = float(silhouette_score(matrix, labels))
    rng, repeats = np.random.default_rng(seed), []
    for trial in range(8):
        take = rng.choice(n, max(k + 1, round(n * .8)), replace=False)
        with threadpool_limits(limits=2):
            sub = KMeans(k, n_init=6, random_state=seed + trial + 1).fit(matrix[take])
        repeats.append(adjusted_rand_score(labels[take], sub.labels_))
    stability = float(np.mean(repeats))
    return dict(
        labels=labels,
        candidates=[dict(method="kmeans", k=k, silhouette=silhouette, stability=stability,
                         coverage=1.0, outliers=0, minimum_cluster_size=int(min(np.bincount(labels))))],
        silhouette=silhouette,
        stability=stability,
        confidence=np.ones(n),
        sample_silhouette=silhouette_samples(matrix, labels),
        outliers=[],
        reason="Manual K-Means cluster count.",
    )


def choose_clusters(matrix: np.ndarray, seed=17, requested_k=0) -> dict:
    n = len(matrix)
    empty = dict(
        labels=np.zeros(n, dtype=int), candidates=[], silhouette=None, stability=None,
        confidence=np.ones(n), sample_silhouette=np.zeros(n), outliers=[],
        reason="No stable subgroup structure detected; the cohort is retained as one morphology.",
    )
    if n < 4 or np.max(np.std(matrix, axis=0)) < 1e-7:
        return empty
    if requested_k:
        if not 2 <= requested_k < n:
            raise ValueError("Cluster count must be between 2 and N-1.")
        return _manual_kmeans(matrix, seed, requested_k)

    trials = []
    fractions = (.05, .08, .12, .16) if n >= 24 else (.18, .25, .33)
    for fraction in fractions:
        min_cluster_size = max(3, min(n - 1, int(round(n * fraction))))
        min_samples = max(2, min_cluster_size // 2)
        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            allow_single_cluster=False,
        ).fit(matrix)
        labels = model.labels_.copy()
        core = labels >= 0
        unique = np.unique(labels[core])
        if len(unique) < 2 or core.sum() < max(6, int(round(n * .35))):
            continue
        counts = [int(np.sum(labels == label)) for label in unique]
        if min(counts) < 2:
            continue
        silhouette = float(silhouette_score(matrix[core], labels[core]))
        probabilities = np.asarray(getattr(model, "probabilities_", np.where(core, 1., 0.)), dtype=float)
        confidence = float(probabilities[core].mean()) if core.any() else 0.
        coverage = float(core.mean())
        objective = .55 * max(-1., silhouette) + .25 * coverage + .20 * confidence
        trials.append(dict(
            objective=objective, labels=labels, probabilities=probabilities, method="hdbscan",
            k=len(unique), silhouette=silhouette, stability=confidence, coverage=coverage,
            outliers=int((~core).sum()), minimum_cluster_size=min(counts),
            min_cluster_size=min_cluster_size, min_samples=min_samples,
        ))
    if not trials:
        return empty
    selected = max(trials, key=lambda row: row["objective"])
    if selected["silhouette"] < .03 or selected["coverage"] < .4:
        empty["candidates"] = [
            {k: v for k, v in row.items() if k not in ("labels", "probabilities", "objective")} for row in trials
        ]
        return empty

    labels = selected["labels"]
    core = labels >= 0
    sample_scores = np.full(n, -1., dtype=float)
    if len(np.unique(labels[core])) >= 2:
        sample_scores[core] = silhouette_samples(matrix[core], labels[core])
    return dict(
        labels=labels,
        candidates=[{k: v for k, v in row.items() if k not in ("labels", "probabilities", "objective")} for row in trials],
        silhouette=selected["silhouette"], stability=selected["stability"],
        confidence=selected["probabilities"], sample_silhouette=sample_scores,
        outliers=np.flatnonzero(~core).tolist(),
        reason="Automatic HDBSCAN density clustering. Noise points are excluded from morphology groups.",
    )


def _balanced_visual_block(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=float)
    n = len(embeddings)
    if n < 3 or np.max(np.std(embeddings, axis=0)) < 1e-9:
        return np.zeros((n, 0), dtype=float)
    components = min(16, n - 1, embeddings.shape[1])
    reduced = PCA(n_components=components, svd_solver="full").fit_transform(embeddings)
    reduced = StandardScaler().fit_transform(reduced)
    return reduced / np.sqrt(max(1, reduced.shape[1]))


def analyse(masks: list[np.ndarray], seed=17, requested_k=0, embeddings=None, backend="shape", depth_results=None) -> dict:
    if len(masks) < 3:
        raise ValueError("At least 3 distinct images are required; 20 or more are recommended.")
    raw = np.stack([shape_features(mask) for mask in masks])
    standard = StandardScaler().fit_transform(raw)
    model = fit_field_model(masks)
    scores = model["scores"]

    spatial = scores[:, :6]
    if spatial.size:
        spatial = StandardScaler().fit_transform(spatial) / np.sqrt(max(1, spatial.shape[1]))
    else:
        spatial = np.zeros((len(masks), 0), dtype=float)
    shape_block = standard / np.sqrt(raw.shape[1])
    blocks = [shape_block, spatial]

    if embeddings is not None:
        if np.asarray(embeddings).shape[0] != len(masks):
            raise ValueError("Visual embedding count does not match image count.")
        visual = _balanced_visual_block(np.asarray(embeddings, dtype=float))
        if visual.size:
            blocks.insert(0, visual)
        backend = f"{backend}+geometry"

    depth_raw, depth_standard = None, None
    if depth_results is not None:
        from .depth import DEPTH_NAMES, DEPTH_LABELS, depth_features
        if len(depth_results) != len(masks):
            raise ValueError("Depth result count does not match image count.")
        identities = {(r["metadata"]["model_id"], r["metadata"]["revision"], r["metadata"]["scale"]) for r in depth_results}
        if len(identities) != 1:
            raise ValueError("All images must use the same depth model and scale.")
        depth_raw = np.stack([depth_features(r) for r in depth_results])
        depth_standard = StandardScaler().fit_transform(depth_raw)
        blocks.append(depth_standard / np.sqrt(len(DEPTH_NAMES)))
        backend += "+depth"

    matrix = np.concatenate([block for block in blocks if block.shape[1]], axis=1)
    clusters = choose_clusters(matrix, seed, requested_k)
    labels = clusters["labels"]
    groups = []
    for label in np.unique(labels):
        if label < 0:
            continue
        indices = np.flatnonzero(labels == label)
        pairwise = cdist(matrix[indices], matrix[indices])
        medoid = int(indices[np.argmin(pairwise.sum(axis=1))])
        differences = standard[indices].mean(axis=0)
        rules = [dict(
            feature=FEATURE_NAMES[j], label=FEATURE_LABELS[j],
            standardised_difference=float(differences[j]),
            group_mean=float(raw[indices, j].mean()), dataset_mean=float(raw[:, j].mean()),
        ) for j in np.argsort(np.abs(differences))[-3:][::-1]]
        group = dict(
            label=int(label), indices=indices.tolist(), medoid=medoid,
            descriptive_rules=rules, model=fit_field_model([masks[i] for i in indices]),
        )
        if depth_raw is not None:
            differences = depth_standard[indices].mean(axis=0)
            group["depth_rules"] = [dict(
                feature=DEPTH_NAMES[j], label=DEPTH_LABELS[j],
                standardised_difference=float(differences[j]),
                group_mean=float(depth_raw[indices, j].mean()), dataset_mean=float(depth_raw[:, j].mean()),
            ) for j in np.argsort(np.abs(differences))[-3:][::-1]]
        groups.append(group)

    if not groups:
        labels = np.zeros(len(masks), dtype=int)
        clusters["labels"] = labels
        clusters["outliers"] = []
        clusters["confidence"] = np.ones(len(masks))
        indices = np.arange(len(masks))
        pairwise = cdist(matrix, matrix)
        medoid = int(np.argmin(pairwise.sum(axis=1)))
        groups = [dict(label=0, indices=indices.tolist(), medoid=medoid, descriptive_rules=[], model=fit_field_model(masks))]

    if np.max(np.std(matrix, axis=0)) < 1e-8:
        projection = np.zeros((len(matrix), 2))
    else:
        projection = PCA(n_components=min(2, matrix.shape[1], len(matrix) - 1), svd_solver="full").fit_transform(matrix)
    if projection.shape[1] < 2:
        projection = np.pad(projection, ((0, 0), (0, 2 - projection.shape[1])))

    distance = np.zeros(len(masks), dtype=float)
    centers = []
    for group in groups:
        indices = group["indices"]
        center = matrix[indices].mean(axis=0)
        centers.append(center)
        distance[indices] = np.linalg.norm(matrix[indices] - center, axis=1)
    outliers = np.flatnonzero(labels < 0)
    if len(outliers) and centers:
        distance[outliers] = cdist(matrix[outliers], np.stack(centers)).min(axis=1)

    correlations = []
    for axis in range(min(4, scores.shape[1])):
        values = []
        for j in range(raw.shape[1]):
            correlation = float(np.corrcoef(scores[:, axis], raw[:, j])[0, 1]) if raw[:, j].std() > 1e-8 else 0.
            values.append(correlation)
        order = np.argsort(np.abs(values))[-3:][::-1]
        correlations.append([dict(label=FEATURE_LABELS[j], correlation=values[j]) for j in order])

    return dict(
        raw_features=raw, features=matrix, labels=labels, groups=groups, model=model,
        projection=projection, distance=distance, clusters=clusters, backend=backend,
        correlations=correlations, seed=seed, depth_features=depth_raw,
        validation=field_validation(model["fields"], seed),
    )


def synthesize_field(analysis: dict, mode: str, group_a=0, group_b=0, mix=.5,
                     coefficients=None, specimen=None) -> tuple[np.ndarray, dict]:
    a = analysis["groups"][group_a]
    if mode == "observed":
        index = a["medoid"] if specimen is None else specimen
        field = analysis["model"]["fields"][index].copy()
        recipe = dict(mode=mode, specimen=int(index), input_indices=[int(index)])
    elif mode == "interpolate":
        b = analysis["groups"][group_b]
        field = (1 - mix) * a["model"]["mean"] + mix * b["model"]["mean"]
        recipe = dict(mode=mode, group_a=group_a, group_b=group_b, mix=float(mix),
                      input_indices=sorted(set(a["indices"] + b["indices"])))
    elif mode == "variation":
        local = a["model"]
        coefficients = list(coefficients or [])[:len(local["components"])]
        field = local["mean"].copy()
        for index, value in enumerate(coefficients):
            field += float(value) * local["std"][index] * local["components"][index]
        recipe = dict(mode=mode, group_a=group_a, coefficients_sigma=coefficients,
                      input_indices=a["indices"], extrapolation=any(abs(value) > 2 for value in coefficients))
    else:
        raise ValueError("Unsupported morphology synthesis mode.")
    mask = field > 0
    mask[:2] = mask[-2:] = False
    mask[:, :2] = mask[:, -2:] = False
    if mask.sum() < 16:
        raise ValueError("This parameter set does not produce a usable morphology.")
    recipe["components_2d"] = int(ndi.label(mask)[1])
    return mask, recipe
