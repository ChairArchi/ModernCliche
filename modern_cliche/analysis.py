from __future__ import annotations

from functools import lru_cache
import importlib.util
import warnings
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from skimage.measure import regionprops, perimeter, euler_number
from skimage.morphology import medial_axis

from .segmentation import signed_distance

FEATURE_NAMES = ["aspect_ratio", "occupancy", "solidity", "eccentricity", "horizontal_symmetry",
                 "vertical_symmetry", "compactness", "holes", "skeleton_endpoints",
                 "branch_regions", "mean_radius", "radius_variation"]
FEATURE_LABELS = ["가로/세로 비율", "영역 점유율", "볼록 외곽 대비 충실도", "길쭉한 정도", "좌우 대칭도",
                  "상하 대칭도", "둘레의 복잡도", "구멍 수", "중심선 끝점 수", "분기 영역 수",
                  "평균 내부 반경", "내부 반경 변동"]


def field_validation(fields: np.ndarray, seed=17) -> dict:
    """Repeated holdout reconstruction, compared with a training-mean baseline."""
    n = len(fields)
    if n < 8:
        return dict(available=False, reason="홀드아웃 비교에는 최소 8장이 필요해.")
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
            iou = lambda pred: np.count_nonzero((pred > 0) & (truth > 0), axis=1) / np.maximum(1, np.count_nonzero((pred > 0) | (truth > 0), axis=1))
            rows.append(dict(repeat=repeat + 1, train_n=len(train), test_n=len(test),
                             field_rmse=float(np.sqrt(np.mean((rebuilt - truth) ** 2))),
                             mean_baseline_rmse=float(np.sqrt(np.mean((baseline - truth) ** 2))),
                             mask_iou=float(iou(rebuilt).mean()), mean_baseline_iou=float(iou(baseline).mean())))
    return dict(available=True, repeats=rows, mean_iou=float(np.mean([row["mask_iou"] for row in rows])),
                mean_baseline_iou=float(np.mean([row["mean_baseline_iou"] for row in rows])),
                interpretation="이미지 표본 내 재구성 비교야. 대상 인식·3D 정확도·외부 데이터 일반화를 측정하지 않아.")


def shape_features(mask: np.ndarray) -> np.ndarray:
    props = regionprops(mask.astype(np.uint8))[0]
    y0, x0, y1, x1 = props.bbox
    skeleton, distance = medial_axis(mask, return_distance=True, rng=17)
    degree = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) - skeleton
    endpoints = np.count_nonzero(skeleton & (degree == 1))
    _, branches = ndi.label(skeleton & (degree >= 3))
    radii = distance[skeleton]
    symmetry = lambda other: np.count_nonzero(mask & other) / max(1, np.count_nonzero(mask | other))
    return np.array([(x1 - x0) / max(1, y1 - y0), mask.mean(), props.solidity, props.eccentricity,
                     symmetry(np.fliplr(mask)), symmetry(np.flipud(mask)),
                     perimeter(mask) ** 2 / max(1, 4 * np.pi * mask.sum()),
                     max(0, 1 - euler_number(mask, connectivity=2)), endpoints, branches,
                     np.mean(radii) / mask.shape[0], np.std(radii) / max(.1, np.mean(radii))], dtype=float)


def fit_field_model(masks: list[np.ndarray]) -> dict:
    fields = np.stack([signed_distance(mask) for mask in masks])
    flat = fields.reshape(len(fields), -1)
    if len(fields) < 2 or np.max(np.std(flat, axis=0)) < 1e-6:
        return dict(mean=flat.mean(axis=0).reshape(masks[0].shape),
                    components=np.empty((0, *masks[0].shape)), scores=np.empty((len(fields), 0)),
                    std=np.empty(0), explained=np.empty(0), retained_variance=0.0, fields=fields)
    pca = PCA(n_components=min(len(fields) - 1, 12), svd_solver="full")
    scores = pca.fit_transform(flat)
    valid = pca.explained_variance_ > 1e-8
    return dict(mean=pca.mean_.reshape(masks[0].shape),
                components=pca.components_[valid].reshape(-1, *masks[0].shape),
                scores=scores[:, valid], std=np.sqrt(pca.explained_variance_[valid]),
                explained=pca.explained_variance_ratio_[valid],
                retained_variance=float(pca.explained_variance_ratio_[valid].sum()), fields=fields)


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
    # Foreground-only crops prevent a scene/background from becoming the dominant grouping cue.
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


def choose_clusters(matrix: np.ndarray, seed=17, requested_k=0) -> dict:
    n = len(matrix)
    empty = dict(labels=np.zeros(n, dtype=int), candidates=[], silhouette=None, stability=None,
                 sample_silhouette=np.zeros(n), reason="표본 수 또는 변형이 부족해서 단일 그룹으로 유지했어.")
    if n < 4 or np.max(np.std(matrix, axis=0)) < 1e-7:
        return empty
    k_values = [requested_k] if requested_k else range(2, min(6, n // 2) + 1)
    candidates, partitions = [], {}
    rng = np.random.default_rng(seed)
    with threadpool_limits(limits=2):
        for k in k_values:
            if not 2 <= k < n:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(k, n_init=12, random_state=seed).fit(matrix)
            labels = km.labels_
            if len(np.unique(labels)) != k or min(np.bincount(labels)) < 2:
                continue
            silhouette = float(silhouette_score(matrix, labels))
            repeats = []
            for trial in range(8):
                take = rng.choice(n, max(k + 1, round(n * .8)), replace=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sub = KMeans(k, n_init=5, random_state=seed + trial + 1).fit(matrix[take])
                repeats.append(adjusted_rand_score(labels[take], sub.labels_))
            stability = float(np.mean(repeats))
            candidates.append(dict(k=k, silhouette=silhouette, subsample_ari=stability,
                                   minimum_cluster_size=int(min(np.bincount(labels)))))
            partitions[k] = labels
    if not candidates:
        return empty
    selected = max(candidates, key=lambda row: row["silhouette"])
    # Explicit exploratory thresholds, not statistical significance or automatic semantic labels.
    if not requested_k and (selected["silhouette"] < .18 or selected["subsample_ari"] < .35):
        empty.update(candidates=candidates, reason="분리도/반복 안정성이 약해서 단일 그룹으로 유지했어.")
        return empty
    labels = partitions[selected["k"]]
    return dict(labels=labels, candidates=candidates, silhouette=selected["silhouette"],
                stability=selected["subsample_ari"], sample_silhouette=silhouette_samples(matrix, labels),
                reason="수동 그룹 수" if requested_k else "실루엣 점수로 선택한 탐색적 그룹")


def analyse(masks: list[np.ndarray], seed=17, requested_k=0, embeddings=None, backend="shape") -> dict:
    if len(masks) < 3:
        raise ValueError("서로 다른 이미지가 최소 3장 필요해. 20장 이상이면 비교하기 좋아.")
    raw = np.stack([shape_features(mask) for mask in masks])
    standard = StandardScaler().fit_transform(raw)
    model = fit_field_model(masks)
    scores = model["scores"]
    if embeddings is not None:
        matrix = np.asarray(embeddings, dtype=float)
        if matrix.shape[0] != len(masks):
            raise ValueError("특징 벡터와 이미지 수가 달라.")
    else:
        # Equal total weight to geometry descriptors and spatial signed-distance variation.
        spatial = scores[:, :6]
        spatial = spatial / max(1e-8, np.sqrt(np.mean(spatial ** 2)))
        matrix = np.concatenate([standard / np.sqrt(raw.shape[1]), spatial / np.sqrt(max(1, spatial.shape[1]))], axis=1)
    clusters = choose_clusters(matrix, seed, requested_k)
    labels = clusters["labels"]
    groups = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        pairwise = cdist(matrix[indices], matrix[indices])
        medoid = int(indices[np.argmin(pairwise.sum(axis=1))])
        differences = standard[indices].mean(axis=0)
        rules = [dict(feature=FEATURE_NAMES[j], label=FEATURE_LABELS[j],
                      standardised_difference=float(differences[j]),
                      group_mean=float(raw[indices, j].mean()), dataset_mean=float(raw[:, j].mean()))
                 for j in np.argsort(np.abs(differences))[-3:][::-1]]
        groups.append(dict(label=int(label), indices=indices.tolist(), medoid=medoid,
                           descriptive_rules=rules, model=fit_field_model([masks[i] for i in indices])))
    if np.max(np.std(matrix, axis=0)) < 1e-8:
        projection = np.zeros((len(matrix), 2))
    else:
        projection = PCA(n_components=min(2, matrix.shape[1], len(matrix) - 1), svd_solver="full").fit_transform(matrix)
    if projection.shape[1] < 2:
        projection = np.pad(projection, ((0, 0), (0, 2 - projection.shape[1])))
    distance = np.zeros(len(masks))
    for group in groups:
        indices = group["indices"]
        distance[indices] = np.linalg.norm(matrix[indices] - matrix[indices].mean(axis=0), axis=1)
    correlations = []
    for axis in range(min(4, scores.shape[1])):
        values = []
        for j in range(raw.shape[1]):
            correlation = float(np.corrcoef(scores[:, axis], raw[:, j])[0, 1]) if raw[:, j].std() > 1e-8 else 0.
            values.append(correlation)
        order = np.argsort(np.abs(values))[-3:][::-1]
        correlations.append([dict(label=FEATURE_LABELS[j], correlation=values[j]) for j in order])
    return dict(raw_features=raw, features=matrix, labels=labels, groups=groups, model=model,
                projection=projection, distance=distance, clusters=clusters,
                backend=backend, correlations=correlations, seed=seed,
                validation=field_validation(model["fields"], seed))


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
        model = a["model"]
        coefficients = list(coefficients or [])[:len(model["components"])]
        field = model["mean"].copy()
        for index, value in enumerate(coefficients):
            field += float(value) * model["std"][index] * model["components"][index]
        recipe = dict(mode=mode, group_a=group_a, coefficients_sigma=coefficients,
                      input_indices=a["indices"],
                      extrapolation=any(abs(value) > 2 for value in coefficients))
    else:
        raise ValueError("알 수 없는 형상 생성 방식이야.")
    mask = field > 0
    mask[:2] = mask[-2:] = False
    mask[:, :2] = mask[:, -2:] = False
    if mask.sum() < 16:
        raise ValueError("이 조건에서는 형상이 사라져. 변형량이나 보간 비율을 줄여줘.")
    recipe["components_2d"] = int(ndi.label(mask)[1])
    return mask, recipe
