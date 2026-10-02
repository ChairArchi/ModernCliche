from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from importlib.metadata import version, PackageNotFoundError
import json
import zipfile
import numpy as np
import pandas as pd
from PIL import Image

from . import __version__
from .analysis import FEATURE_NAMES
from .pipeline import fingerprint


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=json_safe, allow_nan=False)


def sample_table(prepared, analysis):
    rows = []
    for index, sample in enumerate(prepared["samples"]):
        row = dict(sample.metadata)
        row.update(index=index, id=sample.id, title=sample.metadata.get("title", sample.id),
                   dataset_label=sample.metadata.get("dataset_label", ""),
                   group=int(analysis["labels"][index]),
                   distance=float(analysis["distance"][index]),
                   silhouette=float(analysis["clusters"]["sample_silhouette"][index]),
                   **prepared["quality"][index])
        row.update(dict(zip(FEATURE_NAMES, analysis["raw_features"][index].tolist())))
        if analysis.get("depth_features") is not None:
            from .depth import DEPTH_NAMES
            row.update(dict(zip(DEPTH_NAMES, analysis["depth_features"][index].tolist())))
        rows.append(row)
    return pd.DataFrame(rows)


def png_bytes(image):
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def mesh_bytes(mesh, extension):
    if extension == "glb":
        metric = mesh.copy()
        metric.apply_scale(.001)
        return metric.export(file_type="glb")
    return mesh.export(file_type=extension)


def experiment_zip(prepared, analysis, query="", mesh=None, recipe=None, geometry=None, generated_mask=None, selected_depth=None):
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "scikit-image", "rembg", "onnxruntime", "trimesh", "torch", "transformers", "moge", "utils3d"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    samples = [dict(id=sample.id, metadata=sample.metadata, quality=quality)
               for sample, quality in zip(prepared["samples"], prepared["quality"])]
    groups = [{key: value for key, value in group.items() if key != "model"} for group in analysis["groups"]]
    recipe = dict(recipe or {})
    recipe["input_ids"] = [prepared["samples"][i].id for i in recipe.get("input_indices", [])]
    depths = prepared.get("depth_results")
    manifest = dict(schema="modern-cliche-experiment-v3", app_version=__version__,
                    created_utc=datetime.now(timezone.utc).isoformat(), query=query, seed=analysis["seed"],
                    fingerprint=fingerprint(prepared, list(range(len(samples)))),
                    preparation=prepared["settings"], feature_backend=analysis["backend"],
                    samples=samples, excluded_samples=prepared.get("excluded_samples", []),
                    duplicates=prepared["duplicates"], failures=prepared["failures"],
                    cluster_reason=analysis["clusters"]["reason"],
                    cluster_candidates=analysis["clusters"]["candidates"],
                    field_validation=analysis["validation"],
                    requested_k=analysis.get("requested_k", 0),
                    generation=recipe, geometry=geometry, packages=packages,
                    depth_estimation=depths[0]["metadata"] if depths else None,
                    selected_depth=selected_depth["metadata"] if selected_depth is not None else None,
                    limitations=["Exploratory sample morphology, not universal category rules.",
                                 "2D masks preserve viewpoint variation; no semantic part correspondences.",
                                 "Monocular depth and camera estimates are unverified; hidden surfaces are unobserved.",
                                 "Legacy silhouette variation uses a radius prior, not inferred depth.",
                                 "No measurement of human uncanny response or GAN mode collapse."])
    report = f"""# Modern Cliché — Experiment

Query: {query}

Images: {len(samples)}; groups: {len(groups)}; seed: {analysis['seed']}.

Feature backend: {analysis['backend']}.

## Method

Reviewed foregrounds → silhouette descriptors and signed-distance fields → optional embedded monocular depth and 3D descriptors → clustering with silhouette selection and subsample ARI → observed visible-surface meshing, or explicitly selected legacy silhouette variation.

## Interpretation

Group differences describe this collected sample. They do not identify semantic anatomy or population laws. Depth/camera predictions are unverified. Visible surfaces do not include the unseen back. A translated shell is a fabrication assumption. Legacy PCA varies 2D silhouette fields and its ellipsoid lift still assumes thickness from local radius. Field holdout scores evaluate silhouettes, not 3D reconstruction accuracy.

## Reproduction

Input images, masks, features, PCA, provenance, generation settings and any inferred depth/point arrays are included. Run `python -m modern_cliche.cli --experiment experiment.zip --output reproduced`. Match package versions in run.json. CLI reproduction uses saved arrays and does not download models. App reanalysis reuses saved depth when model and resolution settings match.

## Depth model

{dumps(manifest['depth_estimation'] or manifest['selected_depth'] or {})}

## Geometry

{dumps(geometry or {})}

## References

See docs/METHODOLOGY.md and docs/REFERENCES.md in the repository. Pretrained models are used for inference. The statistical silhouette field and shell operations are project adaptations.
"""
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("run.json", dumps(manifest))
        archive.writestr("rules.json", dumps(dict(groups=groups, correlations=analysis["correlations"],
                                                  explained_variance=analysis["model"]["explained"])))
        archive.writestr("samples.csv", sample_table(prepared, analysis).to_csv(index=False))
        archive.writestr("report.md", report)
        arrays = BytesIO()
        model = analysis["model"]
        np.savez_compressed(arrays, features=analysis["features"], raw_features=analysis["raw_features"],
                            labels=analysis["labels"], field_mean=model["mean"],
                            field_components=model["components"], field_scores=model["scores"],
                            field_std=model["std"], field_explained=model["explained"])
        archive.writestr("analysis.npz", arrays.getvalue())
        for sample, raw, mask in zip(prepared["samples"], prepared["raw_masks"], prepared["masks"]):
            archive.writestr(f"images/{sample.id}.png", png_bytes(sample.image))
            archive.writestr(f"masks/raw/{sample.id}.png", png_bytes(Image.fromarray(raw.astype(np.uint8) * 255)))
            archive.writestr(f"masks/aligned/{sample.id}.png", png_bytes(Image.fromarray(mask.astype(np.uint8) * 255)))
        if depths:
            from .depth import encode_depth, depth_preview
            for sample, depth in zip(prepared["samples"], depths):
                archive.writestr(f"depth/{sample.id}.npz", encode_depth(depth))
                archive.writestr(f"depth/{sample.id}.png", png_bytes(depth_preview(depth)))
        if selected_depth is not None:
            from .depth import encode_depth, depth_preview
            archive.writestr("selected_depth.npz", encode_depth(selected_depth))
            archive.writestr("selected_depth.png", png_bytes(depth_preview(selected_depth)))
        if mesh is not None:
            for extension in ("stl", "obj", "glb", "ply"):
                archive.writestr(f"model.{extension}", mesh_bytes(mesh, extension))
        if generated_mask is not None:
            archive.writestr("generated_mask.png", png_bytes(Image.fromarray(generated_mask.astype(np.uint8) * 255)))
    return output.getvalue()
