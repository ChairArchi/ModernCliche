from __future__ import annotations

import argparse
from pathlib import Path
from .data import decode_image, search_images, download_images
from .pipeline import prepare, load_experiment, subset
from .analysis import analyse, synthesize_field
from .geometry import lift_to_mesh
from .depth import infer_cohort, default_model
from .surface import surface_mesh
from .export import experiment_zip, dumps, mesh_bytes


def main():
    parser = argparse.ArgumentParser(description="Modern Cliché image→statistical-shape→3D experiment")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--query", help="Search Wikimedia Commons")
    inputs.add_argument("--input", type=Path, help="Folder of PNG/JPG/WebP images")
    inputs.add_argument("--experiment", type=Path, help="Reproduce a saved experiment ZIP without network")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--provider", choices=["commons", "openverse"], default="commons")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--segmentation", choices=["u2net", "grabcut", "binary"], default="u2net")
    parser.add_argument("--mode", choices=["observed", "variation", "interpolate"], default=None)
    parser.add_argument("--sigma", type=float, nargs="*", default=None)
    parser.add_argument("--group-a", type=int, default=0)
    parser.add_argument("--group-b", type=int, default=1)
    parser.add_argument("--mix", type=float, default=.5)
    parser.add_argument("--geometry", choices=["depth", "silhouette"], default=None)
    parser.add_argument("--depth-model", choices=["moge-small", "moge-base", "moge-large", "depth-anything"], default=None)
    parser.add_argument("--depth-resolution", type=int, default=768)
    parser.add_argument("--detail", type=int, default=9, help="MoGe inference resolution level 0–9")
    parser.add_argument("--edge-threshold", type=float, default=.03)
    parser.add_argument("--mesh-resolution", type=int, default=768)
    parser.add_argument("--thickness-mm", type=float, default=0.)
    parser.add_argument("--specimen", type=int, default=None)
    parser.add_argument("--depth", type=float, default=1.)
    parser.add_argument("--size-mm", type=float, default=120.)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--include-review", action="store_true", help="Include flagged automatic masks for an explicitly unreviewed diagnostic run")
    args = parser.parse_args()
    manifest, overrides = {}, {}
    failures = []
    if args.experiment:
        samples, overrides, manifest = load_experiment(args.experiment.read_bytes())
        query = manifest.get("query", "")
    elif args.query:
        items, query = search_images(args.query, args.limit, args.provider)
        samples, failures = download_images(items)
    else:
        query = args.input.name
        files = sorted(path for path in args.input.iterdir() if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
        samples = [decode_image(path.read_bytes(), {"title": path.name, "provider": "upload"}) for path in files[:80]]
    settings = manifest.get("preparation", {})
    prepared = prepare(samples, method=settings.get("method", args.segmentation),
                       threshold=settings.get("threshold", .5), size=settings.get("size", 96), overrides=overrides)
    prepared["failures"].extend(failures)
    if not args.experiment and not args.include_review:
        keep = [i for i, quality in enumerate(prepared["quality"]) if not quality.get("needs_review", False)]
        excluded = [dict(id=sample.id, metadata=sample.metadata, reason="automatic-mask-needs-review")
                    for i, sample in enumerate(prepared["samples"]) if i not in keep]
        prepared = subset(prepared, keep)
        prepared["excluded_samples"] = excluded
    # Restore audited sample order; saved masks override any automatic foreground model.
    seed = manifest.get("seed", args.seed)
    requested_k = manifest.get("requested_k", 0)
    backend = manifest.get("feature_backend", "shape")
    embeddings = manifest.get("_saved_features")
    geometry_previous = manifest.get("geometry") or {}
    reconstruction = args.geometry or ("depth" if geometry_previous.get("method") == "predicted-camera-point-map-triangulation"
                                      else "silhouette" if args.experiment else "depth")
    saved = manifest.get("_saved_depth")
    if saved and args.depth_model is None:
        prepared["depth_results"] = [saved[s.id] for s in prepared["samples"]]
    elif reconstruction == "depth" and (not manifest.get("selected_depth") or args.depth_model is not None):
        prepared["depth_results"] = infer_cohort(prepared, args.depth_model or default_model(), args.depth_resolution, args.detail)
        embeddings = None
        backend = "shape"
    analysis = analyse(prepared["masks"], seed, requested_k, embeddings, backend, prepared.get("depth_results"))
    analysis["requested_k"] = requested_k
    previous = manifest.get("generation", {})
    mode = args.mode or previous.get("mode", "observed")
    a = previous.get("group_a", args.group_a)
    b = previous.get("group_b", min(args.group_b, len(analysis["groups"]) - 1))
    coefficients = args.sigma if args.sigma is not None else previous.get("coefficients_sigma", [0., 0., 0.])
    mask, recipe = synthesize_field(analysis, mode, a, b, previous.get("mix", args.mix), coefficients,
                                    previous.get("specimen"))
    if reconstruction == "depth":
        if mode != "observed":
            raise ValueError("깊이 표면은 관측 이미지에서 생성합니다. 실루엣 변형에는 --geometry silhouette을 명시하십시오.")
        specimen = args.specimen if args.specimen is not None else previous.get("specimen", analysis["groups"][a]["medoid"])
        if not 0 <= specimen < len(prepared["samples"]):
            raise ValueError("관측 이미지 번호가 범위를 벗어났습니다.")
        if prepared.get("depth_results"):
            result = prepared["depth_results"][specimen]
        elif manifest.get("_selected_depth") is not None and specimen == previous.get("specimen"):
            result = manifest["_selected_depth"]
        else:
            from .depth import infer_depth
            result = infer_depth(prepared["samples"][specimen].image, prepared["raw_masks"][specimen],
                args.depth_model or default_model(), args.depth_resolution, args.detail)
        mesh, geometry = surface_mesh(result, geometry_previous.get("size_mm", args.size_mm),
            geometry_previous.get("edge_threshold", args.edge_threshold), geometry_previous.get("max_resolution", args.mesh_resolution),
            geometry_previous.get("thickness_mm", args.thickness_mm))
        mask = prepared["masks"][specimen]
        recipe = dict(mode="observed", specimen=specimen, input_indices=[specimen], geometry="depth")
    else:
        mesh, geometry = lift_to_mesh(mask, geometry_previous.get("depth_scale", args.depth),
                                     geometry_previous.get("size_mm", args.size_mm))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "experiment.zip").write_bytes(experiment_zip(prepared, analysis, query, mesh, recipe, geometry, mask,
        selected_depth=result if reconstruction == "depth" else None))
    for extension in ("stl", "obj", "glb", "ply"):
        data = mesh_bytes(mesh, extension)
        (args.output / f"model.{extension}").write_bytes(data.encode() if isinstance(data, str) else data)
    (args.output / "geometry.json").write_text(dumps(geometry), encoding="utf-8")
    print(dumps(dict(images=len(prepared["samples"]), groups=len(analysis["groups"]),
                    failures=prepared["failures"], geometry=geometry, output=str(args.output))))


if __name__ == "__main__":
    main()
