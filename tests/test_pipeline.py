from io import BytesIO
import json
import subprocess
import sys
import zipfile
import numpy as np
from PIL import Image
import pytest
import trimesh
from sklearn.metrics import adjusted_rand_score

from modern_cliche.data import deduplicate, normalise_query, Sample
from modern_cliche.segmentation import normalise_mask, extract_mask
from modern_cliche.demo import demo_samples
from modern_cliche.pipeline import prepare, load_experiment, fingerprint
from modern_cliche.analysis import analyse, choose_clusters, synthesize_field
from modern_cliche.geometry import lift_to_mesh
from modern_cliche.morphology import aggregate_depth_results
from modern_cliche.export import experiment_zip, mesh_bytes


@pytest.fixture(scope="module")
def run():
    prepared = prepare(demo_samples(), size=64)
    analysis = analyse(prepared["masks"], requested_k=2)
    return prepared, analysis


def test_duplicate_detection_preserves_shape_variation():
    samples = demo_samples()
    kept, removed = deduplicate(samples + [samples[0]])
    assert len(kept) >= 8
    assert removed[-1]["duplicate_of"] == samples[0].id
    assert len({sample.metadata["dataset_label"] for sample in kept}) == 2


def test_mixed_language_query_is_preserved_and_translated():
    assert normalise_query("가고일 gothic church") == "gargoyle gothic church"
    assert normalise_query("cat 조각") == "cat 조각"


def test_scale_translation_normalisation():
    small = np.zeros((100, 100), bool)
    small[25:65, 35:55] = True
    large = np.zeros((200, 200), bool)
    large[70:150, 90:130] = True
    assert np.array_equal(normalise_mask(small), normalise_mask(large))


def test_known_groups_and_repeatability(run):
    prepared, analysis = run
    ground_truth = [sample.metadata["dataset_label"] for sample in prepared["samples"]]
    assert len(analysis["groups"]) == 2
    assert adjusted_rand_score(ground_truth, analysis["labels"]) > .9
    repeat = analyse(prepared["masks"], requested_k=2)
    assert np.array_equal(repeat["labels"], analysis["labels"])
    assert analysis["validation"]["available"]


def test_identical_inputs_do_not_invent_clusters():
    result = choose_clusters(np.zeros((12, 10)))
    assert len(set(result["labels"])) == 1
    assert result["silhouette"] is None


def test_density_clustering_can_reject_outliers():
    rng = np.random.default_rng(17)
    a = rng.normal((-3, 0), .18, size=(35, 2))
    b = rng.normal((3, 0), .18, size=(35, 2))
    outliers = np.array([[0, 4], [0, -4], [6, 5], [-6, -5]], dtype=float)
    result = choose_clusters(np.vstack([a, b, outliers]), seed=17)
    assert len(set(result["labels"][result["labels"] >= 0])) >= 2
    assert len(result["outliers"]) >= 1


def test_rule_deformation_changes_geometry(run):
    prepared, analysis = run
    group = next(i for i, g in enumerate(analysis["groups"]) if len(g["model"]["components"]))
    first, _ = synthesize_field(analysis, "variation", group, coefficients=[-1.5])
    second, _ = synthesize_field(analysis, "variation", group, coefficients=[1.5])
    assert np.count_nonzero(first != second) > 20
    mesh_a, _ = lift_to_mesh(first)
    mesh_b, _ = lift_to_mesh(second)
    assert abs(mesh_a.volume - mesh_b.volume) > 1.


@pytest.mark.parametrize("depth", [.3, 1., 2.])
def test_volume_preserves_outline_and_has_no_canvas_plate(run, depth):
    _, analysis = run
    mask, _ = synthesize_field(analysis, "observed")
    mesh, metrics = lift_to_mesh(mask, depth, 120.)
    assert mesh.is_watertight and mesh.volume > 0
    assert metrics["projection_iou"] > .95
    assert metrics["extents_mm"][1] > 5
    assert np.isclose(mesh.extents.max(), 120.)
    assert metrics["radius_cover_points"] > 0


def test_glb_uses_metres(run):
    _, analysis = run
    mask, _ = synthesize_field(analysis, "observed")
    mesh, _ = lift_to_mesh(mask)
    scene = trimesh.load(BytesIO(mesh_bytes(mesh, "glb")), file_type="glb")
    assert np.isclose(scene.extents.max(), .120, atol=1e-5)


def test_statistical_point_maps_create_shared_visible_surface():
    results = []
    y, x = np.mgrid[:72, :72]
    for offset in (-.05, 0., .05):
        valid = ((x - 36) / 25) ** 2 + ((y - 36) / 29) ** 2 <= 1
        z = 2 + .14 * np.cos((x - 36) / 14) + offset
        points = np.stack([(x - 36) / 36 * z, (y - 36) / 36 * z, z], axis=-1).astype(np.float32)
        points[~valid] = 0
        results.append(dict(points=points, valid=valid, depth=np.where(valid, z, 0).astype(np.float32)))
    aggregate = aggregate_depth_results(results, size=96, coverage_threshold=.5)
    assert aggregate["valid"].sum() > 500
    assert aggregate["metadata"]["specimens"] == 3
    assert np.all(aggregate["points"][aggregate["valid"], 2] > 0)


def test_fingerprint_changes_with_reviewed_mask(run):
    prepared, _ = run
    before = fingerprint(prepared, [0])
    copy = {**prepared, "masks": [mask.copy() for mask in prepared["masks"]]}
    copy["masks"][0][30, 30] = not copy["masks"][0][30, 30]
    assert fingerprint(copy, [0]) != before


def test_export_reproduction_uses_saved_masks_and_features(run, tmp_path):
    prepared, analysis = run
    mask, recipe = synthesize_field(analysis, "interpolate", 0, 1, mix=.35)
    mesh, metrics = lift_to_mesh(mask, .8, 140.)
    raw = experiment_zip(prepared, analysis, "verification", mesh, recipe, metrics, mask)
    archive_path = tmp_path / "experiment.zip"
    archive_path.write_bytes(raw)
    samples, overrides, manifest = load_experiment(raw)
    restored = prepare(samples, overrides=overrides, size=64)
    assert fingerprint(prepared, list(range(len(samples)))) == fingerprint(restored, list(range(len(samples))))
    assert np.array_equal(manifest["_saved_features"], analysis["features"])
    output = tmp_path / "reproduced"
    result = subprocess.run([
        sys.executable, "-m", "modern_cliche.cli", "--experiment", str(archive_path), "--output", str(output)
    ], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    copied = trimesh.load(output / "model.stl")
    assert np.allclose(copied.extents, mesh.extents, atol=1e-4)
    assert np.isclose(copied.volume, mesh.volume, rtol=1e-5)
    with zipfile.ZipFile(output / "experiment.zip") as archive:
        generated = np.asarray(Image.open(BytesIO(archive.read("generated_mask.png")))) > 0
        record = json.loads(archive.read("run.json"))
    assert np.array_equal(generated, mask)
    assert record["generation"]["mix"] == .35


def test_bad_mask_is_recorded_not_silently_replaced():
    from PIL import Image as PILImage
    image = PILImage.new("RGB", (80, 80), "white")
    sample = Sample("invalid", image, {"title": "empty"})
    prepared = prepare([sample], method="binary")
    assert not prepared["samples"] and prepared["failures"]


def test_alpha_is_kept_without_network():
    sample = demo_samples()[0]
    mask, info = extract_mask(sample.image)
    assert info["method"] == "uploaded-alpha"
    assert mask.sum() > 0
