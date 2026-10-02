from io import BytesIO
import json
import subprocess
import sys
import zipfile
import numpy as np
import pytest
import trimesh

from modern_cliche.depth import backproject, depth_features, encode_depth, decode_depth
from modern_cliche.surface import surface_mesh
from modern_cliche.demo import demo_samples
from modern_cliche.pipeline import prepare, load_experiment
from modern_cliche.analysis import analyse
from modern_cliche.export import experiment_zip, mesh_bytes


def fixture_depth(amplitude=.1, step=False):
    y, x = np.mgrid[:48, :64]
    depth = 2. + amplitude * np.sin(x / 5) * np.cos(y / 7)
    if step:
        depth[:, 32:] += 1.
    valid = np.zeros(depth.shape, bool)
    valid[4:-4, 4:-4] = True
    intrinsics = np.array([[.8, 0, .5], [0, 1.1, .5], [0, 0, 1]], np.float32)
    result = dict(depth=depth.astype(np.float32), points=backproject(depth, intrinsics), intrinsics=intrinsics,
                  valid=valid, foreground=valid.copy(), rgb=np.full((*depth.shape, 3), 140, np.uint8),
                  metadata=dict(model_id="test", revision="1", scale="synthetic", model="moge-small"))
    result["features"] = depth_features(result)
    return result


def test_perspective_backprojection_and_real_internal_detail():
    result = fixture_depth()
    points = result["points"]
    h, w = result["depth"].shape
    k = result["intrinsics"]
    assert np.isclose(points[10, 20, 0] / points[10, 20, 2] * k[0, 0] + k[0, 2], 20.5 / w)
    mesh, metrics = surface_mesh(result)
    assert not mesh.is_watertight
    assert np.ptp(mesh.vertices[:, 1]) > 1.
    # Same silhouette, differing interior depth -> differing geometry.
    flat, _ = surface_mesh(fixture_depth(0))
    assert np.ptp(flat.vertices[:, 1]) == 0
    assert metrics["hidden_surface"] == "unobserved-open-surface"
    assert np.isclose(mesh.extents.max(), 120.)


def test_discontinuities_do_not_bridge_occlusion_boundaries():
    mesh, metrics = surface_mesh(fixture_depth(0, step=True), edge_threshold=.03)
    assert metrics["rejected_discontinuity_faces"] > 0
    assert metrics["components_3d"] == 2
    # All retained faces lie on one side of the depth jump.
    assert np.max(np.ptp(mesh.vertices[mesh.faces, 1], axis=1)) < 1e-6


def test_shell_is_closed_without_inflating_the_front():
    surface, _ = surface_mesh(fixture_depth())
    shell, metrics = surface_mesh(fixture_depth(), thickness_mm=2.)
    assert shell.is_watertight and shell.volume > 0 and shell.is_winding_consistent
    assert np.isclose(shell.extents.max(), 120.)
    assert metrics["hidden_surface"] == "translated-visible-surface-shell"
    assert np.allclose(shell.vertices[len(surface.vertices):] - shell.vertices[:len(surface.vertices)], [0, 2, 0])
    scene = trimesh.load(BytesIO(mesh_bytes(shell, "glb")), file_type="glb")
    assert np.allclose(scene.extents, shell.extents / 1000, atol=1e-6)
    assert len(shell.visual.vertex_colors) == len(shell.vertices)


def test_depth_changes_sorting_even_with_identical_silhouettes():
    mask = np.zeros((64, 64), bool)
    mask[8:-8, 8:-8] = True
    results = [fixture_depth(a) for a in [.001, .002, .003, .2, .21, .22, .23, .24]]
    result = analyse([mask] * 8, depth_results=results)
    assert result["backend"] == "shape+depth"
    assert np.std(result["features"], axis=0).max() > 0
    assert any(g.get("depth_rules") for g in result["groups"])
    with pytest.raises(ValueError, match="same depth model"):
        mixed = fixture_depth(); mixed["metadata"]["model_id"] = "other-model"
        analyse([mask] * 3, depth_results=[results[0], results[1], mixed])


@pytest.mark.parametrize("cohort", [True, False])
def test_depth_zip_replays_without_model_inference(tmp_path, cohort):
    prepared = prepare(demo_samples(), size=64)
    depth = fixture_depth()
    if cohort:
        prepared["depth_results"] = [depth] * len(prepared["samples"])
    analysis = analyse(prepared["masks"], depth_results=prepared.get("depth_results"))
    mesh, metrics = surface_mesh(depth, thickness_mm=2., max_resolution=128)
    recipe = dict(mode="observed", specimen=0, input_indices=[0], geometry="depth")
    archive = experiment_zip(prepared, analysis, "fixture", mesh, recipe, metrics, prepared["masks"][0], selected_depth=depth)
    _, _, manifest = load_experiment(archive)
    assert np.array_equal(manifest["_selected_depth"]["points"], depth["points"])
    source = tmp_path / "experiment.zip"; source.write_bytes(archive)
    out = tmp_path / "out"
    completed = subprocess.run([sys.executable, "-m", "modern_cliche.cli", "--experiment", str(source), "--output", str(out)],
                               text=True, capture_output=True, timeout=90)
    assert completed.returncode == 0, completed.stderr
    loaded = trimesh.load(out / "model.ply")
    assert np.allclose(loaded.vertices, mesh.vertices, atol=1e-5)
    record = json.loads((out / "geometry.json").read_text())
    assert record["thickness_mm"] == 2.
    assert record["faces"] == len(mesh.faces)


def test_invalid_maps_and_empty_foreground_are_rejected():
    result = fixture_depth(); result["valid"][:] = False
    with pytest.raises(ValueError):
        surface_mesh(result)
    bad = fixture_depth(); bad["points"] = bad["points"][:-1]
    with pytest.raises(ValueError):
        decode_depth(encode_depth(bad))


def test_point_contacts_split_before_closing_shell():
    result = fixture_depth(0)
    valid = np.zeros_like(result['valid'])
    valid[5:15, 5:15] = True
    valid[14:24, 14:24] = True  # Two patches meet at exactly one grid vertex.
    result['valid'] = valid
    result['foreground'] = valid.copy()
    mesh, metadata = surface_mesh(result, thickness_mm=2.)
    assert metadata['split_contact_vertices'] == 1
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    assert metadata['components_3d'] == 2


def test_inference_cache_is_bound_to_reviewed_mask_and_handles_corruption(monkeypatch, tmp_path):
    import torch
    from PIL import Image
    import modern_cliche.depth as module
    calls = []
    result = fixture_depth()
    class FakeModel:
        def infer(self, tensor, **kwargs):
            calls.append(1)
            return {key: torch.from_numpy(result[key]) for key in ['depth','points','intrinsics','valid']} | {'mask': torch.from_numpy(np.ones((48,64),bool))}
    monkeypatch.setattr(module,'load_depth_model',lambda *args:(None,FakeModel()))
    monkeypatch.setenv('MODERN_CLICHE_DEPTH_CACHE',str(tmp_path))
    image=Image.fromarray(result['rgb']);mask=result['valid'].copy()
    a=module.infer_depth(image,mask,'moge-small',max_size=384)
    b=module.infer_depth(image,mask,'moge-small',max_size=384)
    assert len(calls)==1 and np.array_equal(a['points'],b['points'])
    mask[4,4]=False
    c=module.infer_depth(image,mask,'moge-small',max_size=384)
    assert len(calls)==2 and a['metadata']['cache_key']!=c['metadata']['cache_key']
    (tmp_path / (c['metadata']['cache_key']+'.npz')).write_bytes(b'corrupt cache')
    module.infer_depth(image,mask,'moge-small',max_size=384)
    assert len(calls)==3


def test_original_mask_position_invalidates_depth_analysis_signature():
    from modern_cliche.pipeline import fingerprint
    prepared=prepare(demo_samples(),size=64)
    before=fingerprint(prepared,[0])
    changed={**prepared,'raw_masks':[mask.copy() for mask in prepared['raw_masks']]}
    changed['raw_masks'][0]=np.roll(changed['raw_masks'][0],3,axis=1)
    # Aligned masks alone cannot distinguish selection of another image region.
    assert np.array_equal(changed['masks'][0],prepared['masks'][0])
    assert fingerprint(changed,[0])!=before
