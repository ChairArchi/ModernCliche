from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
from skimage.morphology import medial_axis
import trimesh


def lift_to_mesh(mask: np.ndarray, depth_scale: float = 1., size_mm: float = 120.) -> tuple[trimesh.Trimesh, dict]:
    """Lift a 2D medial-axis radius model into 3D ellipsoids.

    The measured x/y structure is preserved. Depth is an explicit isotropic prior,
    not recovered geometry, an image-intensity height map, or a pretrained 3D decoder.
    """
    if not .3 <= depth_scale <= 2.:
        raise ValueError("깊이 가정은 0.3~2.0 사이로 설정해줘.")
    if not 10 <= size_mm <= 1000:
        raise ValueError("모델 크기는 10~1000mm 사이로 설정해줘.")
    skeleton, distance = medial_axis(mask, return_distance=True, rng=17)
    medial_points = np.argwhere(skeleton)
    if not len(medial_points):
        raise ValueError("중심선을 추출할 수 없어.")
    # Discrete thinning may omit extremities. Retain a maximal inscribed-disc cover
    # of all measured foreground pixels, removing analytically contained discs.
    candidates = np.argwhere(mask)
    radii = np.maximum(.6, distance[mask] - .5)
    order = np.argsort(-radii, kind="stable")
    centres, retained_radii = [], []
    for index in order:
        point, radius = candidates[index], float(radii[index])
        if centres:
            centre_distance = np.linalg.norm(np.asarray(centres) - point, axis=1)
            if np.any(centre_distance + radius <= np.asarray(retained_radii) + 1e-6):
                continue
        centres.append(point)
        retained_radii.append(radius)
    h, w = mask.shape
    max_radius = max(retained_radii)
    zhalf = int(np.ceil(max_radius * depth_scale)) + 3
    volume = np.full((h + 4, w + 4, zhalf * 2 + 1), -1., dtype=np.float32)
    for (y, x), radius in zip(centres, retained_radii):
        cy, cx = int(y + 2), int(x + 2)
        reach, zreach = int(np.ceil(radius)) + 1, int(np.ceil(radius * depth_scale)) + 1
        y0, y1 = max(0, cy - reach), min(h + 4, cy + reach + 1)
        x0, x1 = max(0, cx - reach), min(w + 4, cx + reach + 1)
        z0, z1 = max(0, zhalf - zreach), min(volume.shape[2], zhalf + zreach + 1)
        yy, xx, zz = np.ogrid[y0:y1, x0:x1, z0:z1]
        ball = radius - np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2 + ((zz - zhalf) / depth_scale) ** 2)
        block = volume[y0:y1, x0:x1, z0:z1]
        np.maximum(block, ball, out=block)
    volume[0] = volume[-1] = -1
    volume[:, 0] = volume[:, -1] = -1
    volume[:, :, 0] = volume[:, :, -1] = -1
    vertices, faces, _, _ = marching_cubes(volume, level=0, allow_degenerate=False)
    # Convert array y/x/z into right-handed x/depth/up coordinates.
    vertices = vertices[:, [1, 2, 0]] * np.array([1., 1., -1.])
    vertices -= (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    vertices *= size_mm / max(np.ptp(vertices, axis=0).max(), 1e-8)
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.fix_normals(multibody=True)
    if not mesh.is_watertight:
        raise ValueError("닫힌 표면을 만들지 못했어. 마스크의 얇은 부분이나 분리된 영역을 확인해줘.")
    projected = volume[2:h + 2, 2:w + 2, :].max(axis=2) >= 0
    iou = np.count_nonzero(projected & mask) / max(1, np.count_nonzero(projected | mask))
    return mesh, dict(method="maximal-inscribed-disc-ellipsoid-union", depth_prior="isotropic_local_radius",
                      depth_scale=float(depth_scale), size_mm=float(size_mm),
                      medial_points=len(medial_points), radius_cover_points=len(centres), projection_iou=float(iou),
                      components_3d=len(mesh.split(only_watertight=False)),
                      watertight=bool(mesh.is_watertight), vertices=len(mesh.vertices), faces=len(mesh.faces),
                      extents_mm=mesh.extents.tolist(), volume_mm3=float(mesh.volume))
