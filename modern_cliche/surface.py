"""Image-grid triangulation of predicted camera-space points; no volume inflation."""
from __future__ import annotations
import numpy as np
import trimesh


def split_vertex_fans(vertices, faces, colors):
    """Duplicate a grid vertex shared by otherwise disjoint surface fans.

    At a cut/diagonal mask contact, a single vertex can have four boundary edges.
    Extruding it would create a nonmanifold edge with four incident side faces.
    Splitting these fans changes topology only, leaving every coordinate intact.
    """
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    boundary = edges[counts[inverse] == 1]
    degree = np.bincount(boundary.ravel(), minlength=len(vertices))
    candidates = np.flatnonzero(degree > 2)
    extra_vertices, extra_colors = [], []
    for vertex in candidates:
        incident = np.flatnonzero(np.any(faces == vertex, axis=1))
        remaining = set(incident.tolist())
        fans = []
        while remaining:
            stack, fan = [remaining.pop()], []
            while stack:
                index = stack.pop(); fan.append(index)
                other = set(faces[index]) - {vertex}
                adjacent = [j for j in remaining if other & (set(faces[j]) - {vertex})]
                for j in adjacent:
                    remaining.remove(j); stack.append(j)
            fans.append(fan)
        for fan in fans[1:]:
            replacement = len(vertices) + len(extra_vertices)
            extra_vertices.append(vertices[vertex]); extra_colors.append(colors[vertex])
            for index in fan:
                faces[index, faces[index] == vertex] = replacement
    if extra_vertices:
        vertices = np.concatenate([vertices, extra_vertices])
        colors = np.concatenate([colors, extra_colors])
    return vertices, faces, colors, len(extra_vertices)


def surface_mesh(result, size_mm=120., edge_threshold=.03, max_resolution=768, thickness_mm=0.):
    if not np.isfinite(size_mm) or not 1 <= size_mm <= 10000 or not 0 < edge_threshold <= 1:
        raise ValueError("모델 크기 또는 깊이 불연속 기준이 올바르지 않습니다.")
    if not 0 <= thickness_mm < size_mm / 4:
        raise ValueError("추가 두께는 최대 크기의 1/4 미만이어야 합니다.")
    if not 32 <= max_resolution <= 768:
        raise ValueError("메시 해상도는 32~768 범위여야 합니다.")
    points, valid, rgb = result["points"], result["valid"], result["rgb"]
    step = max(1, int(np.ceil(max(valid.shape) / max_resolution)))
    points, valid, rgb = points[::step, ::step], valid[::step, ::step], rgb[::step, ::step]
    h, w = valid.shape
    grid = np.arange(h * w).reshape(h, w)
    tl, tr, bl, br = grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]
    faces = np.concatenate([np.stack([tl, bl, tr], -1).reshape(-1, 3),
                            np.stack([tr, bl, br], -1).reshape(-1, 3)])
    flat = points.reshape(-1, 3)
    good = valid.reshape(-1) & np.isfinite(flat).all(axis=1) & (flat[:, 2] > 0)
    supported = good[faces].all(axis=1)
    faces = faces[supported]
    z = flat[faces, 2]
    continuous = (z.max(axis=1) - z.min(axis=1)) / np.maximum(1e-8, z.min(axis=1)) <= edge_threshold
    rejected = int((~continuous).sum())
    faces = faces[continuous]
    area = np.linalg.norm(np.cross(flat[faces[:, 1]] - flat[faces[:, 0]], flat[faces[:, 2]] - flat[faces[:, 0]]), axis=1)
    faces = faces[area > 1e-12]
    if len(faces) < 16:
        raise ValueError("메시를 구성할 연속 표면이 부족합니다. 마스크 또는 불연속 기준을 검토하십시오.")
    selected = np.unique(faces)
    remap = np.full(h * w, -1, dtype=int)
    remap[selected] = np.arange(len(selected))
    faces = remap[faces]
    # OpenCV right/down/forward -> right/forward/up, retaining handedness.
    vertices = flat[selected][:, [0, 2, 1]].astype(float)
    vertices[:, 2] *= -1
    vertices -= (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    span = np.ptp(vertices, axis=0)
    limits = np.array([size_mm, size_mm - thickness_mm, size_mm])
    vertices *= np.min(limits / np.maximum(1e-8, span))
    colors = rgb.reshape(-1, 3)[selected]
    vertices, faces, colors, splits = split_vertex_fans(vertices, faces, colors)
    shell_boundary = 0
    if thickness_mm:
        n = len(vertices)
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
        boundary = edges[counts[inverse] == 1]
        shell_boundary = len(boundary)
        u, v = boundary.T
        sides = np.concatenate([np.stack([u, u + n, v + n], -1), np.stack([u, v + n, v], -1)])
        vertices = np.concatenate([vertices, vertices + np.array([0, thickness_mm, 0])])
        faces = np.concatenate([faces, faces[:, ::-1] + n, sides])
        colors = np.concatenate([colors, colors])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
    if thickness_mm and not mesh.is_watertight:
        raise ValueError("추가 두께의 경계가 닫히지 않았습니다. 표면 모드로 저장하거나 마스크를 수정하십시오.")
    metadata = dict(method="predicted-camera-point-map-triangulation", depth_model=result["metadata"],
                    size_mm=float(size_mm), edge_threshold=float(edge_threshold), max_resolution=int(max_resolution),
                    sampling_step=step, thickness_mm=float(thickness_mm),
                    hidden_surface="translated-visible-surface-shell" if thickness_mm else "unobserved-open-surface",
                    vertices=len(mesh.vertices), faces=len(mesh.faces), watertight=bool(mesh.is_watertight),
                    components_3d=len(mesh.split(only_watertight=False)), rejected_discontinuity_faces=rejected,
                    surface_pixel_coverage=float(len(selected) / max(1, valid.sum())), shell_boundary_edges=shell_boundary,
                    split_contact_vertices=splits,
                    extents_mm=mesh.extents.tolist(), area_mm2=float(mesh.area),
                    volume_mm3=float(mesh.volume) if thickness_mm else None,
                    color="source-image-vertex-colors", scale="user-normalized-size; not a measurement")
    return mesh, metadata
