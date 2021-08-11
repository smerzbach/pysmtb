"""raytracing helper functions for deferred rendering using pyembree as a backend

Currently, meshes are represented as simple dictionaries with vertices, faces, uvs and optionally additional per-vertex
attributes (e.g., tangents, bitangents and normals). This is due to a limitation in the trimesh.Trimesh class, namely
that texture coordinates can only be specified per vertex and not per face, preventing proper unwrapping. Inputs of type
trimesh.Trimesh or trimesh.Scene are therefore converted to dictionaries or lists of dictionaries with the above keys.

The most important function is embree_render_deferred():

embree_render_deferred() produces inputs for deferred shading. It takes a list of meshes and a camera dict, as well as a
point light position as input and returns np.ndarray buffers with all relevant geometric quantities per intersected
pixel: 3D intersection points, interpolated vertex normals and tangents, normalized local and unnormalized global light
and view directions.

For additional light sources, get_local_dirs() can be used with the buffers returned from embree_render_deferred() and
a light position.

embree_render_deferred() calls the following functions: embree_create_scene(), embree_intersect_scene(),
interpolate_vertex_attributes() and get_local_dirs().
- embree_create_scene() constructs an EmbreeScene object, given the list of meshes. Camera parameters are either user-
specified, or will be set automatically (random camera position facing the scene center). Unless specified, the camera
focal length is automatically set so that the scene tightly fits onto the camera sensor. Unless explicitly disabled, it
computes per-vertex tangent frames, using each mesh's texture coordinates.
- embree_intersect_scene() performs the actual ray tracing and returns a pixel mask with the camera's resolution
indicating which pixels are hit, as well as buffers for all intersected pixels with the following attributes:
geometry and triangle IDs for each intersection, intersection depth (distance from camera), 3D intersection point,
barycentric coordinates within each triangle
- interpolate_vertex_attributes() computes per pixel texture coordinates and tangent frames by interpolating with the
barycentric coordinates returned from the embree_intersect_scene()

missing features:
- tracing shadow rays
- camera distortion model
- fallback tangent computing for meshes without texture coordinates
"""
from copy import deepcopy
import numpy as np
import trimesh.visual.texture
from warnings import warn
from pyembree.rtcore_scene import EmbreeScene
from pyembree.mesh_construction import TriangleMesh

from pysmtb.utils import Dct, find_dim, dims_execpt, assign_masked

try:
    # we only optionally depend on trimesh, a lot of functionality also works without
    from trimesh import Scene, Trimesh
except ModuleNotFoundError:
    from types import NoneType
    Scene = NoneType
    Trimesh = NoneType


def safe_divide(dividend, divisor, eps=1e-17):
    mask = divisor < eps
    divisor[mask] = 1
    return dividend / divisor


def normalize(vec, axis=0):
    """normalize input array along specified dimension"""
    # return vec / np.linalg.norm(vec, axis=axis, keepdims=True)
    return safe_divide(vec, np.linalg.norm(vec, axis=axis, keepdims=True))


def get_vertices(mesh, join_geometries=False):
    """given an input mesh (trimesh.Trimesh, trimesh.Scene, dict), return the vertices as NV x 3 array"""
    if isinstance(mesh, Trimesh):
        vertices = mesh.vertices
    elif isinstance(mesh, Scene):
        if not join_geometries and len(mesh.geometry) > 1:
            raise Exception('mesh has multiple geometries, set join_geometries=True')
        vertices = np.concatenate([g.vertices for g in mesh.geometry.values()], axis=0)
    elif isinstance(mesh, dict):
        if not hasattr(mesh, 'vertices'):
            raise Exception('mesh dict should have field vertices')
        vertices = mesh['vertices']
    else:
        raise Exception('unsupported input type: ' + str(type(mesh)))
    return vertices


def get_normals(mesh, join_geometries=False):
    """given an input mesh (trimesh.Trimesh, trimesh.Scene, dict), return the vertex normals as NV x 3 array"""
    if isinstance(mesh, Trimesh):
        normals = mesh.normals
    elif isinstance(mesh, Scene):
        if not join_geometries and len(mesh.geometry) > 1:
            raise Exception('mesh has multiple geometries, set join_geometries=True')
        normals = np.concatenate([g.normals for g in mesh.geometry.values()], axis=0)
    elif isinstance(mesh, dict):
        if not hasattr(mesh, 'normals'):
            raise Exception('mesh dict should have field normals')
        normals = mesh['normals']
    else:
        raise Exception('unsupported input type: ' + str(type(mesh)))
    return normals


def get_vertex_normals(mesh):
    """return per-vertex normal vectors, which are usually obtained by weighted averaging of each vertex's surrounding
    face normals"""
    if isinstance(mesh, Trimesh):
        vertex_normals = mesh.vertex_normals
    elif isinstance(mesh, Scene):
        vertex_normals = np.concatenate([g.vertex_normals for g in mesh.geometry.values()], axis=0)
    elif isinstance(mesh, dict):
        if not hasattr(mesh, 'vertex_normals'):
            raise Exception('mesh dict should have field vertex_normals')
        vertex_normals = mesh['vertex_normals']
    else:
        raise Exception('unsupported input type: ' + str(type(mesh)))
    return vertex_normals


def get_uv_coords(mesh, join_geometries=False):
    """given an input mesh (trimesh.Trimesh, trimesh.Scene, dict), return the texture coordinates as NV x 2 array"""
    if isinstance(mesh, Trimesh) and hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
        uvs = mesh.visual.uv
    elif isinstance(mesh, Trimesh) and 'ply_raw' in mesh.metadata.keys():
        props = mesh.metadata['ply_raw']['vertex']['properties'].keys()
        u_key = 'u' if 'u' in props else 's'
        v_key = 'v' if 'v' in props else 't'
        u = mesh.metadata['ply_raw']['vertex']['data'][u_key]
        v = mesh.metadata['ply_raw']['vertex']['data'][v_key]
        uvs = np.stack((u, v), axis=1)
    elif isinstance(mesh, Scene):
        if not join_geometries and len(mesh.geometry) > 1:
            raise Exception('mesh has multiple geometries, set join_geometries=True')
        uvs = np.concatenate([g.visual.uv for g in mesh.geometry.values()], axis=0)
    elif isinstance(mesh, dict):
        if not hasattr(mesh, 'uvs'):
            raise Exception('mesh dict should have field uvs')
        uvs = mesh['uvs']
    else:
        raise Exception('unsupported input type: ' + str(type(mesh)))
    return uvs


def get_faces(mesh, join_geometries=False):
    """given an input mesh (trimesh.Trimesh, trimesh.Scene, dict), return the faces as NF x 3 array"""
    if isinstance(mesh, Trimesh):
        faces = mesh.faces
    elif isinstance(mesh, Scene):
        if not join_geometries and len(mesh.geometry) > 1:
            raise Exception('mesh has multiple geometries, set join_geometries=True')
        raise NotImplementedError('join_geometries is not supported for faces')
    elif isinstance(mesh, dict):
        if not hasattr(mesh, 'faces'):
            raise Exception('mesh dict should have field faces')
        faces = mesh['faces']
    else:
        raise Exception('unsupported input type: ' + str(type(mesh)))

    if faces.ndim == 1:
        faces = np.stack([face[1] for face in faces], axis=0)
    if faces.shape[1] == 4:
        faces = np.concatenate((faces[:, :3], faces[:, [0, 2, 3]]), axis=0)
    return faces


def get_tangent_frames(mesh):
    """compute tangent frames per vertex, returns the mesh as dict, even if input was a Trimesh"""
    # algorithm:
    # source: Eric Lengyel, Foundations of Game Engine Development - Volume 2: Rendering
    # originally under http://www.terathon.com/code/tangent.html
    # https://web.archive.org/web/20190211192552/http://www.terathon.com/code/tangent.html
    # given triangle vertices p_0, p_1, p_2 (counterclockwise) with corresponding texture coordinates (u_i, v_i)
    # it holds: p_i − p_j = (u_i − u_j) * t + (v_i − v_j) * b
    #
    # define:
    # e_1 = p_1 - p_0,   (x_1, y_1) = (u_1 - u_0, v_1 - v_0)
    # e_2 = p_2 - p_0,   (x_2, y_2) = (u_2 - u_0, v_2 - v_0)
    #
    # we solve for tangent and bitangent vectors t and b using a system of equations:
    # e_1 = x_1 * t + y_1 * b
    # e_2 = x_2 * t + y_2 * b
    #
    # [t, b] = 1 / (x_1 * y_2 - x_2 * y_1) * [e_1, e_2] * [[y_2, -x_2], [-y_1, x_1]]
    #
    # once solved, compute per-vertex tangents and bitangents by summing up the vectors from each triangle and
    # re-orthonormalizing, given per-vertex normal vectors:
    # t' = normalize(t - dot(t, n) * n)
    # b' = normalize(b - dot(b, n) * n - dot(b, t') * t')

    # convert trimesh inputs to dicts
    if isinstance(mesh, Trimesh):
        mesh = trimesh_to_dict(mesh)

    if isinstance(mesh, Scene):
        meshes = trimesh_scene_to_dicts(mesh)
        for mesh_id in range(len(meshes)):
            meshes[mesh_id] = get_tangent_frames(meshes[mesh_id])
        return meshes

    if hasattr(mesh, 'vertex_tangent_frames'):
        # tangent space already computed
        return mesh

    vertices = get_vertices(mesh)
    faces = get_faces(mesh)
    uvs = get_uv_coords(mesh)
    vertex_tangents = np.zeros_like(vertices)
    vertex_bitangents = np.zeros_like(vertices)
    vertex_normals = mesh.vertex_normals

    face_verts = vertices[faces.T, :]
    face_uvs = uvs[faces.T, :]

    e1 = face_verts[1] - face_verts[0]
    e2 = face_verts[2] - face_verts[0]

    x1 = face_uvs[1, :, 0:1] - face_uvs[0, :, 0:1]
    y1 = face_uvs[1, :, 1:2] - face_uvs[0, :, 1:2]
    x2 = face_uvs[2, :, 0:1] - face_uvs[0, :, 0:1]
    y2 = face_uvs[2, :, 1:2] - face_uvs[0, :, 1:2]

    r = safe_divide(1, (x1 * y2 - x2 * y1))
    t = (e1 * y2 - e2 * y1) * r
    b = (e2 * x1 - e1 * x2) * r

    # sum (bi)tangents that are associated to a vertex from its neighboring triangles
    vertex_tangents[faces[:, 0], :] += t
    vertex_tangents[faces[:, 1], :] += t
    vertex_tangents[faces[:, 2], :] += t
    vertex_bitangents[faces[:, 0], :] += b
    vertex_bitangents[faces[:, 1], :] += b
    vertex_bitangents[faces[:, 2], :] += b

    # normalize summed vectors
    vertex_tangents = normalize(vertex_tangents, axis=1)
    vertex_bitangents = normalize(vertex_bitangents, axis=1)

    # re-orthonormalization
    vertex_tangents = normalize(vertex_tangents - np.sum(vertex_normals * vertex_tangents, axis=1, keepdims=True) * vertex_normals, axis=1)
    vertex_bitangents = normalize(vertex_bitangents - np.sum(vertex_bitangents * vertex_normals, axis=1, keepdims=True) * vertex_normals
                                  - np.sum(vertex_bitangents * vertex_tangents, axis=1, keepdims=True) * vertex_tangents, axis=1)

    # determine handedness of tangent frame
    sgn_mask = np.sum(np.cross(vertex_tangents, vertex_bitangents, axis=1) * vertex_normals, axis=1, keepdims=True) > 0
    det_sgn = np.ones_like(sgn_mask, dtype=np.float32)
    det_sgn[~sgn_mask] = -1.

    # NV x 3 x 3 array (tangent, bitangent & normal stacked in rows)
    matrix_to_tangent_space = np.stack((vertex_tangents, det_sgn * vertex_bitangents, vertex_normals), axis=1)

    mesh.vertex_tangent_frames = matrix_to_tangent_space
    mesh.det_sgn = det_sgn
    return mesh


def create_unit_sphere(res=100):
    """create sphere centered on origin with radius 1 and res many segments in azimuth axis"""
    thetas = np.linspace(0, np.pi, res // 2)
    phis = np.linspace(0, 2 * np.pi, res)
    phis, thetas = np.meshgrid(phis, thetas)
    cp = np.cos(phis)
    sp = np.sin(phis)
    ct = np.cos(thetas)
    st = np.sin(thetas)
    xyz = np.stack((cp * st, sp * st, ct), axis=2).reshape((-1, 3))
    normals = xyz
    uvs = np.stack((phis / (2 * np.pi), thetas / np.pi), axis=2).reshape((-1, 2))

    nv = res * (res // 2)
    faces = np.r_[:nv].reshape(res // 2, res, order='C')

    # # faces:
    # # 0, 1, 2,
    # # 3, 4, 5,
    # # 6, 7, 8
    # faces = np.stack((
    #     faces[:-1],  # 0, 1, 2,
    #     np.roll(faces, -1, axis=0)[:-1],  # 3, 4, 5,
    #     np.roll(faces, -1, axis=1)[:-1],  # 1, 2, 0
    #     np.roll(np.roll(faces, -1, axis=0)[:-1], -1, axis=1),  # 4, 5, 3
    #     np.roll(faces, -1, axis=1)[:-1],  # 1, 2, 0
    #     np.roll(faces, -1, axis=0)[:-1],  # 3, 4, 5
    # ), axis=2).reshape((-1, 3))

    # faces:
    # 0, 1, 2,
    # 3, 4, 5,
    # 6, 7, 8
    faces = np.stack((
        faces[:-1],  # 0, 1, 2,
        np.roll(faces, -1, axis=0)[:-1],  # 3, 4, 5,
        np.roll(faces, -1, axis=1)[:-1],  # 1, 2, 0
        np.roll(np.roll(faces, -1, axis=0)[:-1], -1, axis=1),  # 4, 5, 3
        np.roll(faces, -1, axis=1)[:-1],  # 1, 2, 0
        np.roll(faces, -1, axis=0)[:-1],  # 3, 4, 5
    ), axis=2).reshape((-1, 3))

    # mesh = Trimesh(vertices=xyz.reshape((-1, 3)), faces=faces.reshape((-1, 3)))
    # mesh.show()

    return Dct(vertices=xyz, uvs=uvs, faces=faces, vertex_normals=normals)


def create_unit_cube():
    """create cube centered on origin with side length 1"""
    # x: side, y: front-back, z: up-down
    xyz = np.array([[0, 0, 0],  # 0, bottom
                    [1, 0, 0],  # 1
                    [1, 1, 0],  # 2
                    [0, 1, 0],  # 3
                    [0, 0, 0],  # 4 front
                    [1, 0, 0],  # 5
                    [1, 0, 1],  # 6
                    [0, 0, 1],  # 7
                    [1, 0, 0],  # 8 right
                    [1, 1, 0],  # 9
                    [1, 1, 1],  # 10
                    [1, 0, 1],  # 11
                    [1, 1, 0],  # 12 back
                    [0, 1, 0],  # 13
                    [0, 1, 1],  # 14
                    [1, 1, 1],  # 15
                    [0, 1, 0],  # 16 left
                    [0, 0, 0],  # 17
                    [0, 0, 1],  # 18
                    [0, 1, 1],  # 19
                    [0, 0, 1],  # 20 top
                    [1, 0, 1],  # 21
                    [1, 1, 1],  # 22
                    [0, 1, 1],  # 23
                    ], dtype=np.float32)
    normals = np.array(([[ 0,  0, -1]] * 4) +  # bottom
                       ([[ 0, -1,  0]] * 4) +  # front
                       ([[ 1,  0,  0]] * 4) +  # right
                       ([[ 0,  1,  0]] * 4) +  # back
                       ([[-1,  0,  0]] * 4) +  # left
                       ([[ 0,  0,  1]] * 4)    # top
                       )
    uvs = np.array([[0.25, 0.375],  # 0, bottom
                    [0.50, 0.375],  # 1
                    [0.50, 0.125],  # 2
                    [0.25, 0.125],  # 3
                    [0.25, 0.375],  # 4 front
                    [0.50, 0.375],  # 5
                    [0.50, 0.625],  # 6
                    [0.25, 0.625],  # 7
                    [0.50, 0.375],  # 8 right
                    [0.75, 0.375],  # 9
                    [0.75, 0.625],  # 10
                    [0.50, 0.625],  # 11
                    [0.75, 0.375],  # 12 back
                    [1.00, 0.375],  # 13
                    [1.00, 0.625],  # 14
                    [0.75, 0.625],  # 15
                    [0.00, 0.375],  # 16 left
                    [0.25, 0.375],  # 17
                    [0.25, 0.625],  # 18
                    [0.00, 0.625],  # 19
                    [0.25, 0.625],  # 20 top
                    [0.50, 0.625],  # 21
                    [0.50, 0.875],  # 22
                    [0.25, 0.875],  # 23
                    ])

    # from pysmtb.plotting import text3
    # th = text3(xyz)
    xyz = xyz - np.r_[0.5, 0.5, 0.5][None]

    faces = np.array([
        [0, 2, 1],  # bottom
        [2, 0, 3],
        [4, 5, 6],  # front
        [6, 7, 4],
        [8, 9, 10],  # right
        [10, 11, 8],
        [12, 13, 14],  # back
        [14, 15, 12],
        [16, 17, 18],  # left
        [18, 19, 16],
        [20, 21, 22],  # top
        [22, 23, 20],
                      ])
    # mesh = Trimesh(vertices=xyz.reshape((-1, 3)), faces=faces.reshape((-1, 3)))
    # mesh.show()

    return Dct(vertices=xyz, uvs=uvs, faces=faces, vertex_normals=normals)


def get_bbox(inp, dims: tuple = None):
    """get bounding box of vertices"""
    bbox = None
    if isinstance(inp, Scene):
        return inp.bounds

    if isinstance(inp, list):
        for mesh in inp:
            bbox_ = get_bbox(mesh)
            bbox = bbox_ if bbox is None else  np.array([np.minimum(bbox[0, :], bbox_[0, :]),
                                                         np.maximum(bbox[1, :], bbox_[1, :])])
        return bbox

    if isinstance(inp, Trimesh):
        inp = inp.vertices
    elif isinstance(inp, dict):
        inp = inp['vertices']
    elif not isinstance(inp, np.ndarray):
        raise Exception('input must be trimesh.Trimesh, trimesh.Scene or list of dicts with vertices key')

    if dims is None:
        dims = dims_execpt(inp, find_dim(inp, size=3))
    lower = np.min(inp, axis=dims).squeeze()
    upper = np.max(inp, axis=dims).squeeze()
    bbox = np.r_[lower[None], upper[None]]
    return bbox


def get_bbox_corners(bbox):
    """construct 8 corner points given extreme diagonal points of AABB"""
    xl, yl, zl = bbox[0]
    xu, yu, zu = bbox[1]
    corners = np.r_[np.r_[xl, yl, zl][None],
                    np.r_[xu, yl, zl][None],
                    np.r_[xu, yu, zl][None],
                    np.r_[xl, yu, zl][None],
                    np.r_[xl, yl, zu][None],
                    np.r_[xu, yl, zu][None],
                    np.r_[xu, yu, zu][None],
                    np.r_[xl, yu, zu][None]
                    ]
    return corners


def trimesh_scene_to_dicts(scene: Scene):
    """convert trimesh.Scene into list of dicts with vertices, faces and uv coordinates"""
    output = []
    for mesh in list(scene.geometry.values()):
        output.append(trimesh_to_dict(mesh))
    return output


def trimesh_to_dict(mesh: Trimesh):
    """extract vertices, faces and uv coordinates from trimesh.Trimesh and return them in dict"""
    return Dct(vertices=get_vertices(mesh),
               faces=get_faces(mesh),
               uvs=get_uv_coords(mesh),
               vertex_normals=get_vertex_normals(mesh))


def cam_extrinsics(cam):
    """compute camera's extrinsics matrix given position, lookat and up vectors in the input dict"""
    if not hasattr(cam, 'forward'):
        cam.forward = normalize(cam.lookat - cam.position)

    if not hasattr(cam, 'up'):
        cam.up = np.r_[0., 1., 0.]

    # re-orthonormalization
    cam_side = normalize(np.cross(cam.up, cam.forward))
    cam.up = normalize(np.cross(cam.forward, cam_side))

    # set up extrinsics matrix
    cam.extrinsics = np.r_[np.r_[np.r_[cam_side[None], cam.up[None], cam.forward[None]].T, np.r_[0., 0., 0.][None]].T, np.r_[cam.position, 1.][None]].T

    return cam


def set_tight_cam(vertices, cam: dict = None, use_bbox: bool = True, visualize: bool = False):
    # backward compatibility
    warn('set_tight_cam() is deprecated and should be replaced by cam_auto_zoom()')
    return cam_auto_zoom(vertices=vertices, cam=cam, use_bbox=use_bbox, visualize=visualize)


def cam_auto_zoom(vertices, cam: dict = None, use_bbox: bool = True, visualize: bool = False, debug: bool = False):
    """automatically set camera focal length to enclose a scene (either all vertices, or their axis aligned bounding
    box) in the view frustum"""
    if not isinstance(vertices, np.ndarray) and not isinstance(vertices, list):
        raise Exception('vertices must be nv x 3 np.ndarray or list of such arrays')

    bbox = get_bbox(vertices)
    bbox_diam = np.linalg.norm(np.diff(bbox, axis=0))
    scene_center = np.mean(bbox, axis=0)

    if use_bbox:
        # project only scene bounding box corners, not the vertices themselves to estimate the camera's FOV
        # (less accurate but faster)
        vertices = get_bbox_corners(bbox)

    if isinstance(vertices, list):
        vertices = np.concatenate(vertices, axis=0)

    # set up camera if not provided: lookat: scene center, position: random direction & distance from lookat
    if cam is None:
        cam = Dct(res_x=512, res_y=256)
    cam = Dct(cam)

    if not hasattr(cam, 'cx'):
        cam.cx = cam.res_x / 2.
        cam.cy = cam.res_y / 2.

    if not hasattr(cam, 'extrinsics'):
        if not hasattr(cam, 'position'):
            random_dir = normalize(np.random.rand(3) - 0.5)
            random_dist = (bbox_diam / 2 + 10. * np.random.rand())
            cam.position = scene_center + random_dist * random_dir

        if not hasattr(cam, 'lookat'):
            cam.lookat = scene_center

        if not hasattr(cam, 'up'):
            cam.up = np.r_[0., 1., 0.]

        # compute extrinsics matrix from position, lookat and up vectors
        cam = cam_extrinsics(cam)

    # bring bbox corners into camera space
    corners_cam = (np.linalg.inv(cam.extrinsics) @ np.r_[vertices.T, np.ones((vertices.shape[0], 1)).T]).T[:, :3]
    if np.abs(corners_cam[:, 2]).min() == 0.0:
        raise Exception('camera inside object, cannot automatically set fov / focal length')

    # perspective projection
    corners_projected = corners_cam / corners_cam[:, 2:3]
    # compute bbox of projected points in image plane
    bbox_projected = np.r_[np.min(corners_projected, axis=0)[None], np.max(corners_projected, axis=0)[None]]

    # at unit focal length (image plane) compute sensor size necessary to enclose projected scene bbox
    min_sensor_width = 2 * np.maximum(np.abs(bbox_projected[0, 0]), bbox_projected[1, 0])
    min_sensor_height = 2 * np.maximum(np.abs(bbox_projected[0, 1]), bbox_projected[1, 1])
    xy_ratio = min_sensor_width / min_sensor_height

    # account for physical aspect ratio
    film_xy_ratio = cam.res_x / cam.res_y

    # adjust sensor dimension with its actual aspect ratio
    if xy_ratio > film_xy_ratio:
        # x too wide --> increase y
        min_sensor_height = min_sensor_width / film_xy_ratio
    else:
        # y too high --> increase x
        min_sensor_width = film_xy_ratio * min_sensor_height
    cam.sensor_width = min_sensor_width
    cam.sensor_height = min_sensor_height

    cam.fov_x_deg = np.rad2deg(2 * np.arctan(cam.sensor_width / 2))
    cam.fov_y_deg = np.rad2deg(2 * np.arctan(cam.sensor_height / 2))

    # now compute focal length for a 36mm sensor (along its larger axis)
    if cam.res_x > cam.res_y:
        film_width = 1  # 36
        film_height = film_width * cam.res_y / cam.res_x
    else:
        film_height = 1  # 36
        film_width = film_height * cam.res_x / cam.res_y
    cam.pix_per_mm_x = cam.res_x / film_width
    cam.pix_per_mm_y = cam.res_y / film_height

    cam.fx_mm = film_width / (2 * np.tan(np.deg2rad(cam.fov_x_deg) / 2))
    cam.fy_mm = film_height / (2 * np.tan(np.deg2rad(cam.fov_y_deg) / 2))
    cam.fx = cam.pix_per_mm_x * cam.fx_mm
    cam.fy = cam.pix_per_mm_y * cam.fy_mm

    cam.intrinsics = np.array([[cam.fx, 0., cam.cx], [0., cam.fy, cam.cy], [0., 0., 1.]])

    if debug:
        output = Dct(cam=cam, corners_projected=corners_projected, bbox_projected=bbox_projected)
    else:
        output = cam

    if visualize:
        corners_bbox_projected_world = (cam.extrinsics @ np.r_[get_bbox_corners(bbox_projected).T, np.ones((8, 1)).T]).T[:, :3]
        ray_origins, ray_dirs = create_rays(cam)
        corners_projected_world = (cam.extrinsics @ np.r_[corners_projected.T, np.ones((corners_projected.shape[0], 1)).T])[:3].T
        from pysmtb.plotting import plot3, quiver3, scatter3
        sh = scatter3(cam.position, marker='s', color=np.r_[0, 0, 0])
        ax = sh.axes
        quiver3(cam.position[None], cam.up[None], color=np.r_[0.6, 0.6, 0.6], axes=ax)
        quiver3(cam.position[None], scene_center[None] - cam.position[None], color=np.r_[0.2, 0.2, 0.2], axes=ax)
        plot3(corners_bbox_projected_world[np.r_[np.r_[:4], 0]].T, axes=ax, color=np.r_[0.5, 0.4, 0.0])
        if use_bbox:
            for ci in range(8):
                plot3(np.r_[vertices[ci:ci+1, :], corners_projected_world[ci:ci+1, :], cam.position[None]].T, axes=ax, color=0.85 * np.r_[1, 1, 1])
            scatter3(vertices, axes=ax, color=np.r_[0.1, 0.4, 0.85])
            plot3(vertices[np.r_[np.r_[:4], 0, np.r_[4:8], 4, 5, 1, 2, 6, 7, 3]].T, axes=ax, color=np.r_[0.1, 0.4, 0.85])
            scatter3(corners_projected_world, axes=ax, color=np.r_[0.85, 0.6, 0.1])
            plot3(corners_projected_world[np.r_[np.r_[:4], 0, np.r_[4:8], 4, 5, 1, 2, 6, 7, 3]].T, axes=ax, color=np.r_[0.85, 0.6, 0.1])
        else:
            scatter3(vertices[::123, :], marker='.', axes=ax, color=np.r_[0.1, 0.4, 0.85])
            scatter3(corners_projected_world[::123, :], marker='.', axes=ax, color=np.r_[0.85, 0.6, 0.1])

        scatter3(ray_origins[::3*123] + ray_dirs[::3*123], marker='.', axes=ax)
        # quiver3(ray_origins[::123], ray_dirs[::123], axes=ax)
    return output


def cam_auto_zoom_trajectory(vertices, cam_positions, cam: dict = None, use_bbox: bool = True, visualize: bool = False):
    """given scene vertices and an N x 3 array of camera positions, as well as a camera dict with initial intrisics
    (resolution, principal point, ...), set the camera's focal length to ensure the projected scene is always contained
    in the view frustum; returns cam dict with fx and fy set to the mininum over all focal lengths, as well as a list of
    extrinsics matrices corresponding to the camera positions"""
    cam_orig = deepcopy(cam)
    lookat = np.mean(get_bbox(vertices), axis=0)
    fx = []
    fy = []
    extrinsics = []
    centers = []
    corners_projected = []
    bbox_projected = []
    for cam_pos in cam_positions:
        cam = deepcopy(cam_orig)
        cam.position = cam_pos
        cam.lookat = lookat
        cam = cam_extrinsics(cam)

        debug = cam_auto_zoom(vertices=vertices, cam=cam, use_bbox=use_bbox, debug=True)
        corners_projected.append(debug.corners_projected)
        bbox_projected.append(debug.bbox_projected)
        cam = debug.cam
        centers.append(cam.lookat)
        fx.append(cam.fx)
        fy.append(cam.fy)
        extrinsics.append(cam.extrinsics)
    # final focal length is the minimum (largest field of view) over each position's focal length
    cam.fx = np.min(fx)
    cam.fy = np.min(fy)

    if visualize:
        from pysmtb.plotting import scatter3
        sh = scatter3(vertices, '.')
        ax = sh.axes
        scatter3(cam_positions, 's', axes=ax)
        scatter3(np.array([(ex @ np.concatenate((bbp, np.ones((bbp.shape[0], 1))), axis=1).T).T[:, :3] for ex, bbp in
                           zip(extrinsics, bbox_projected)]).reshape((-1, 3)), '.', axes=ax)

    return cam, extrinsics


def create_rays(cam, repeat_origin=True, xs: np.ndarray = None, ys: np.ndarray = None):
    """given camera specifications (resolution in x and y, extrinsics and intrinsics), compute ray origins and
    directions for the full pixel raster (or optionally a patch specified by x and y ranges)"""
    if xs is None:
        xs = np.r_[0.5: float(cam.res_x) + 0.5]
    if ys is None:
        ys = np.r_[0.5: float(cam.res_y) + 0.5]
    xs, ys = np.meshgrid(xs, ys)
    pts_pixels = np.stack((xs.ravel(), ys.ravel(), np.ones(xs.size)), axis=0).astype(np.float32)

    # construct ray origins and directions
    ray_origins = np.array(cam.extrinsics @ np.r_[0, 0, 0, 1])[None, :3].astype(np.float32)
    if repeat_origin:
        ray_origins = ray_origins.repeat(xs.size, axis=0)
    ray_dirs = (np.linalg.inv(cam.intrinsics) @ pts_pixels).T
    ray_dirs = normalize(ray_dirs, axis=1)
    ray_dirs = (cam.extrinsics[:3, :3] @ ray_dirs.T).astype(np.float32).T
    return ray_origins, ray_dirs


def embree_create_scene(meshes,
                        with_tangent_frames: bool = True,
                        scene: EmbreeScene = None,
                        cam: dict = None,
                        auto_cam: bool = True,
                        auto_cam_bbox: bool = True,
                        auto_cam_visualize: bool = False):
    """given a scene as a list of meshes, construct a pyembree scene, optionally adjust the camera to automatically
    focus the scene, returns embree scene and camera"""

    if scene is None:
        # create new scene
        scene = EmbreeScene()

    if isinstance(meshes, Trimesh):
        meshes = meshes.scene()

    if isinstance(meshes, Scene):
        meshes = trimesh_scene_to_dicts(meshes)

    if isinstance(meshes, dict):
        meshes = [meshes]

    if not isinstance(meshes, list):
        raise Exception('input of type ' + str(type(meshes)) + ' is not supported')

    # precompute per-vertex tangent frames
    if with_tangent_frames:
        for mesh_id in range(len(meshes)):
            meshes[mesh_id] = get_tangent_frames(meshes[mesh_id])

    if auto_cam and auto_cam_bbox:
        bbox = get_bbox(meshes)
    all_vertices = []
    for mesh in meshes:
        TriangleMesh(scene, mesh['vertices'], mesh['faces'])
        if auto_cam and not auto_cam_bbox:
            all_vertices.append(mesh.vertices)

    if auto_cam:
        cam = cam_auto_zoom(bbox if auto_cam_bbox else all_vertices, cam=cam, use_bbox=auto_cam_bbox, visualize=auto_cam_visualize)
    return scene, cam


def embree_intersect_scene(scene: EmbreeScene, cam: dict, return_as_rasters: bool = False):
    """given a pyembree scene and a camera dict, shoot rays and intersect the scene, returning a dict with either
    N x C or H x W x C arrays if return_as_rasters == True; dict has keys:
    hit: binary mask indicating hit / miss (always H x W x 1)
    tfar: depth in camera space, 1e37 for missed pixels
    Ng: (unnormalized!) geometric (per-face) normal vectors
    u, v: barycentric coordinates for each intersected triangle
    primID: triangle ID
    geomID: sub-mesh ID
    """
    cam = Dct(cam)
    res_x = cam.res_x
    res_y = cam.res_y
    intrinsics = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]])

    if hasattr(cam, 'extrinsics'):
        extrinsics = cam.extrinsics
    else:
        extrinsics = np.eye(4)

    # generate pixel array
    xs, ys = np.meshgrid(np.r_[0.5: float(res_x) + 0.5],
                         np.r_[0.5: float(res_y) + 0.5])
    pts_pixels = np.stack((xs.ravel(), ys.ravel(), np.ones(xs.size)), axis=0).astype(np.float32)
    num_pix = res_y * res_x

    # construct ray origins and directions
    ray_origins = np.array(extrinsics @ np.r_[0, 0, 0, 1])[None, :3].astype(np.float32).repeat(xs.size, axis=0)
    ray_dirs = (np.linalg.inv(intrinsics) @ pts_pixels).T
    ray_dirs = normalize(ray_dirs, axis=1)
    ray_dirs = (extrinsics[:3, :3] @ ray_dirs.T).astype(np.float32).T

    # trace rays
    its = scene.run(ray_origins, ray_dirs, output=1)
    its = Dct(its)

    # compute hit mask
    hit = its.tfar < 1e37
    # always reshape hit mask to bitmap resolution
    its.hit = hit.reshape((res_y, res_x, 1))

    # normalize normals
    its.Ng[hit, :] = normalize(its.Ng[hit, :], axis=1)

    # reshape all fields to image size, optionally multiply with hit mask to set missed pixels to 0
    # geomIDs are 0-based so we cannot overwrite their background with 0
    for key in its.keys():
        if return_as_rasters:
            its[key] = its[key].reshape((res_y, res_x, -1))
        elif key not in ['hit']:
            its[key] = its[key].reshape((num_pix, -1))
            its[key] = its[key][hit, :]

    # compute intersection points
    its['points'] = ray_origins[hit, :] + its['tfar'] * ray_dirs[hit, :]
    return its


def interpolate_vertex_attributes(its, meshes, return_as_rasters: bool = False):
    """given intersection dict with barycentric coordinates ('u', 'v'), geometry IDs and triangle IDs, as well as the
    corresponding list of meshes, compute texture coordinates coordinates for each intersected pixel"""

    if isinstance(meshes, Scene):
        meshes = list(meshes.geometry.values())
    elif not isinstance(meshes, list):
        raise Exception('meshes should be list of trimesh.Trimesh objects or one trimesh.Scene object')

    # compute uv coordinates
    hit = its['hit']
    num_hit = np.count_nonzero(hit)
    its['uvs'] = np.zeros((num_hit, 2), dtype=np.float32)
    its['tangent_frames'] = np.zeros((num_hit, 3, 3), dtype=np.float32)
    for mesh_id in range(len(meshes)):
        # select only those intersections corresponding to the current mesh
        mesh_mask = its['geomID'] == mesh_id

        # barycentric coordinates are used to interpolate vertex attributes
        u = its['u'][mesh_mask]
        v = its['v'][mesh_mask]
        uvw = np.stack([1 - u - v, u, v], axis=0)

        # get per-triangle-vertex attributes (currently uv coordinates and tangent frames)
        triangle_ids = its['primID'][mesh_mask]
        its_vertex_ids = get_faces(meshes[mesh_id])[triangle_ids, :].T
        face_vertex_uvs = get_uv_coords(meshes[mesh_id])[its_vertex_ids, :]
        meshes[mesh_id] = get_tangent_frames(meshes[mesh_id])
        face_vertex_tangent_frames = meshes[mesh_id].vertex_tangent_frames[its_vertex_ids, :, :]  # 3 x (3 * NF) x 3 x 3

        # interpolate per pixel texture coordinates
        its['uvs'][mesh_mask.flatten(), :] = np.sum(face_vertex_uvs * uvw[..., None], axis=0)
        # interpolate per pixel tangent frames
        its['tangent_frames'][mesh_mask.flatten(), :, :] = np.sum(face_vertex_tangent_frames * uvw[:, :, None, None], axis=0)

    if return_as_rasters:
        # reshape to image size
        its['uvs'] = its['uvs'].reshape(its['u'].shape[:2] + (2, ))
        its['tangent_frames'] = its['tangent_frames'].reshape(its['u'].shape[:2] + (3, 3))
    return its


def get_local_dirs(its, global_position, normalized=True):
    """given intersection dict with per-pixel interpolated tangent frames and intersection points, as well as a global
    positions (e.g. light or view), compute per pixel local directions and unnormalized global directions (e.g., for
    computing light attenuation)"""
    dirs_global = global_position[None, :] - its['points']
    frames = its['tangent_frames']
    dirs_local = np.einsum('nij,nj->ni', frames, dirs_global)
    if normalized:
        dirs_local = normalize(dirs_local, axis=-1)
    return Dct(dirs_local=dirs_local, dirs_global=dirs_global)


def embree_render_deferred(meshes = None,
                           with_tangent_frames: bool = True,
                           cam: dict = None,
                           auto_cam: bool = True,
                           auto_cam_bbox: bool = False,
                           light_position: tuple = (0., 10., 0.),
                           return_as_rasters: bool = False):
    """given a list of meshes, optionally a camera and a point light position, render the scene via ray tracing;
    returns pixel buffers with:
    - 3D intersection points
    -
    - view and light directions in tangent space
    - """

    if meshes is None:
        meshes = []
        meshes.append(create_unit_sphere())
        meshes.append(create_unit_cube())

    bbox = get_bbox(meshes)
    bbox_diam = np.linalg.norm(np.diff(bbox, axis=0))

    if cam is None:
        cam = Dct(res_x=384, res_y=256)
    if not hasattr(cam, 'cx'):
        cam.cx = cam.res_x / 2.
        cam.cy = cam.res_y / 2.

    if not hasattr(cam, 'position'):
        cam.position = np.mean(bbox, axis=0) + (bbox_diam / 2 + 10. * np.random.rand()) * normalize(np.random.rand(3) - 0.5)

    # scene & camera setup
    scene, cam = embree_create_scene(meshes=meshes, with_tangent_frames=with_tangent_frames, cam=cam, auto_cam=auto_cam,
                                     auto_cam_bbox=auto_cam_bbox, auto_cam_visualize=False)

    # ray tracing
    buffers = embree_intersect_scene(scene=scene, cam=cam, return_as_rasters=return_as_rasters)

    # per pixel tangent frames
    buffers = interpolate_vertex_attributes(buffers, meshes, return_as_rasters=return_as_rasters)
    h, w = cam.res_y, cam.res_x

    # local light & view directions
    view_dirs = get_local_dirs(buffers, cam.position, normalized=True)
    light_dirs = get_local_dirs(buffers, light_position, normalized=True)

    if return_as_rasters:
        mask = buffers['hit']
        buffers['view_dirs_local'] = assign_masked(mask, view_dirs.pop('dirs_local'))
        buffers['view_dirs_global'] = assign_masked(mask, view_dirs.pop('dirs_global'))
        buffers['light_dirs_local'] = assign_masked(mask, light_dirs.pop('dirs_local'))
        buffers['light_dirs_global'] = assign_masked(mask, light_dirs.pop('dirs_global'))
    else:
        buffers['view_dirs_local'] = view_dirs.pop('dirs_local')
        buffers['view_dirs_global'] = view_dirs.pop('dirs_global')
        buffers['light_dirs_local'] = light_dirs.pop('dirs_local')
        buffers['light_dirs_global'] = light_dirs.pop('dirs_global')

    return buffers
