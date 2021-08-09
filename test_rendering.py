from copy import deepcopy
import numpy as np
from tqdm import tqdm
import trimesh
from trimesh import Trimesh
from trimesh.visual import TextureVisuals
from pysmtb.rendering import normalize, create_unit_sphere, create_unit_cube, \
    embree_intersect_scene, interpolate_vertex_attributes, get_local_dirs, get_bbox
from pysmtb.rendering import embree_create_scene, embree_render_deferred, cam_auto_zoom_trajectory
from pysmtb.plotting import plot3, quiver3, scatter3, text3
from pysmtb.utils import Dct, assign_masked

def test_rendering():
    # given initial camera position, rotate camera on a trajectory around the scene, setting focal length such that the
    # entire scene is enclosed on the camera sensor at each position
    meshes = []
    meshes.append(create_unit_sphere(40))
    cube = create_unit_cube()
    # create shifted copies of cube
    cube.vertices += np.r_[1.5, 0., 0.][None]
    meshes.append(deepcopy(cube))
    cube.vertices += np.r_[-3, 0., 1.][None]
    meshes.append(deepcopy(cube))
    cube.vertices += np.r_[0, -2., -2.][None]
    meshes.append(deepcopy(cube))
    cube.vertices += np.r_[-1, 2., 0.5][None]
    meshes.append(deepcopy(cube))
    all_vertices = np.concatenate([mesh.vertices for mesh in meshes], axis=0)

    bbox = get_bbox(meshes)
    bbox_diam = np.linalg.norm(np.diff(bbox, axis=0))

    # automatically adjust camera for scene
    cam = Dct(res_x=384, res_y=256)
    num_positions = 15
    angles = np.linspace(0, 2 * np.pi, num_positions + 1)[:-1]
    cam_dirs = np.stack((np.cos(angles), np.zeros_like(angles), np.sin(angles)), axis=1)
    cam_distance = 2 * bbox_diam
    cam_positions = np.mean(bbox, axis=0, keepdims=True) + cam_distance * cam_dirs
    cam, extrinsics = cam_auto_zoom_trajectory(vertices=all_vertices, cam_positions=cam_positions, cam=cam,
                                               use_bbox=False)

    # ray trace scene, writing one of the tracing buffers instead of evaluating reflectance for renderings
    light_pos = cam_positions[0]
    masks = []
    renderings = []
    for i in tqdm(range(num_positions), 'rendering'):
        cam.position = cam_positions[i]
        cam.extrinsics = extrinsics[i]
        buffers = embree_render_deferred(meshes, with_tangent_frames=True, cam=cam, auto_cam=False,
                                         light_position=light_pos)
        masks.append(buffers.hit)
        renderings.append(assign_masked(buffers.hit, buffers.Ng))
    return renderings, masks, buffers

if __name__ == '__main__':
    from pysmtb.iv import iv
    import matplotlib.pyplot as plt
    plt.switch_backend('qt5agg')
    plt.ion()

    # given initial camera position, rotate camera on a trajectory around the scene, setting focal length such that the
    # entire scene is enclosed on the camera sensor at each position
    meshes = []
    ply_cube = trimesh.load('data/cube.ply')
    ply_cube = ply_cube.apply_translation(np.r_[1., 1., 1.])
    meshes.append(ply_cube)
    meshes.append(create_unit_sphere(40))
    cube = create_unit_cube()
    # create shifted copies of cube
    cube.vertices += np.r_[1.5, 0., 0.][None]
    meshes.append(deepcopy(cube))
    cube.vertices += np.r_[-3, 0., 1.][None]
    meshes.append(deepcopy(cube))
    cube.vertices += np.r_[0, -2., -2.][None]
    meshes.append(deepcopy(cube))
    cube.vertices += np.r_[-1, 2., 0.5][None]
    meshes.append(deepcopy(cube))
    all_vertices = np.concatenate([mesh.vertices for mesh in meshes], axis=0)

    bbox = get_bbox(meshes)
    bbox_diam = np.linalg.norm(np.diff(bbox, axis=0))

    # automatically adjust camera for scene
    cam = Dct(res_x=384, res_y=256)
    num_positions = 15
    angles = np.linspace(0, 2 * np.pi, num_positions + 1)[:-1]
    cam_dirs = np.stack((np.cos(angles), np.zeros_like(angles), np.sin(angles)), axis=1)
    cam_distance = 2 * bbox_diam
    cam_positions = np.mean(bbox, axis=0, keepdims=True) + cam_distance * cam_dirs
    cam, extrinsics = cam_auto_zoom_trajectory(vertices=all_vertices, cam_positions=cam_positions, cam=cam, use_bbox=False)

    # ray trace scene, writing one of the tracing buffers instead of evaluating reflectance for renderings
    light_pos = cam_positions[0]
    renderings = []
    for i in tqdm(range(num_positions), 'rendering'):
        cam.position = cam_positions[i]
        cam.extrinsics = extrinsics[i]
        buffers = embree_render_deferred(meshes, with_tangent_frames=True, cam=cam, auto_cam=False, light_position=light_pos)
        renderings.append(buffers.hit)
        renderings.append(assign_masked(buffers.hit, buffers.Ng))
        # renderings.append(assign_masked(buffers.hit, buffers.uvs))

    v = iv(renderings[1::2])

    # test individual steps
    # load / create geometry
    sphere = create_unit_sphere(40)
    cube = create_unit_cube()
    # shift cube to the right
    cube.vertices += np.r_[1.5, 0., 0.][None]

    # visualize "scene"
    sh = scatter3(sphere['vertices'], '.')
    ax = sh.axes
    scatter3(cube['vertices'], '.', axes=ax)
    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])
    ax.set_zlim([-1, 2])
    ax.set_box_aspect([1., 1., 1.])

    # trimesh would be nice to use but has limitations regarding per-face texture coordinates (see above), hence we use
    # lists of dicts to represent scenes
    # meshes = Trimesh(vertices=sphere.vertices, faces=sphere.faces, visual=TextureVisuals(uv=sphere.uvs)).scene()
    # meshes += Trimesh(vertices=cube.vertices, faces=cube.faces, visual=TextureVisuals(uv=cube.uvs)).scene()
    # to load from disk
    # meshes = trimesh.load('sphere.obj').scene()
    # meshes += trimesh.load('cube.obj').scene()
    meshes = []
    meshes.append(sphere)
    meshes.append(cube)

    bbox = get_bbox(meshes)
    bbox_diam = np.linalg.norm(np.diff(bbox, axis=0))

    # ray trace scene (after automatic camera adjustments)
    cam = Dct(res_x=384, res_y=256)
    # cam.position = np.r_[0.68372341, -2.05557506, -1.5378463]

    cam.position = np.mean(bbox, axis=0) + (bbox_diam / 2 + 10. * np.random.rand()) * normalize(np.random.rand(3) - 0.5)

    buffers = embree_render_deferred(meshes, with_tangent_frames=True, cam=cam, auto_cam=True, auto_cam_bbox=True, light_position=np.r_[0., 10., 0.])
    view_dirs_local = assign_masked(buffers['hit'], buffers['view_dirs_local'])
    light_dirs_local = assign_masked(buffers['hit'], buffers['light_dirs_local'])

    iv(view_dirs_local, light_dirs_local)

    # manually render for a bunch of random camera positions
    np.random.seed(seed=0)
    buffers = []
    for i in range(16):
        cam = Dct(res_x=128, res_y=96)
        cam.cx = cam.res_x / 2.
        cam.cy = cam.res_y / 2.
        cam.position = np.mean(bbox, axis=0) + (bbox_diam / 2 + 10. * np.random.rand()) * normalize(np.random.rand(3) - 0.5)

        # set up EmbreeScene
        scene, cam = embree_create_scene(meshes=meshes, cam=cam, auto_cam=True, auto_cam_bbox=True, auto_cam_visualize=False)

        # trace rays
        buffers.append(embree_intersect_scene(scene=scene, cam=cam))

        # get per pixel interpolated vertex attributes (texture coordinates & tangent frames)
        buffers[-1] = interpolate_vertex_attributes(buffers[-1], meshes)

        # compute global & local view directions
        view_dirs = get_local_dirs(buffers[-1], cam.position, normalized=True)
        buffers[-1]['view_dirs_local'] = view_dirs['dirs_local']
        buffers[-1]['view_dirs_global'] = view_dirs['dirs_global']

        # also compute global & local light directions
        light_dirs = get_local_dirs(buffers[-1], np.r_[0., 10., 0.], normalized=True)
        buffers[-1]['light_dirs_local'] = light_dirs['dirs_local']
        buffers[-1]['light_dirs_global'] = light_dirs['dirs_global']

        # num_hits x c --> res_y x res_x x c buffers
        for key in buffers[-1].keys():
            if key in ['hit']:
                continue
            buffers[-1][key] = assign_masked(buffers[-1]['hit'], buffers[-1][key])

    from pysmtb.utils import collage

    # iv(dict(tangents=collage([i['tangent_frames'][:, :, 0, :] for i in buffers], crop=True),
    #         bitangents=collage([i['tangent_frames'][:, :, 1, :] for i in buffers], crop=True),
    #         normals=collage([i['tangent_frames'][:, :, 2, :] for i in buffers], crop=True),
    #         global_normals=-collage([i['Ng'] for i in buffers], crop=True)))
    #
    # iv(collage([i['view_dirs_local'] for i in buffers], crop=True),
    #    collage([i['view_dirs_global'] for i in buffers], crop=True),
    #    collage([np.linalg.norm(i['view_dirs_global'], axis=2) for i in buffers], crop=True))
    #
    # iv(dict(uvs=collage([i['uvs'] for i in buffers], crop=True),
    #         barycentric=collage([np.concatenate((i['u'], i['v']), axis=2) for i in buffers], crop=True)))
    # iv(collage([i['tfar'] for i in buffers], crop=True),
    #    collage([i['points'] for i in buffers], crop=True),
    #    collage([i['primID'] for i in buffers], crop=True, crop_value=0, bv=0),
    #    collage([i['hit'] for i in buffers], crop=True, crop_value=0, bv=0))
    # iv(collage([(i['geomID'] + 1) * i['hit'] for i in buffers], crop=True, crop_value=-1, bv=-1))

    import imageio

    def blend(alpha, im1, im2=None, v2=1.):
        alpha = alpha.astype(np.float32)
        if im2 is None:
            im2 = v2 * np.ones_like(im1)
        return im1 * alpha + im2 * (1 - alpha)

    def tm(im, gamma=1.):
        lower = np.min(im)
        upper = np.max(im)
        im = np.atleast_3d(im)
        if im.shape[2] == 2:
            im = np.concatenate((im, np.zeros(im.shape[:2])[..., None]), axis=2)
        return np.clip((im - lower) / (upper - lower), 0, 1) ** (1 / gamma)

    def imwrite(fn, im, gamma=1.):
        imageio.imwrite(fn, (255 * tm(im, gamma=gamma)).astype(np.uint8))

    imwrite('/data/repos/pysmtb/examples/geometry_vertex_tangents.png', collage([i['tangent_frames'][:, :, 0, :] for i in buffers], crop=True))
    imwrite('/data/repos/pysmtb/examples/geometry_vertex_bitangents.png', collage([i['tangent_frames'][:, :, 1, :] for i in buffers], crop=True))
    imwrite('/data/repos/pysmtb/examples/geometry_vertex_normals.png', collage([i['tangent_frames'][:, :, 2, :] for i in buffers], crop=True))
    imwrite('/data/repos/pysmtb/examples/geometry_face_normals.png', collage([i['Ng'] for i in buffers], crop=True))
    imwrite('/data/repos/pysmtb/examples/geometry_view_dirs_local.png', collage([i['view_dirs_local'] for i in buffers], crop=True))
    # imwrite('/data/repos/pysmtb/examples/geometry_view_dirs_global.png', collage([i['view_dirs_global'] for i in buffers], crop=True))
    imwrite('/data/repos/pysmtb/examples/geometry_view_dirs_global.png', collage([blend(i['hit'], tm(i['view_dirs_global'])) for i in buffers], crop=True, crop_value=1., bv=1.))
    imwrite('/data/repos/pysmtb/examples/geometry_light_dirs_local.png', collage([i['light_dirs_local'] for i in buffers], crop=True))
    # imwrite('/data/repos/pysmtb/examples/geometry_light_dirs_global.png', collage([i['light_dirs_global'] for i in buffers], crop=True))
    imwrite('/data/repos/pysmtb/examples/geometry_light_dirs_global.png', collage([blend(i['hit'], tm(i['light_dirs_global'])) for i in buffers], crop=True, crop_value=1., bv=1.))
    imwrite('/data/repos/pysmtb/examples/geometry_depth.png', collage([np.linalg.norm(i['view_dirs_global'], axis=2) for i in buffers], crop=True))
    imwrite('/data/repos/pysmtb/examples/geometry_tfar.png', collage([i['tfar'] for i in buffers], crop=True))
    imwrite('/data/repos/pysmtb/examples/geometry_barycentric_coordinates.png', collage([np.concatenate((i['u'], i['v']), axis=2) for i in buffers], crop=True))
    imwrite('/data/repos/pysmtb/examples/geometry_texture_coordinates.png', collage([i['uvs'] for i in buffers], crop=True))
    imwrite('/data/repos/pysmtb/examples/geometry_points.png', collage([i['points'] for i in buffers], crop=True))
    imwrite('/data/repos/pysmtb/examples/geometry_geom_id.png', collage([(i['geomID'] + 1) * i['hit'] for i in buffers], crop=True, crop_value=0, bv=0))
    imwrite('/data/repos/pysmtb/examples/geometry_prim_id.png', collage([(i['primID'] + 1) * i['hit'] for i in buffers], crop=True, crop_value=0, bv=0), gamma=4)
    imwrite('/data/repos/pysmtb/examples/geometry_hit_mask.png', collage([i['hit'] for i in buffers], crop=True, crop_value=0, bv=0))

    print()
