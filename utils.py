import argparse
import collections
import os
import shutil

import cv2
import dominate
import imageio
import numpy as np
import ray
from numba import jit


def get_line_mask(masks, pixel, angle_id, resolution):
    p = [resolution - 1 - pixel[0], resolution - 1 - pixel[1]]
    return masks[angle_id]['mask'][p[0]: p[0] + resolution, p[1]: p[1] + resolution]

def get_obj_mask(color_image):
    return (cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)[:, :, 2] > 100).astype(float)

def reset_envs(envs, task, task_num, task_ids=None):
    if len(envs) == 1:
        max_cover_area, cover_area, observation = envs[0].reset()
        return [max_cover_area], [cover_area], [observation]
    else:
        results = list()
        for i, env in enumerate(envs):
            idx = np.random.choice(task_num) if task_ids is None else task_ids[i]
            task_path = os.path.join(task, f'{idx}.pkl')
            results.append(env.reset.remote(task_path))
        results = ray.get(results)
        max_cover_area = [result[0] for result in results]
        cover_area = [result[1] for result in results]
        observation = [result[2] for result in results]
        return max_cover_area, cover_area, observation


def get_grasping_acitons(envs):
    results = list()
    for env in envs:
        results.append(env.get_random_grasping.remote())
    return ray.get(results)


def lift_and_stretch(envs, grasping_actions, lifting_height=0.12):
    if len(envs) == 1:
        p1, p2 = grasping_actions[0]
        lift_observation, stretch_observation, cover_area = envs[0].lift_and_stretch_primitive(p1, p2)
        return [lift_observation], [stretch_observation], [cover_area]
    else:
        results = list()
        for env, grasping_action in zip(envs, grasping_actions):
            p1, p2 = grasping_action
            results.append(env.lift_and_stretch_primitive.remote(p1, p2, lifting_height))
        results = ray.get(results)
        lift_observation = [result[0] for result in results]
        stretch_observation = [result[1] for result in results]
        cover_area = [result[2] for result in results]
        return lift_observation, stretch_observation, cover_area



def pick_and_place(envs, grasping_actions, lifting_height=0.12):
    if len(envs) == 1:
        p1, p2 = grasping_actions[0]
        lift_observation, stretch_observation, cover_area = envs[0].pick_and_place(p1, p2)
        return [lift_observation], [stretch_observation], [cover_area]
    else:
        results = list()
        for env, grasping_action in zip(envs, grasping_actions):
            p1, p2 = grasping_action
            results.append(env.pick_and_place.remote(p1, p2, lifting_height))
        results = ray.get(results)
        lift_observation = [result[0] for result in results]
        stretch_observation = [result[1] for result in results]
        cover_area = [result[2] for result in results]
        return lift_observation, stretch_observation, cover_area



def blow(envs, blow_actions, blow_time):
    if len(envs) == 1:
        position, orientation = blow_actions[0][:3], blow_actions[0][3:]
        cover_area, observation = envs[0].blow(position, orientation)
        return [cover_area], [observation]
    else:
        results = list()
        for env, blow_action in zip(envs, blow_actions):
            position, orientation = blow_action[:3], blow_action[3:]
            results.append(env.blow.remote(
                position, orientation,
                num_layer=2,
                alpha=5.0,
                velocity=5,
                mass=0.1,
                step_num=blow_time
            ))
        results = ray.get(results)
        cover_area = [result[0] for result in results]
        observation = [result[1] for result in results]
        return cover_area, observation


def fling(envs):
    if len(envs) == 1:
        cover_area, observation = envs[0].fling()
        return [cover_area], [observation]
    else:
        results = list()
        for env in envs:
            results.append(env.fling_cloth.remote())
        results = ray.get(results)
        cover_area = [result[0] for result in results]
        observation = [result[1] for result in results]
        return cover_area, observation


def place(envs):
    if len(envs) == 1:
        cover_area, observation = envs[0].place()
        return [cover_area], [observation]
    else:
        results = list()
        for env in envs:
            results.append(env.place_cloth.remote())
        results = ray.get(results)
        cover_area = [result[0] for result in results]
        observation = [result[1] for result in results]
        return cover_area, observation


def rot2d(angle, degrees=True):
    if degrees:
        angle = np.pi*angle/180
    return np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]).T


def translate2d(translation):
    return np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1],
    ]).T


def scale2d(scale):
    return np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1],
    ]).T


def get_transform_matrix(original_dim, resized_dim, scale):
    # resize
    resize_mat = scale2d(original_dim/resized_dim)
    # scale
    scale_mat = np.matmul(
        np.matmul(
            translate2d(-np.ones(2)*(resized_dim//2)),
            scale2d(scale),
        ), translate2d(np.ones(2)*(resized_dim//2)))
    return np.matmul(scale_mat, resize_mat)


def crop_center(img, crop):
    startx = img.shape[1]//2-(crop//2)
    starty = img.shape[0]//2-(crop//2)
    return img[starty:starty+crop, startx:startx+crop, ...]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_tableau_palette():
    """Get Tableau color palette (10 colors) https://www.tableau.com/.

    Returns:
        palette: 10x3 uint8 array of color values in range 0-255 (each row is a color)
    """
    palette = np.array([[ 78,121,167], # blue
                        [255, 87, 89], # red
                        [ 89,169, 79], # green
                        [242,142, 43], # orange
                        [237,201, 72], # yellow
                        [176,122,161], # purple
                        [255,157,167], # pink 
                        [118,183,178], # cyan
                        [156,117, 95], # brown
                        [186,176,172]  # gray
                        ],dtype=np.uint8)
    return palette


def transform_pointcloud(xyz_pts, rigid_transform):
    """Apply rigid transformation to 3D pointcloud.

    Args:
        xyz_pts: Nx3 float array of 3D points
        rigid_transform: 3x4 or 4x4 float array defining a rigid transformation (rotation and translation)

    Returns:
        xyz_pts: Nx3 float array of transformed 3D points
    """
    xyz_pts = np.dot(rigid_transform[:3,:3],xyz_pts.T) # apply rotation
    xyz_pts = xyz_pts+np.tile(rigid_transform[:3,3].reshape(3,1),(1,xyz_pts.shape[1])) # apply translation
    return xyz_pts.T


def get_heightmap(xyz_pts, color_pts, view_bounds, heightmap_pix_sz, zero_level):
    """Get top-down (along z-axis) orthographic heightmap image from 3D pointcloud

    Args:
        cam_pts: Nx3 float array of 3D points in world coordinates
        color_pts: Nx3 uint8 array of color values in range 0-255 corresponding to xyz_pts
        view_bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining region in 3D space of heightmap in world coordinates
        heightmap_pix_sz: float value defining size of each pixel in meters (determines heightmap resolution)
        zero_level: float value defining z coordinate of zero level (i.e. bottom) of heightmap
    
    Returns:
        depth_heightmap: HxW float array of height values (from zero level) in meters
        color_heightmap: HxWx3 uint8 array of backprojected color values in range 0-255 aligned with depth_heightmap
    """

    heightmap_size = np.round(((view_bounds[1,1]-view_bounds[1,0])/heightmap_pix_sz,
                               (view_bounds[0,1]-view_bounds[0,0])/heightmap_pix_sz)).astype(int)

    # Remove points outside workspace bounds
    heightmap_valid_ind = np.logical_and(np.logical_and(
                          np.logical_and(np.logical_and(xyz_pts[:,0] >= view_bounds[0,0],
                                                        xyz_pts[:,0] <  view_bounds[0,1]),
                                                        xyz_pts[:,1] >= view_bounds[1,0]),
                                                        xyz_pts[:,1] <  view_bounds[1,1]),
                                                        xyz_pts[:,2] <  view_bounds[2,1])
    cam_pts = xyz_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]

    # Sort points by z value (works in tandem with array assignment to ensure heightmap uses points with highest z values)
    sort_z_ind = np.argsort(cam_pts[:,2])
    cam_pts = cam_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]

    # Backproject 3D pointcloud onto heightmap
    heightmap_pix_x = np.floor((cam_pts[:,0]-view_bounds[0,0])/heightmap_pix_sz).astype(int)
    heightmap_pix_y = np.floor((cam_pts[:,1]-view_bounds[1,0])/heightmap_pix_sz).astype(int)

    # Get height values from z values minus zero level
    depth_heightmap = np.zeros(heightmap_size)
    depth_heightmap[heightmap_pix_y,heightmap_pix_x] = cam_pts[:,2]
    depth_heightmap = depth_heightmap-zero_level
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -zero_level] = 0

    # Map colors
    color_heightmap = np.zeros((heightmap_size[0],heightmap_size[1],3),dtype=np.uint8)
    for c in range(3):
        color_heightmap[heightmap_pix_y,heightmap_pix_x,c] = color_pts[:,c]
    
    return color_heightmap, depth_heightmap


def get_pointcloud(depth_img, color_img, cam_intr, cam_pose=None):
    """Get 3D pointcloud from depth image.
    
    Args:
        depth_img: HxW float array of depth values in meters aligned with color_img
        color_img: HxWx3 uint8 array of color image
        cam_intr: 3x3 float array of camera intrinsic parameters
        cam_pose: (optional) 3x4 float array of camera pose matrix
        
    Returns:
        cam_pts: Nx3 float array of 3D points in camera/world coordinates
        color_pts: Nx3 uint8 array of color points
    """

    img_h = depth_img.shape[0]
    img_w = depth_img.shape[1]

    # Project depth into 3D pointcloud in camera coordinates
    pixel_x,pixel_y = np.meshgrid(np.linspace(0,img_w-1,img_w),
                                  np.linspace(0,img_h-1,img_h))
    cam_pts_x = np.multiply(pixel_x-cam_intr[0,2],depth_img/cam_intr[0,0])
    cam_pts_y = np.multiply(pixel_y-cam_intr[1,2],depth_img/cam_intr[1,1])
    cam_pts_z = depth_img
    cam_pts = np.array([cam_pts_x,cam_pts_y,cam_pts_z]).transpose(1,2,0).reshape(-1,3)

    if cam_pose is not None:
        cam_pts = transform_pointcloud(cam_pts, cam_pose)
    color_pts = None if color_img is None else color_img.reshape(-1, 3)

    return cam_pts, color_pts


def project_pts_to_2d(pts, camera_view_matrix, camera_intrisic):
    """Project points to 2D.

    Args:
        pts: Nx3 float array of 3D points in world coordinates.
        camera_view_matrix: 4x4 float array. A wrd2cam transformation defining camera's totation and translation.
        camera_intrisic: 3x3 float array. [ [f,0,0],[0,f,0],[0,0,1] ]. f is focal length.

    Returns:
        coord_2d: Nx3 float array of 2D pixel. (w, h, d) the last one is depth
    """
    pts_c = transform_pointcloud(pts, camera_view_matrix[0:3, :])
    rot_algix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]])
    pts_c = transform_pointcloud(pts_c, rot_algix) # Nx3
    coord_2d = np.dot(camera_intrisic, pts_c.T) # 3xN
    coord_2d[0:2, :] = coord_2d[0:2, :] / np.tile(coord_2d[2, :], (2, 1))
    coord_2d[2, :] = pts_c[:, 2]
    coord_2d = np.array([coord_2d[1], coord_2d[0], coord_2d[2]])
    return coord_2d.T


def pixel_to_3d(depth_im, pix, cam_pose, cam_intr, depth_scale=1):
    cam_pts_z = depth_im[pix[:, 0], pix[:, 1]]
    cam_pts_z *= depth_scale
    cam_pts_x = (pix[:, 1]-cam_intr[0, 2]) * cam_pts_z/cam_intr[0, 0]
    cam_pts_y = (pix[:, 0]-cam_intr[1, 2]) * cam_pts_z/cam_intr[1, 1]
    cam_pts = np.array([cam_pts_x,cam_pts_y,cam_pts_z]).T
    wrd_pts = transform_pointcloud(cam_pts, cam_pose)
    return wrd_pts


def mkdir(path, clean=False):
    """Make directory.
    
    Args:
        path: path of the target directory
        clean: If there exist such directory, remove the original one or not
    """
    if clean and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)
        

def imretype(im, dtype):
    """Image retype.
    
    Args:
        im: original image. dtype support: float, float16, float32, float64, uint8, uint16
        dtype: target dtype. dtype support: float, float16, float32, float64, uint8, uint16
    
    Returns:
        image of new dtype
    """
    im = np.array(im)

    if im.dtype in ['float', 'float16', 'float32', 'float64']:
        im = im.astype(np.float)
    elif im.dtype == 'uint8':
        im = im.astype(np.float) / 255.
    elif im.dtype == 'uint16':
        im = im.astype(np.float) / 65535.
    else:
        raise NotImplementedError('unsupported source dtype: {0}'.format(im.dtype))

    assert np.min(im) >= 0 and np.max(im) <= 1

    if dtype in ['float', 'float16', 'float32', 'float64']:
        im = im.astype(dtype)
    elif dtype == 'uint8':
        im = (im * 255.).astype(dtype)
    elif dtype == 'uint16':
        im = (im * 65535.).astype(dtype)
    else:
        raise NotImplementedError('unsupported target dtype: {0}'.format(dtype))

    return im


def imwrite(path, obj):
    """Save Image.
    
    Args:
        path: path to save the image. Suffix support: png or jpg or gif
        image: array or list of array(list of image --> save as gif). Shape support: WxHx3 or WxHx1 or 3xWxH or 1xWxH
    """
    if not isinstance(obj, (collections.Sequence, collections.UserList)):
        obj = [obj]
    writer = imageio.get_writer(path)
    for im in obj:
        im = imretype(im, dtype='uint8').squeeze()
        if len(im.shape) == 3 and im.shape[0] == 3:
            im = np.transpose(im, (1, 2, 0))
        writer.append_data(im)
    writer.close()


def compute_view_and_pose_matrix(cam_position, lookat, up):
    cam_position = np.asarray(cam_position, dtype=np.float)
    lookat = np.asarray(lookat, dtype=np.float)
    up = np.asarray(up, dtype=np.float)
    up /= np.linalg.norm(up)
    f = lookat - cam_position
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    up = np.cross(s, f)
    view_matrix = np.eye(4)
    view_matrix[0, :3] = s
    view_matrix[1, :3] = up
    view_matrix[2, :3] = -f
    view_matrix[0, 3] = -np.dot(s, cam_position)
    view_matrix[1, 3] = -np.dot(up, cam_position)
    view_matrix[2, 3] = np.dot(f, cam_position)
    pose_matrix = np.linalg.inv(view_matrix)
    pose_matrix[:, 1:3] = -pose_matrix[:, 1:3]
    return view_matrix, pose_matrix


def html_visualize(web_path, data, ids, cols, others=[], title='visualization', clean=True, html_file_name='index', group_ids=None):
    """Visualization in html.
    
    Args:
        web_path: string; directory to save webpage. It will clear the old data!
        data: dict; 
            key: {id}_{col}. 
            value: figure or text
                - figure: ndarray --> .png or [ndarrays] --> .gif
                - text: string or [string]
        ids: [string]; name of each row
        cols: [string]; name of each column
        others: (optional) [dict]; other figures
            - name: string; name of the data, visualize using h2()
            - data: string or ndarray(image)
            - height: (optional) int; height of the image (default 256)
        title: (optional) string; title of the webpage (default 'visualization')
        clean: [bool] clean folder or not
        html_file_name: [str] html_file_name
        id_groups: list of (id_list, group_name)
    """
    mkdir(web_path, clean=clean)
    figure_path = os.path.join(web_path, 'figures')
    mkdir(figure_path, clean=clean)
    imwrite_ray = ray.remote(imwrite).options(num_cpus=0.1)
    obj_ids = list()
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            obj_ids.append(imwrite_ray.remote(os.path.join(figure_path, key + '.png'), value))
        elif isinstance(value, list) and isinstance(value[0], np.ndarray):
            obj_ids.append(imwrite_ray.remote(os.path.join(figure_path, key + '.gif'), value))
    ray.get(obj_ids)
    
    group_ids = group_ids if group_ids is not None else [('', ids)]

    with dominate.document(title=title) as web:
        dominate.tags.h1(title)
        for idx, other in enumerate(others):
            dominate.tags.h2(other['name'])
            if isinstance(other['data'], str):
                dominate.tags.p(other['data'])
            else:
                imwrite(os.path.join(figure_path, '_{}_{}.png'.format(idx, other['name'])), other['data'])
                dominate.tags.img(style='height:{}px'.format(other.get('height', 256)),
                    src=os.path.join('figures', '_{}_{}.png'.format(idx, other['name'])))
                    
        for group_name, ids in group_ids:
            if group_name != '':
                dominate.tags.h2(group_name)
            with dominate.tags.table(border=1, style='table-layout: fixed;'):
                with dominate.tags.tr():
                    with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', width='64px'):
                        dominate.tags.p('id')
                    for col in cols:
                        with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center'):
                            dominate.tags.p(col)
                for id in ids:
                    with dominate.tags.tr():
                        bgcolor = 'F1C073' if id.startswith('train') else 'C5F173'
                        with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', bgcolor=bgcolor):
                            for part in id.split('_'):
                                dominate.tags.p(part)
                        for col in cols:
                            with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='top'):
                                value = data[f'{id}_{col}']
                                if isinstance(value, str):
                                    dominate.tags.p(value)
                                elif isinstance(value, list) and isinstance(value[0], str):
                                    for v in value:
                                        dominate.tags.p(v)
                                elif isinstance(value, list) and isinstance(value[0], np.ndarray):
                                    dominate.tags.img(style='height:128px', src=os.path.join('figures', '{}_{}.gif'.format(id, col)))
                                elif isinstance(value, np.ndarray):
                                    dominate.tags.img(style='height:128px', src=os.path.join('figures', '{}_{}.png'.format(id, col)))
                                else:
                                    raise NotImplementedError()
    
    with open(os.path.join(web_path, f'{html_file_name}.html'), 'w') as fp:
        fp.write(web.render())


def meshwrite(filename, verts, colors, nocs=None, faces=None):
    """Save 3D mesh to a polygon .ply file.

    Args:
        filename: string; path to mesh file. (suffix should be .ply)
        verts: [N, 3]. Coordinates of each vertex
        colors: [N, 3]. RGB or each vertex. (type: uint8)
        faces: (optional) [M, 4]
    """
    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    if nocs is not None:
        ply_file.write("property float nocs_x\n")
        ply_file.write("property float nocs_y\n")
        ply_file.write("property float nocs_z\n")
    if faces is not None:
        ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        if nocs is not None:
            ply_file.write(
            "%f %f %f %d %d %d %f %f %f\n" %
            (verts[i, 0], verts[i, 1], verts[i, 2], 
            colors[i, 0], colors[i, 1], colors[i, 2],
            nocs[i, 0], nocs[i, 1], nocs[i, 2]))
        else:
            ply_file.write(
            "%f %f %f %d %d %d\n" %
            (verts[i, 0], verts[i, 1], verts[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))

    # Write face list
    if faces is not None:
        for i in range(faces.shape[0]):
            ply_file.write("4 %d %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2], faces[i, 3]))

    ply_file.close()



@jit(nopython=True, nogil=True)
def draw_points(img, image_size0, image_size1, sort_id, cam_pix, colors, large_pts_num=0):
    for id in sort_id:
        kernel_size = 2 if id < large_pts_num else 0
        x = int(cam_pix[id, 0])
        y = int(cam_pix[id, 1])
        img[max(0, x - kernel_size): min(image_size0, x + kernel_size + 1),
            max(0, y - kernel_size): min(image_size1, y + kernel_size + 1), :] = colors[id][:3]
    return img


def render_pts(pts, angle, blower=None):
    large_pts_num = len(pts)

    pts = np.stack([pts[:, 0], pts[:, 1], pts[:, 2]], axis=1)
    colors = list()
    for i in range(len(pts)):
        colors.append([i / len(pts), i / len(pts), 0.4, 1])
    colors = np.asarray(colors)

    # blower
    if blower is not None:
        blower_position, blower_orientation = blower
        blower_theta = -(blower_orientation[0] + 90 ) / 180 * np.pi
        blower_pts = list()
        d = 0.015
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    blower_pts.append([x * d + blower_position[0], y * d + blower_position[1], z * d + blower_position[2]])
        blower_pts = np.array(blower_pts)
        blower_color = np.array([[0.8, 0, 0, 1] for i in range(len(blower_pts))])
        pts = np.concatenate([pts, blower_pts], axis=0)
        colors = np.concatenate([colors, blower_color], axis=0)
        large_pts_num += len(blower_pts)

        wind_pts = list()
        N = 90
        for i in range(N):
            wind_pts.append([blower_position[0] - np.cos(blower_theta) * i * 0.01, blower_position[1] - np.sin(blower_theta) * i * 0.01, blower_position[2]])
        wind_pts = np.array(wind_pts).astype(float)
        wind_color = np.array([[0.5, 0.2, 0.2, 1] for i in range(len(wind_pts))])
        pts = np.concatenate([pts, wind_pts], axis=0)
        colors = np.concatenate([colors, wind_color], axis=0)
    
    # boundary
    bnd_pts = list()
    N = 100
    h = 0.25
    for i in range(N+1):
        k = - 0.48 + 0.96 / N * i
        bnd_pts.append([- 0.48, k, h])
        bnd_pts.append([0.48, k, h])
        bnd_pts.append([k, -0.48, h])
        bnd_pts.append([k, 0.48, h])
    bnd_pts = np.array(bnd_pts)
    bnd_color = np.array([[0.4, 0.4, 0.4, 1] for i in range(len(bnd_pts))])
    pts = np.concatenate([pts, bnd_pts], axis=0)
    colors = np.concatenate([colors, bnd_color], axis=0)

    image_size = [256, 256]
    length = 1.4
    cam_position = [length * np.cos(angle), length * np.sin(angle), 0.7]
    cam_lookat = [0, 0, 0.2]
    cam_up_direction = [0, 0, 1]
    cam_view_matrix, cam_pose_matrix = compute_view_and_pose_matrix(cam_position, cam_lookat, cam_up_direction)
    cam_intrinsics = np.array([[200, 0, float(image_size[1])/2],
                               [0, 200, float(image_size[0])/2],
                               [0, 0, 1]])
    cam_pix = project_pts_to_2d(pts, cam_view_matrix, cam_intrinsics)
    sort_id = np.argsort(-cam_pix[:, 2])

    img = np.ones([image_size[0], image_size[1], 3])
    img = draw_points(img, image_size[0], image_size[1], sort_id, cam_pix, colors, large_pts_num)
    
    return img


def rgb_to_hex(rgb):
    if rgb.dtype == np.uint8:
        pass
    elif rgb.dtype in (np.float16, np.float32, np.float64):
        print('Assuming Value in [0.0, 1.0]')
        rgb = (rgb * 255).astype(np.uint8)

    assert(rgb.dtype == np.uint8)
    hex = np.sum(rgb.astype(np.uint32) * np.array([1, 256, 256 ** 2])[::-1], axis=1)
    return hex


def apply_transformation(pts, transformation_matrix):
    # pts: (N, 3) or (3)
    return (transformation_matrix[:3, :3] @ pts.T + transformation_matrix[:3, 3]).T


def get_valid_idx(pts, bnd):
    # pts: [N, 3]
    # bnd: (3, 2)
    valid_idx = np.logical_and(
        np.logical_and(
            np.logical_and(pts[:, 0] >= bnd[0, 0], pts[:, 0] <= bnd[0, 1]),
            np.logical_and(pts[:, 1] >= bnd[1, 0], pts[:, 1] <= bnd[1, 1])
        ),
        np.logical_and(pts[:, 2] >= bnd[2, 0], pts[:, 2] <= bnd[2, 1])
    )
    return valid_idx
