import numpy as np
import trimesh
import math
from render.mesh_utils import MeshViewer
from render.utils import colors
import imageio
import pyrender
from PIL import Image
import torch
from PIL import ImageDraw 

def c2rgba(c):
    if len(c) == 3:
        c.append(1)
    c = [c_i/255 for c_i in c[:3]]

    return c
def visualize_body(body_verts, body_face, save_path,
                       multi_angle=False, h=512, w=512, bg_color='white', show_frame=False):
    """[summary]

    Args:
        rec (torch.tensor): [description]
        inp (torch.tensor, optional): [description]. Defaults to None.
        multi_angle (bool, optional): Whether to use different angles. Defaults to False.

    Returns:
        np.array :   Shape of output (view_angles, seqlen, 3, im_width, im_height, )

    """
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    im_height = h
    im_width = w
    seqlen = len(body_verts)

    mesh_rec = body_verts
    # obj_mesh_rec = obj_verts
    
    minx, _, miny = mesh_rec.min(axis=(0, 1))
    maxx, _, maxy = mesh_rec.max(axis=(0, 1))
    minsxy = (minx, maxx, miny, maxy)
    height_offset = np.min(mesh_rec[:, :, 1])  # Min height

    mesh_rec = mesh_rec.copy()
    # obj_mesh_rec = obj_mesh_rec.copy()
    # mesh_rec[:, :, 1] -= height_offset
    # obj_mesh_rec[:, :, 1] -= height_offset
    mesh_rec[:, :, 0] -= (minx + maxx) / 2
    # obj_mesh_rec[:, :, 0] -= (minx + maxx) / 2
    mesh_rec[:, :, 2] -= (miny + maxy) / 2
    # obj_mesh_rec[:, :, 2] -= (miny + maxy) / 2

    mv = MeshViewer(width=im_width, height=im_height,
                    add_ground_plane=True, plane_mins=minsxy,
                    use_offscreen=True,
                    bg_color=bg_color)

    mv.render_wireframe = False

    if multi_angle:
        video = np.zeros([seqlen, 1 * im_width, 2 * im_height, 3])
    else:
        video = np.zeros([seqlen, im_width, im_height, 3])

    for i in range(seqlen):

        # obj_mesh_color = np.tile(c2rgba(colors['pink']), (obj_mesh_rec.shape[1], 1))

        # obj_m_rec = trimesh.Trimesh(vertices=obj_mesh_rec[i],
        #                             faces=obj_face,
        #                             vertex_colors=obj_mesh_color)

        mesh_color = np.tile(c2rgba(colors['yellow_pale']), (mesh_rec.shape[1], 1))

        m_rec = trimesh.Trimesh(vertices=mesh_rec[i],
                                faces=body_face,
                                vertex_colors=mesh_color)
        all_meshes = []

        all_meshes = all_meshes + [ m_rec]
        mv.set_meshes(all_meshes, group_name='static')
        video_i = mv.render()

        if multi_angle:
            video_views = [video_i]
            for _ in range(1):
                all_meshes = []
                Ry = trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0])
                obj_m_rec.apply_transform(Ry)
                m_rec.apply_transform(Ry)
                all_meshes = all_meshes + [ m_rec]
                mv.set_meshes(all_meshes, group_name='static')
                
                video_views.append(mv.render())
            # video_i = np.concatenate((np.concatenate((video_views[0], video_views[1]), axis=1),
            #                           np.concatenate((video_views[3], video_views[2]), axis=1)), axis=1)
            video_i = np.concatenate((video_views[0], video_views[1]), axis=1)
        video[i] = video_i

    video_writer = imageio.get_writer(save_path, fps=30)
    video = video.astype(np.uint8)
    for i in range(seqlen):
        frame = video[i]
        pil_image = Image.fromarray(frame)
        if show_frame:
            draw = ImageDraw.Draw(pil_image)
            text = f"{i}".zfill(4)
            draw.text((5, 5), text,fill='red')
        frame_with_text = np.array(pil_image).astype(np.uint8)
        video_writer.append_data(frame_with_text)
    video_writer.close()
    del mv

def visualize_body_obj(body_verts, body_face, obj_verts, obj_face, save_path,
                       multi_angle=False, h=512, w=512, bg_color='white', show_frame=False):
    """[summary]

    Args:
        rec (torch.tensor): [description]
        inp (torch.tensor, optional): [description]. Defaults to None.
        multi_angle (bool, optional): Whether to use different angles. Defaults to False.

    Returns:
        np.array :   Shape of output (view_angles, seqlen, 3, im_width, im_height, )

    """
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    im_height = h
    im_width = w
    seqlen = len(body_verts)

    mesh_rec = body_verts
    obj_mesh_rec = obj_verts
    
    minx, _, miny = mesh_rec.min(axis=(0, 1))
    maxx, _, maxy = mesh_rec.max(axis=(0, 1))
    minsxy = (minx, maxx, miny, maxy)
    height_offset = np.min(mesh_rec[:, :, 1])  # Min height

    mesh_rec = mesh_rec.copy()
    obj_mesh_rec = obj_mesh_rec.copy()
    mesh_rec[:, :, 1] -= height_offset
    obj_mesh_rec[:, :, 1] -= height_offset
    mesh_rec[:, :, 0] -= (minx + maxx) / 2
    obj_mesh_rec[:, :, 0] -= (minx + maxx) / 2
    mesh_rec[:, :, 2] -= (miny + maxy) / 2
    obj_mesh_rec[:, :, 2] -= (miny + maxy) / 2

    mv = MeshViewer(width=im_width, height=im_height,
                    add_ground_plane=True, plane_mins=minsxy,
                    use_offscreen=True,
                    bg_color=bg_color)

    mv.render_wireframe = False

    if multi_angle:
        video = np.zeros([seqlen, 1 * im_width, 2 * im_height, 3])
    else:
        video = np.zeros([seqlen, im_width, im_height, 3])

    for i in range(seqlen):

        obj_mesh_color = np.tile(c2rgba(colors['pink']), (obj_mesh_rec.shape[1], 1))

        obj_m_rec = trimesh.Trimesh(vertices=obj_mesh_rec[i],
                                    faces=obj_face,
                                    vertex_colors=obj_mesh_color)

        mesh_color = np.tile(c2rgba(colors['yellow_pale']), (mesh_rec.shape[1], 1))

        m_rec = trimesh.Trimesh(vertices=mesh_rec[i],
                                faces=body_face,
                                vertex_colors=mesh_color)
        all_meshes = []

        all_meshes = all_meshes + [obj_m_rec, m_rec]
        mv.set_meshes(all_meshes, group_name='static')
        video_i = mv.render()

        if multi_angle:
            video_views = [video_i]
            for _ in range(1):
                all_meshes = []
                Ry = trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0])
                obj_m_rec.apply_transform(Ry)
                m_rec.apply_transform(Ry)
                all_meshes = all_meshes + [obj_m_rec, m_rec]
                mv.set_meshes(all_meshes, group_name='static')
                
                video_views.append(mv.render())
            # video_i = np.concatenate((np.concatenate((video_views[0], video_views[1]), axis=1),
            #                           np.concatenate((video_views[3], video_views[2]), axis=1)), axis=1)
            video_i = np.concatenate((video_views[0], video_views[1]), axis=1)
        video[i] = video_i

    video_writer = imageio.get_writer(save_path, fps=30)
    video = video.astype(np.uint8)
    for i in range(seqlen):
        frame = video[i]
        pil_image = Image.fromarray(frame)
        if show_frame:
            draw = ImageDraw.Draw(pil_image)
            text = f"{i}".zfill(4)
            draw.text((5, 5), text,fill='red')
        frame_with_text = np.array(pil_image).astype(np.uint8)
        video_writer.append_data(frame_with_text)
    video_writer.close()
    del mv
    


def visualize_body_obj(body_verts, body_face, obj_verts, obj_face, save_path,
                       multi_angle=False, h=512, w=512, bg_color='white', show_frame=False):
    """[summary]

    Args:
        rec (torch.tensor): [description]
        inp (torch.tensor, optional): [description]. Defaults to None.
        multi_angle (bool, optional): Whether to use different angles. Defaults to False.

    Returns:
        np.array :   Shape of output (view_angles, seqlen, 3, im_width, im_height, )

    """
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    im_height = h
    im_width = w
    seqlen = len(body_verts)

    mesh_rec = body_verts
    obj_mesh_rec = obj_verts
    
    minx, _, miny = mesh_rec.min(axis=(0, 1))
    maxx, _, maxy = mesh_rec.max(axis=(0, 1))
    minsxy = (minx, maxx, miny, maxy)
    height_offset = np.min(mesh_rec[:, :, 1])  # Min height

    mesh_rec = mesh_rec.copy()
    obj_mesh_rec = obj_mesh_rec.copy()
    mesh_rec[:, :, 1] -= height_offset
    obj_mesh_rec[:, :, 1] -= height_offset
    mesh_rec[:, :, 0] -= (minx + maxx) / 2
    obj_mesh_rec[:, :, 0] -= (minx + maxx) / 2
    mesh_rec[:, :, 2] -= (miny + maxy) / 2
    obj_mesh_rec[:, :, 2] -= (miny + maxy) / 2

    mv = MeshViewer(width=im_width, height=im_height,
                    add_ground_plane=True, plane_mins=minsxy,
                    use_offscreen=True,
                    bg_color=bg_color)

    mv.render_wireframe = False

    if multi_angle:
        video = np.zeros([seqlen, 1 * im_width, 2 * im_height, 3])
    else:
        video = np.zeros([seqlen, im_width, im_height, 3])

    for i in range(seqlen):

        obj_mesh_color = np.tile(c2rgba(colors['pink']), (obj_mesh_rec.shape[1], 1))

        obj_m_rec = trimesh.Trimesh(vertices=obj_mesh_rec[i],
                                    faces=obj_face,
                                    vertex_colors=obj_mesh_color)

        mesh_color = np.tile(c2rgba(colors['yellow_pale']), (mesh_rec.shape[1], 1))

        m_rec = trimesh.Trimesh(vertices=mesh_rec[i],
                                faces=body_face,
                                vertex_colors=mesh_color)
        all_meshes = []

        all_meshes = all_meshes + [obj_m_rec, m_rec]
        mv.set_meshes(all_meshes, group_name='static')
        video_i = mv.render()

        if multi_angle:
            video_views = [video_i]
            for _ in range(1):
                all_meshes = []
                Ry = trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0])
                obj_m_rec.apply_transform(Ry)
                m_rec.apply_transform(Ry)
                all_meshes = all_meshes + [obj_m_rec, m_rec]
                mv.set_meshes(all_meshes, group_name='static')
                
                video_views.append(mv.render())
            # video_i = np.concatenate((np.concatenate((video_views[0], video_views[1]), axis=1),
            #                           np.concatenate((video_views[3], video_views[2]), axis=1)), axis=1)
            video_i = np.concatenate((video_views[0], video_views[1]), axis=1)
        video[i] = video_i

    video_writer = imageio.get_writer(save_path, fps=30)
    video = video.astype(np.uint8)
    for i in range(seqlen):
        frame = video[i]
        pil_image = Image.fromarray(frame)
        if show_frame:
            draw = ImageDraw.Draw(pil_image)
            text = f"{i}".zfill(4)
            draw.text((5, 5), text,fill='red')
        frame_with_text = np.array(pil_image).astype(np.uint8)
        video_writer.append_data(frame_with_text)
    video_writer.close()
    del mv



def points_to_spheres(points, radius=0.01):
    spheres = []
    for p in points:
        sphere = trimesh.creation.icosphere(radius=radius)
        sphere.apply_translation(p)
        spheres.append(sphere)
    return trimesh.util.concatenate(spheres)
def visualize_body_objs(body_verts, body_face, obj_verts_all, obj_face_all, save_path,
                       multi_angle=False, h=768, w=768, bg_color='white', show_frame=True,
                       highlight_frame=None, highlight_vertex=None):
    """Visualize body and object with optional highlight for a specific frame and vertex."""
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    im_height = h
    im_width = w
    seqlen = len(body_verts)

    # Convert tensors to numpy if needed
    if torch.is_tensor(body_verts):
        mesh_rec = body_verts.cpu().numpy()
    else:
        mesh_rec = body_verts
        
    # if torch.is_tensor(obj_verts):
    #     obj_mesh_rec = obj_verts.cpu().numpy()
    # else:
    #     obj_mesh_rec = obj_verts

    # Compute bounding box for scaling the marker size
    minx, _, miny = mesh_rec.min(axis=(0, 1))
    maxx, _, maxy = mesh_rec.max(axis=(0, 1))
    bbox_size = max(maxx - minx, maxy - miny)
    marker_radius = bbox_size * 0.01  # Scale marker size based on mesh size

    minsxy = (minx, maxx, miny, maxy)
    height_offset = np.min(mesh_rec[:, :, 1])  # Min height

    mesh_rec = mesh_rec.copy()
    

    mesh_rec[:, :, 0] -= (minx + maxx) / 2
    
    mesh_rec[:, :, 2] -= (miny + maxy) / 2
    
    obj_mesh_recs = []
    for obj_mesh_rec in obj_verts_all:
        obj_mesh_rec = obj_mesh_rec.copy()
        obj_mesh_rec[:, :, 0] -= (minx + maxx) / 2
        obj_mesh_rec[:, :, 2] -= (miny + maxy) / 2
        obj_mesh_recs.append(obj_mesh_rec)

    mv = MeshViewer(width=im_width, height=im_height,
                    add_ground_plane=True, plane_mins=minsxy,
                    use_offscreen=True,
                    bg_color=bg_color)

    mv.render_wireframe = False

    if multi_angle:
        video = np.zeros([seqlen, 1 * im_width, 2 * im_height, 3])
    else:
        video = np.zeros([seqlen, im_width, im_height, 3])

    for i in range(seqlen):
        # Set object mesh color (pink)
        obj_m_rec =[]
        for k,obj_mesh_rec in enumerate(obj_mesh_recs):
            rgba_color = np.concatenate([c2rgba(colors['pink'])[:3], [1]])  # RGB + Alpha
            obj_mesh_color = np.tile(rgba_color, (obj_mesh_rec.shape[1], 1))
            obj_m_rec.append(trimesh.Trimesh(vertices=obj_mesh_rec[i],
                                        faces=obj_face_all[k],
                                        vertex_colors=obj_mesh_color))

        # Set body mesh color (yellow pale)
        rgba_color_2 = np.concatenate([c2rgba(colors['yellow_pale'])[:3], [1]])  # RGB + Alpha
        mesh_color = np.tile(rgba_color_2, (mesh_rec.shape[1], 1))
        m_rec = trimesh.Trimesh(vertices=mesh_rec[i],
                                faces=body_face,
                                vertex_colors=mesh_color)

        # Initialize mesh list
        all_meshes = obj_m_rec+ [m_rec]

        # Add the highlight sphere if conditions are met
        if highlight_frame is not None and highlight_vertex is not None:
            if i >= highlight_frame - 15 and i <= highlight_frame+15:
                print(f"Highlighting frame {highlight_frame}, vertex {highlight_vertex}")
                # Create a small red sphere at the vertex position
                vertex_pos = mesh_rec[i, highlight_vertex]
                marker_sphere = trimesh.creation.icosphere(subdivisions=3, radius=marker_radius)
                marker_sphere.apply_translation(vertex_pos)
                marker_sphere.visual.vertex_colors = [1.0, 0.0, 0.0, 1.0]  # Red sphere

                # Append the marker sphere to the mesh list
                all_meshes.append(marker_sphere)

        mv.set_meshes(all_meshes, group_name='static')
        video_i = mv.render()

        if multi_angle:
            video_views = [video_i]
            for _ in range(1):
                Ry = trimesh.transformations.rotation_matrix(np.radians(270), [0, 1, 0])
                
                all_meshes =[]
                for obj_m in obj_m_rec:
                    obj_m.apply_transform(Ry)
                    all_meshes.append(obj_m)
                m_rec.apply_transform(Ry)
                all_meshes.append(m_rec)
                
                if highlight_frame is not None and highlight_vertex is not None:
                    if i >= highlight_frame - 15 and i <= highlight_frame+15:
                        vertex_pos = mesh_rec[i, highlight_vertex]
                        marker_sphere = trimesh.creation.icosphere(subdivisions=3, radius=marker_radius)
                        marker_sphere.apply_translation(vertex_pos)
                        marker_sphere.apply_transform(Ry)
                        marker_sphere.visual.vertex_colors = [1.0, 0.0, 0.0, 1.0]
                        all_meshes.append(marker_sphere)
                mv.set_meshes(all_meshes, group_name='static')
                video_views.append(mv.render())
            video_i = np.concatenate((video_views[0], video_views[1]), axis=1)
        video[i] = video_i

    video_writer = imageio.get_writer(save_path, fps=30)
    video = video.astype(np.uint8)
    for i in range(seqlen):
        frame = video[i]
        pil_image = Image.fromarray(frame)
        if show_frame:
            draw = ImageDraw.Draw(pil_image)
            text = f"{i}".zfill(4)
            draw.text((5, 5), text, fill='red')
        frame_with_text = np.array(pil_image).astype(np.uint8)
        video_writer.append_data(frame_with_text)
    video_writer.close()
    del mv

def visualize_mesh(obj_verts, obj_face, save_path,
                       multi_angle=False, h=512, w=512, bg_color='white', show_frame=False):
    """[summary]

    Args:
        rec (torch.tensor): [description]
        inp (torch.tensor, optional): [description]. Defaults to None.
        multi_angle (bool, optional): Whether to use different angles. Defaults to False.

    Returns:
        np.array :   Shape of output (view_angles, seqlen, 3, im_width, im_height, )

    """
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    im_height = h
    im_width = w
    seqlen = len(obj_verts)

    # mesh_rec = m_pcd
    obj_mesh_rec = obj_verts
    
    minx, _, miny = obj_mesh_rec.min(axis=(0, 1))
    maxx, _, maxy = obj_mesh_rec.max(axis=(0, 1))
    minsxy = (minx, maxx, miny, maxy)
    height_offset = np.min(obj_mesh_rec[:, :, 1])  # Min height

    # mesh_rec = mesh_rec.copy()
    obj_mesh_rec = obj_mesh_rec.copy()
    # mesh_rec[:, :, 1] -= height_offset
    # obj_mesh_rec[:, :, 1] -= height_offset
    # mesh_rec[:, :, 0] -= (minx + maxx) / 2
    obj_mesh_rec[:, :, 0] -= (minx + maxx) / 2
    # mesh_rec[:, :, 2] -= (miny + maxy) / 2
    obj_mesh_rec[:, :, 2] -= (miny + maxy) / 2

    mv = MeshViewer(width=im_width, height=im_height,
                    add_ground_plane=True, plane_mins=minsxy,
                    use_offscreen=True,
                    bg_color=bg_color)

    mv.render_wireframe = False

    if multi_angle:
        video = np.zeros([seqlen, 1 * im_width, 2 * im_height, 3])
    else:
        video = np.zeros([seqlen, im_width, im_height, 3])

    for i in range(seqlen):

        obj_mesh_color = np.tile(c2rgba(colors['pink']), (obj_mesh_rec.shape[1], 1))

        obj_m_rec = trimesh.Trimesh(vertices=obj_mesh_rec[i],
                                    faces=obj_face,
                                    vertex_colors=obj_mesh_color)

        # mesh_color = np.tile(c2rgba(colors['yellow_pale']), (mesh_rec.shape[1], 1))

        # m_rec = trimesh.points.PointCloud(mesh_rec[i])
        # m_rec = points_to_spheres(mesh_rec[i])

        all_meshes = []

        all_meshes = all_meshes + [obj_m_rec]
        mv.set_meshes(all_meshes, group_name='static')
        video_i = mv.render()

        if multi_angle:
            video_views = [video_i]
            for _ in range(1):
                all_meshes = []
                Ry = trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0])
                obj_m_rec.apply_transform(Ry)
                # m_rec.apply_transform(Ry)
                all_meshes = all_meshes + [obj_m_rec]
                mv.set_meshes(all_meshes, group_name='static')
                
                video_views.append(mv.render())
            # video_i = np.concatenate((np.concatenate((video_views[0], video_views[1]), axis=1),
            #                           np.concatenate((video_views[3], video_views[2]), axis=1)), axis=1)
            video_i = np.concatenate((video_views[0], video_views[1]), axis=1)
        video[i] = video_i

    video_writer = imageio.get_writer(save_path, fps=30)
    video = video.astype(np.uint8)
    for i in range(seqlen):
        frame = video[i]
        pil_image = Image.fromarray(frame)
        if show_frame:
            draw = ImageDraw.Draw(pil_image)
            text = f"{i}".zfill(4)
            draw.text((5, 5), text,fill='red')
        frame_with_text = np.array(pil_image).astype(np.uint8)
        video_writer.append_data(frame_with_text)
    video_writer.close()
    del mv


def visualize_points_obj(m_pcd, obj_verts, obj_face, save_path,
                       multi_angle=False, h=256, w=256, bg_color='white', show_frame=False):
    """[summary]

    Args:
        rec (torch.tensor): [description]
        inp (torch.tensor, optional): [description]. Defaults to None.
        multi_angle (bool, optional): Whether to use different angles. Defaults to False.

    Returns:
        np.array :   Shape of output (view_angles, seqlen, 3, im_width, im_height, )

    """
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    im_height = h
    im_width = w
    seqlen = len(m_pcd)

    mesh_rec = m_pcd
    obj_mesh_rec = obj_verts
    
    minx, _, miny = mesh_rec.min(axis=(0, 1))
    maxx, _, maxy = mesh_rec.max(axis=(0, 1))
    minsxy = (minx, maxx, miny, maxy)
    height_offset = np.min(mesh_rec[:, :, 1])  # Min height

    mesh_rec = mesh_rec.copy()
    obj_mesh_rec = obj_mesh_rec.copy()
    # mesh_rec[:, :, 1] -= height_offset
    # obj_mesh_rec[:, :, 1] -= height_offset
    mesh_rec[:, :, 0] -= (minx + maxx) / 2
    obj_mesh_rec[:, :, 0] -= (minx + maxx) / 2
    mesh_rec[:, :, 2] -= (miny + maxy) / 2
    obj_mesh_rec[:, :, 2] -= (miny + maxy) / 2

    mv = MeshViewer(width=im_width, height=im_height,
                    add_ground_plane=True, plane_mins=minsxy,
                    use_offscreen=True,
                    bg_color=bg_color)

    mv.render_wireframe = False

    if multi_angle:
        video = np.zeros([seqlen, 1 * im_width, 2 * im_height, 3])
    else:
        video = np.zeros([seqlen, im_width, im_height, 3])

    for i in range(seqlen):

        obj_mesh_color = np.tile(c2rgba(colors['pink']), (obj_mesh_rec.shape[1], 1))

        obj_m_rec = trimesh.Trimesh(vertices=obj_mesh_rec[i],
                                    faces=obj_face,
                                    vertex_colors=obj_mesh_color)

        mesh_color = np.tile(c2rgba(colors['yellow_pale']), (mesh_rec.shape[1], 1))

        # m_rec = trimesh.points.PointCloud(mesh_rec[i])
        m_rec = points_to_spheres(mesh_rec[i])

        all_meshes = []

        all_meshes = all_meshes + [obj_m_rec, m_rec]
        mv.set_meshes(all_meshes, group_name='static')
        video_i = mv.render()

        if multi_angle:
            video_views = [video_i]
            for _ in range(1):
                all_meshes = []
                Ry = trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0])
                obj_m_rec.apply_transform(Ry)
                m_rec.apply_transform(Ry)
                all_meshes = all_meshes + [obj_m_rec, m_rec]
                mv.set_meshes(all_meshes, group_name='static')
                
                video_views.append(mv.render())
            # video_i = np.concatenate((np.concatenate((video_views[0], video_views[1]), axis=1),
            #                           np.concatenate((video_views[3], video_views[2]), axis=1)), axis=1)
            video_i = np.concatenate((video_views[0], video_views[1]), axis=1)
        video[i] = video_i

    video_writer = imageio.get_writer(save_path, fps=30)
    video = video.astype(np.uint8)
    for i in range(seqlen):
        frame = video[i]
        pil_image = Image.fromarray(frame)
        if show_frame:
            draw = ImageDraw.Draw(pil_image)
            text = f"{i}".zfill(4)
            draw.text((5, 5), text,fill='red')
        frame_with_text = np.array(pil_image).astype(np.uint8)
        video_writer.append_data(frame_with_text)
    video_writer.close()
    del mv
    
def visualize_points(m_pcd, save_path,
                       multi_angle=False, h=256, w=256, bg_color='white', show_frame=False):
    """[summary]

    Args:
        rec (torch.tensor): [description]
        inp (torch.tensor, optional): [description]. Defaults to None.
        multi_angle (bool, optional): Whether to use different angles. Defaults to False.

    Returns:
        np.array :   Shape of output (view_angles, seqlen, 3, im_width, im_height, )

    """
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    im_height = h
    im_width = w
    seqlen = len(m_pcd)

    mesh_rec = m_pcd
    # obj_mesh_rec = obj_verts
    
    minx, _, miny = mesh_rec.min(axis=(0, 1))
    maxx, _, maxy = mesh_rec.max(axis=(0, 1))
    minsxy = (minx, maxx, miny, maxy)
    height_offset = np.min(mesh_rec[:, :, 1])  # Min height

    mesh_rec = mesh_rec.copy()
    # obj_mesh_rec = obj_mesh_rec.copy()
    # mesh_rec[:, :, 1] -= height_offset
    # obj_mesh_rec[:, :, 1] -= height_offset
    mesh_rec[:, :, 0] -= (minx + maxx) / 2
    # obj_mesh_rec[:, :, 0] -= (minx + maxx) / 2
    mesh_rec[:, :, 2] -= (miny + maxy) / 2
    # obj_mesh_rec[:, :, 2] -= (miny + maxy) / 2

    mv = MeshViewer(width=im_width, height=im_height,
                    add_ground_plane=True, plane_mins=minsxy,
                    use_offscreen=True,
                    bg_color=bg_color)

    mv.render_wireframe = False

    if multi_angle:
        video = np.zeros([seqlen, 1 * im_width, 2 * im_height, 3])
    else:
        video = np.zeros([seqlen, im_width, im_height, 3])

    for i in range(seqlen):

        # obj_mesh_color = np.tile(c2rgba(colors['pink']), (obj_mesh_rec.shape[1], 1))

        # obj_m_rec = trimesh.Trimesh(vertices=obj_mesh_rec[i],
        #                             faces=obj_face,
        #                             vertex_colors=obj_mesh_color)

        mesh_color = np.tile(c2rgba(colors['yellow_pale']), (mesh_rec.shape[1], 1))

        # m_rec = trimesh.points.PointCloud(mesh_rec[i])
        m_rec = points_to_spheres(mesh_rec[i])

        all_meshes = []

        all_meshes = all_meshes + [ m_rec]
        mv.set_meshes(all_meshes, group_name='static')
        video_i = mv.render()

        if multi_angle:
            video_views = [video_i]
            for _ in range(1):
                all_meshes = []
                Ry = trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0])
                # obj_m_rec.apply_transform(Ry)
                m_rec.apply_transform(Ry)
                all_meshes = all_meshes + [ m_rec]
                mv.set_meshes(all_meshes, group_name='static')
                
                video_views.append(mv.render())
            # video_i = np.concatenate((np.concatenate((video_views[0], video_views[1]), axis=1),
            #                           np.concatenate((video_views[3], video_views[2]), axis=1)), axis=1)
            video_i = np.concatenate((video_views[0], video_views[1]), axis=1)
        video[i] = video_i

    video_writer = imageio.get_writer(save_path, fps=30)
    video = video.astype(np.uint8)
    for i in range(seqlen):
        frame = video[i]
        pil_image = Image.fromarray(frame)
        if show_frame:
            draw = ImageDraw.Draw(pil_image)
            text = f"{i}".zfill(4)
            draw.text((5, 5), text,fill='red')
        frame_with_text = np.array(pil_image).astype(np.uint8)
        video_writer.append_data(frame_with_text)
    video_writer.close()
    del mv

def create_arrow(origin, vector, scale=10, color=[1.0, 0.0, 0.0, 1.0]):
    # 如果vector长度为0，避免除零
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        norm = 1e-6
    direction = vector / norm
    # 这里定义箭头的总长度为向量长度乘以缩放因子
    length = norm
    # 使用 trimesh 创建箭头网格（注意：确保你使用的 trimesh 版本支持 creation.arrow）
    arrow = create_arrow_1(origin=origin, direction=direction, length=length,scale=scale,
                                   shaft_radius=0.005, head_radius=0.01, head_length=0.02)
    # 设置箭头颜色（RGBA, 0-255）
    arrow.visual.vertex_colors = np.tile((np.array(color)*255).astype(np.uint8), (arrow.vertices.shape[0], 1))
    # np.tile((np.array(color)*255).astype(np.uint8), (arrow.vertices.shape[0], 1))
    return arrow



def visualize_body_obj_arrow(body_verts, body_face, obj_verts, obj_face, linear_vel,angular_vel, save_path,
                       multi_angle=False, h=512, w=512, bg_color='white', show_frame=False):
    """[summary]

    Args:
        rec (torch.tensor): [description]
        inp (torch.tensor, optional): [description]. Defaults to None.
        multi_angle (bool, optional): Whether to use different angles. Defaults to False.

    Returns:
        np.array :   Shape of output (view_angles, seqlen, 3, im_width, im_height, )

    """
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    im_height = h
    im_width = w
    seqlen = len(body_verts)

    mesh_rec = body_verts
    obj_mesh_rec = obj_verts
    
    minx, _, miny = mesh_rec.min(axis=(0, 1))
    maxx, _, maxy = mesh_rec.max(axis=(0, 1))
    minsxy = (minx, maxx, miny, maxy)
    height_offset = np.min(mesh_rec[:, :, 1])  # Min height

    mesh_rec = mesh_rec.copy()
    obj_mesh_rec = obj_mesh_rec.copy()
    # mesh_rec[:, :, 1] -= height_offset
    # obj_mesh_rec[:, :, 1] -= height_offset
    mesh_rec[:, :, 0] -= (minx + maxx) / 2
    obj_mesh_rec[:, :, 0] -= (minx + maxx) / 2
    mesh_rec[:, :, 2] -= (miny + maxy) / 2
    obj_mesh_rec[:, :, 2] -= (miny + maxy) / 2

    mv = MeshViewer(width=im_width, height=im_height,
                    add_ground_plane=True, plane_mins=minsxy,
                    use_offscreen=True,
                    bg_color=bg_color)

    mv.render_wireframe = False

    if multi_angle:
        video = np.zeros([seqlen, 1 * im_width, 2 * im_height, 3])
    else:
        video = np.zeros([seqlen, im_width, im_height, 3])

    for i in range(seqlen):

        obj_mesh_color = np.tile(c2rgba(colors['pink']), (obj_mesh_rec.shape[1], 1))

        obj_m_rec = trimesh.Trimesh(vertices=obj_mesh_rec[i],
                                    faces=obj_face,
                                    vertex_colors=obj_mesh_color)

        mesh_color = np.tile(c2rgba(colors['yellow_pale']), (mesh_rec.shape[1], 1))

        m_rec = trimesh.Trimesh(vertices=mesh_rec[i],
                                faces=body_face,
                                vertex_colors=mesh_color)
        all_meshes = []
        
        ## insert arrow
        
        lin_speed = np.linalg.norm(linear_vel[i])
        ang_speed = np.linalg.norm(angular_vel[i])
        # 设定最大速度（用于归一化）
        max_lin_speed = 10.0  # 根据实际情况调整
        max_ang_speed = 1.0   # 根据实际情况调整

        lin_color = speed_to_color_lin(lin_speed*30, max_lin_speed)
        ang_color = speed_to_color_ang(ang_speed*30, max_ang_speed)
        
        # 如果需要让箭头不被物体遮挡，可以设置起点偏移（例如在中心上方）
        # center = obj_m_rec.vertices.mean(axis=0)
        # offset = np.array([0, 0.1, 0])  # 根据实际情况调整
        # origin_arrow = center + offset
        
        bbox_min = obj_m_rec.vertices.min(axis=0)
        bbox_max = obj_m_rec.vertices.max(axis=0)
        center = obj_m_rec.vertices.mean(axis=0)
        height = bbox_max[1] - bbox_min[1]
        width  = bbox_max[0] - bbox_min[0]

        # 自动计算偏移：顶部偏移和右侧偏移
        margin_y = 0.05 * height  # 顶部偏移量，比如物体高度的10%
        margin_x = 0.05 * width   # 右侧偏移量，比如物体宽度的10%

        # 为线速度箭头设置起点：物体中心向上偏移
        origin_lin = center + np.array([0, (bbox_max[1] - center[1]) + margin_y, 0])
        # 为角速度箭头设置起点：物体中心向右偏移
        origin_ang = center + np.array([(bbox_max[0] - center[0]) + margin_x, 0, 0])
        
        # 生成箭头时，将颜色归一化到 [0,1]
        lin_arrow = create_arrow(origin=origin_lin, vector=linear_vel[i], scale=10, 
                                color=[c/255.0 for c in lin_color])
        ang_arrow = create_arrow(origin=origin_ang, vector=angular_vel[i], scale=10, 
                                color=[c/255.0 for c in ang_color])
        
        ## end
        

        all_meshes = all_meshes + [obj_m_rec, m_rec,lin_arrow,ang_arrow]
        mv.set_meshes(all_meshes, group_name='static')
        video_i = mv.render()

        if multi_angle:
            video_views = [video_i]
            for _ in range(1):
                all_meshes = []
                Ry = trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0])
                obj_m_rec.apply_transform(Ry)
                m_rec.apply_transform(Ry)
                all_meshes = all_meshes + [obj_m_rec, m_rec]
                mv.set_meshes(all_meshes, group_name='static')
                
                video_views.append(mv.render())
            # video_i = np.concatenate((np.concatenate((video_views[0], video_views[1]), axis=1),
            #                           np.concatenate((video_views[3], video_views[2]), axis=1)), axis=1)
            video_i = np.concatenate((video_views[0], video_views[1]), axis=1)
        video[i] = video_i

    video_writer = imageio.get_writer(save_path, fps=30)
    video = video.astype(np.uint8)
    for i in range(seqlen):
        frame = video[i]
        pil_image = Image.fromarray(frame)
        if show_frame:
            draw = ImageDraw.Draw(pil_image)
            text = f"{i}".zfill(4)
            draw.text((5, 5), text,fill='red')
        frame_with_text = np.array(pil_image).astype(np.uint8)
        video_writer.append_data(frame_with_text)
    video_writer.close()
    del mv

def speed_to_color_lin(speed, max_speed):
    """
    将线速度速度值映射为红色系RGBA颜色。
    速度越大，红色越深。
    """
    norm_speed = np.clip(speed / max_speed, 0, 1)
    # 红色分量从0.5到1.0变化，绿色和蓝色固定为0
    r = 0.5 + 0.5 * norm_speed
    g = 0.0
    b = 0.0
    a = 1.0
    return (int(r * 255), int(g * 255), int(b * 255), int(a * 255))

def speed_to_color_ang(speed, max_speed):
    """
    将角速度速度值映射为蓝色系RGBA颜色。
    速度越大，蓝色越深。
    """
    norm_speed = np.clip(speed / max_speed, 0, 1)
    # 蓝色分量从0.5到1.0变化，红色和绿色固定为0
    r = 0.0
    g = 0.0
    b = 0.5 + 0.5 * norm_speed
    a = 1.0
    return (int(r * 255), int(g * 255), int(b * 255), int(a * 255))

def create_arrow_1(origin, direction, length,scale, shaft_radius=0.005, head_radius=0.01, head_length=0.02):
    """
    根据起点和方向创建一个箭头网格
    :param origin: 箭头起点 [x, y, z]
    :param direction: 箭头方向向量
    :param length: 箭头总长度
    :param shaft_radius: 箭杆半径
    :param head_radius: 箭头半径
    :param head_length: 箭头头部的长度
    :return: 一个trimesh.Trimesh对象
    """
    # 箭杆长度为总长度减去箭头头部长度
    shaft_length = length*scale
    if shaft_length < 0:
        raise ValueError("Length must be greater than head_length.")
    
    # 创建箭杆：沿z轴方向的圆柱体
    shaft = trimesh.creation.cylinder(radius=shaft_radius*scale, height=shaft_length)
    # 将箭杆平移，使得底部在原点
    shaft.apply_translation([0, 0, shaft_length / 2])
    
    # 创建箭头头部：沿z轴方向的圆锥体
    head = trimesh.creation.cone(radius=head_radius*scale, height=head_length*scale)
    # 将箭头平移，使得底部与箭杆连接
    head.apply_translation([0, 0, shaft_length + head_length*scale / 2])
    
    # 合并箭杆和箭头
    arrow = trimesh.util.concatenate([shaft, head])
    
    # 默认生成的箭头沿z轴方向，我们需要将它旋转到与 direction 对齐
    # 计算方向向量的单位向量
    direction_norm = direction / np.linalg.norm(direction)
    # z轴向量
    z_axis = np.array([0, 0, 1])
    # 计算旋转矩阵：将 z_axis 旋转到 direction_norm
    transform, _ = trimesh.geometry.align_vectors(z_axis, direction_norm, return_angle=True)
    arrow.apply_transform(transform)
    
    # 平移箭头到指定的起点
    arrow.apply_translation(origin)
    
    return arrow