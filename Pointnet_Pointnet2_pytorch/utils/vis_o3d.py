import cv2
import numpy as np
import open3d as o3d
import os
from utils import bbox3d2corners


COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
COLORS_IMG = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]]

LINES = [
        [0, 1],
        [1, 2], 
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [2, 6],
        [7, 3],
        [1, 5],
        [4, 0]
    ]


def npy2ply(npy):
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(npy[:, :3])
    density = npy[:, 3]
    colors = [[item, item, item] for item in density]
    ply.colors = o3d.utility.Vector3dVector(colors)
    return ply


def ply2npy(ply):
    return np.array(ply.points)


def bbox_obj(points, color=[1, 0, 0]):
    colors = [color for i in range(len(LINES))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(LINES),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def vis_core(plys):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    PAR = os.path.dirname(os.path.abspath(__file__))
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(os.path.join(PAR, 'viewpoint.json'))
    for ply in plys:
        vis.add_geometry(ply)
    ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()
    # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters(os.path.join(PAR, 'viewpoint.json'), param)
    vis.destroy_window()


def vis_pc(pc, bboxes=None, labels=None):
    '''
    pc: ply or np.ndarray (N, 4)
    bboxes: np.ndarray, (n, 7) or (n, 8, 3)
    labels: (n, )
    '''
    if isinstance(pc, np.ndarray):
        pc = npy2ply(pc)
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.05, origin=[0, 0, 0])

    if bboxes is None:
        vis_core([pc, mesh_frame])
        return
    
    if len(bboxes.shape) == 2:
        bboxes = bbox3d2corners(bboxes)
    
    vis_objs = [pc, mesh_frame]
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        if labels is None:
            color = [1, 1, 0]
        else:
            if labels[i] >= 0 and labels[i] < 3:
                color = COLORS[labels[i]]
            else:
                color = COLORS[-1]
        vis_objs.append(bbox_obj(bbox, color=color))
    vis_core(vis_objs)
