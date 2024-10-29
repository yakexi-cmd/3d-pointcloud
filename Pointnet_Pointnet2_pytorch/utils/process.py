import copy
import numba
import numpy as np
import random
import torch
import pdb
from ops.iou3d_module import boxes_overlap_bev, boxes_iou_bev


def setup_seed(seed=0, deterministic = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def bbox3d2corners(bboxes):
    '''
    bboxes: shape=(n, 7)
    return: shape=(n, 8, 3)
           ^ z   x            6 ------ 5
           |   /             / |     / |
           |  /             2 -|---- 1 |   
    y      | /              |  |     | | 
    <------|o               | 7 -----| 4
                            |/   o   |/    
                            3 ------ 0 
    x: front, y: left, z: top
    '''
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bboxes_corners = np.array([[-0.5, -0.5, 0], [-0.5, -0.5, 1.0], [-0.5, 0.5, 1.0], [-0.5, 0.5, 0.0],
                               [0.5, -0.5, 0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [0.5, 0.5, 0.0]], 
                               dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :] # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)

    # 2. rotate around z axis
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, -angle
    rot_mat = np.array([[rot_cos, rot_sin, np.zeros_like(rot_cos)],
                        [-rot_sin, rot_cos, np.zeros_like(rot_cos)],
                        [np.zeros_like(rot_cos), np.zeros_like(rot_cos), np.ones_like(rot_cos)]], 
                        dtype=np.float32) # (3, 3, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0)) # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat # (n, 8, 3)

    # 3. translate to centers
    bboxes_corners += centers[:, None, :]
    return bboxes_corners


def bbox3d2corners_camera(bboxes):
    '''
    bboxes: shape=(n, 7)
    return: shape=(n, 8, 3)
        z (front)            6 ------ 5
        /                  / |     / |
       /                  2 -|---- 1 |   
      /                   |  |     | | 
    |o ------> x(right)   | 7 -----| 4
    |                     |/   o   |/    
    |                     3 ------ 0 
    |
    v y(down)                   
    '''
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bboxes_corners = np.array([[0.5, 0.0, -0.5], [0.5, -1.0, -0.5], [-0.5, -1.0, -0.5], [-0.5, 0.0, -0.5],
                               [0.5, 0.0, 0.5], [0.5, -1.0, 0.5], [-0.5, -1.0, 0.5], [-0.5, 0.0, 0.5]], 
                               dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :] # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)

    # 2. rotate around y axis
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, angle
    rot_mat = np.array([[rot_cos, np.zeros_like(rot_cos), rot_sin],
                        [np.zeros_like(rot_cos), np.ones_like(rot_cos), np.zeros_like(rot_cos)],
                        [-rot_sin, np.zeros_like(rot_cos), rot_cos]], 
                        dtype=np.float32) # (3, 3, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0)) # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat # (n, 8, 3)

    # 3. translate to centers
    bboxes_corners += centers[:, None, :]
    return bboxes_corners


def group_rectangle_vertexs(bboxes_corners):
    '''
    bboxes_corners: shape=(n, 8, 3)
    return: shape=(n, 6, 4, 3)
    '''
    rec1 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 1], bboxes_corners[:, 3], bboxes_corners[:, 2]], axis=1) # (n, 4, 3)
    rec2 = np.stack([bboxes_corners[:, 4], bboxes_corners[:, 7], bboxes_corners[:, 6], bboxes_corners[:, 5]], axis=1) # (n, 4, 3)
    rec3 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 4], bboxes_corners[:, 5], bboxes_corners[:, 1]], axis=1) # (n, 4, 3)
    rec4 = np.stack([bboxes_corners[:, 2], bboxes_corners[:, 6], bboxes_corners[:, 7], bboxes_corners[:, 3]], axis=1) # (n, 4, 3)
    rec5 = np.stack([bboxes_corners[:, 1], bboxes_corners[:, 5], bboxes_corners[:, 6], bboxes_corners[:, 2]], axis=1) # (n, 4, 3)
    rec6 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 3], bboxes_corners[:, 7], bboxes_corners[:, 4]], axis=1) # (n, 4, 3)
    group_rectangle_vertexs = np.stack([rec1, rec2, rec3, rec4, rec5, rec6], axis=1)
    return group_rectangle_vertexs


# # modified from https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/utils.py#L11
def limit_period(val, offset=0.5, period=np.pi):
    """
    val: array or float
    offset: float
    period: float
    return: Value in the range of [-offset * period, (1-offset) * period]
    """
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val


def nearest_bev(bboxes):
    '''
    bboxes: (n, 7), (x, y, z, w, l, h, theta)
    return: (n, 4), (x1, y1, x2, y2)
    '''    
    bboxes_bev = copy.deepcopy(bboxes[:, [0, 1, 3, 4]])
    bboxes_angle = limit_period(bboxes[:, 6].cpu(), offset=0.5, period=np.pi).to(bboxes_bev)
    bboxes_bev = torch.where(torch.abs(bboxes_angle[:, None]) > np.pi / 4, bboxes_bev[:, [0, 1, 3, 2]], bboxes_bev)
    
    bboxes_xy = bboxes_bev[:, :2]
    bboxes_wl = bboxes_bev[:, 2:]
    bboxes_bev_x1y1x2y2 = torch.cat([bboxes_xy - bboxes_wl / 2, bboxes_xy + bboxes_wl / 2], dim=-1)
    return bboxes_bev_x1y1x2y2


def iou2d(bboxes1, bboxes2, metric=0):
    '''
    bboxes1: (n, 4), (x1, y1, x2, y2)
    bboxes2: (m, 4), (x1, y1, x2, y2)
    return: (n, m)
    '''
    bboxes_x1 = torch.maximum(bboxes1[:, 0][:, None], bboxes2[:, 0][None, :]) # (n, m)
    bboxes_y1 = torch.maximum(bboxes1[:, 1][:, None], bboxes2[:, 1][None, :]) # (n, m)
    bboxes_x2 = torch.minimum(bboxes1[:, 2][:, None], bboxes2[:, 2][None, :])
    bboxes_y2 = torch.minimum(bboxes1[:, 3][:, None], bboxes2[:, 3][None, :])

    bboxes_w = torch.clamp(bboxes_x2 - bboxes_x1, min=0)
    bboxes_h = torch.clamp(bboxes_y2 - bboxes_y1, min=0)

    iou_area = bboxes_w * bboxes_h # (n, m)
    
    bboxes1_wh = bboxes1[:, 2:] - bboxes1[:, :2]
    area1 = bboxes1_wh[:, 0] * bboxes1_wh[:, 1] # (n, )
    bboxes2_wh = bboxes2[:, 2:] - bboxes2[:, :2]
    area2 = bboxes2_wh[:, 0] * bboxes2_wh[:, 1] # (m, )
    if metric == 0:
        iou = iou_area / (area1[:, None] + area2[None, :] - iou_area + 1e-8)
    elif metric == 1:
        iou = iou_area / (area1[:, None] + 1e-8)
    return iou


def iou2d_nearest(bboxes1, bboxes2):
    '''
    bboxes1: (n, 7), (x, y, z, w, l, h, theta)
    bboxes2: (m, 7),
    return: (n, m)
    '''
    bboxes1_bev = nearest_bev(bboxes1)
    bboxes2_bev = nearest_bev(bboxes2)
    iou = iou2d(bboxes1_bev, bboxes2_bev)
    return iou


def iou3d(bboxes1, bboxes2):
    '''
    bboxes1: (n, 7), (x, y, z, w, l, h, theta)
    bboxes2: (m, 7)
    return: (n, m)
    '''
    # 1. height overlap
    bboxes1_bottom, bboxes2_bottom = bboxes1[:, 2], bboxes2[:, 2] # (n, ), (m, )
    bboxes1_top, bboxes2_top = bboxes1[:, 2] + bboxes1[:, 5], bboxes2[:, 2] + bboxes2[:, 5] # (n, ), (m, )
    bboxes_bottom = torch.maximum(bboxes1_bottom[:, None], bboxes2_bottom[None, :]) # (n, m) 
    bboxes_top = torch.minimum(bboxes1_top[:, None], bboxes2_top[None, :])
    height_overlap =  torch.clamp(bboxes_top - bboxes_bottom, min=0)

    # 2. bev overlap
    bboxes1_x1y1 = bboxes1[:, :2] - bboxes1[:, 3:5] / 2
    bboxes1_x2y2 = bboxes1[:, :2] + bboxes1[:, 3:5] / 2
    bboxes2_x1y1 = bboxes2[:, :2] - bboxes2[:, 3:5] / 2
    bboxes2_x2y2 = bboxes2[:, :2] + bboxes2[:, 3:5] / 2
    bboxes1_bev = torch.cat([bboxes1_x1y1, bboxes1_x2y2, bboxes1[:, 6:]], dim=-1)
    bboxes2_bev = torch.cat([bboxes2_x1y1, bboxes2_x2y2, bboxes2[:, 6:]], dim=-1)
    bev_overlap = boxes_overlap_bev(bboxes1_bev, bboxes2_bev) # (n, m)

    # 3. overlap and volume
    overlap = height_overlap * bev_overlap
    volume1 = bboxes1[:, 3] * bboxes1[:, 4] * bboxes1[:, 5]
    volume2 = bboxes2[:, 3] * bboxes2[:, 4] * bboxes2[:, 5]
    volume = volume1[:, None] + volume2[None, :] # (n, m)

    # 4. iou
    iou = overlap / (volume - overlap + 1e-8)

    return iou
    

def iou3d_camera(bboxes1, bboxes2):
    '''
    bboxes1: (n, 7), (x, y, z, w, l, h, theta)
    bboxes2: (m, 7)
    return: (n, m)
    '''
    # 1. height overlap
    bboxes1_bottom, bboxes2_bottom = bboxes1[:, 1] - bboxes1[:, 4], bboxes2[:, 1] -  bboxes2[:, 4] # (n, ), (m, )
    bboxes1_top, bboxes2_top = bboxes1[:, 1], bboxes2[:, 1] # (n, ), (m, )
    bboxes_bottom = torch.maximum(bboxes1_bottom[:, None], bboxes2_bottom[None, :]) # (n, m) 
    bboxes_top = torch.minimum(bboxes1_top[:, None], bboxes2_top[None, :])
    height_overlap =  torch.clamp(bboxes_top - bboxes_bottom, min=0)

    # 2. bev overlap
    bboxes1_x1y1 = bboxes1[:, [0, 2]] - bboxes1[:, [3, 5]] / 2
    bboxes1_x2y2 = bboxes1[:, [0, 2]] + bboxes1[:, [3, 5]] / 2
    bboxes2_x1y1 = bboxes2[:, [0, 2]] - bboxes2[:, [3, 5]] / 2
    bboxes2_x2y2 = bboxes2[:, [0, 2]] + bboxes2[:, [3, 5]] / 2
    bboxes1_bev = torch.cat([bboxes1_x1y1, bboxes1_x2y2, bboxes1[:, 6:]], dim=-1)
    bboxes2_bev = torch.cat([bboxes2_x1y1, bboxes2_x2y2, bboxes2[:, 6:]], dim=-1)
    bev_overlap = boxes_overlap_bev(bboxes1_bev, bboxes2_bev) # (n, m)

    # 3. overlap and volume
    overlap = height_overlap * bev_overlap
    volume1 = bboxes1[:, 3] * bboxes1[:, 4] * bboxes1[:, 5]
    volume2 = bboxes2[:, 3] * bboxes2[:, 4] * bboxes2[:, 5]
    volume = volume1[:, None] + volume2[None, :] # (n, m)

    # 4. iou
    iou = overlap / (volume - overlap + 1e-8)

    return iou


def iou_bev(bboxes1, bboxes2):
    '''
    bboxes1: (n, 5), (x, z, w, h, theta)
    bboxes2: (m, 5)
    return: (n, m)
    '''
    bboxes1_x1y1 = bboxes1[:, :2] - bboxes1[:, 2:4] / 2
    bboxes1_x2y2 = bboxes1[:, :2] + bboxes1[:, 2:4] / 2
    bboxes2_x1y1 = bboxes2[:, :2] - bboxes2[:, 2:4] / 2
    bboxes2_x2y2 = bboxes2[:, :2] + bboxes2[:, 2:4] / 2
    bboxes1_bev = torch.cat([bboxes1_x1y1, bboxes1_x2y2, bboxes1[:, 4:]], dim=-1)
    bboxes2_bev = torch.cat([bboxes2_x1y1, bboxes2_x2y2, bboxes2[:, 4:]], dim=-1)
    bev_overlap = boxes_iou_bev(bboxes1_bev, bboxes2_bev) # (n, m)

    return bev_overlap


def keep_bbox_from_lidar_range(result, pcd_limit_range):
    '''
    result: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    pcd_limit_range: []
    return: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    '''
    lidar_bboxes, labels, scores = result['lidar_bboxes'], result['labels'], result['scores']
    if 'bboxes2d' not in result:
        result['bboxes2d'] = np.zeros_like(lidar_bboxes[:, :4])
    if 'camera_bboxes' not in result:
        result['camera_bboxes'] = np.zeros_like(lidar_bboxes)
    bboxes2d, camera_bboxes = result['bboxes2d'], result['camera_bboxes']
    flag1 = lidar_bboxes[:, :3] > pcd_limit_range[:3][None, :] # (n, 3)
    flag2 = lidar_bboxes[:, :3] < pcd_limit_range[3:][None, :] # (n, 3)
    keep_flag = np.all(flag1, axis=-1) & np.all(flag2, axis=-1)
    
    result = {
        'lidar_bboxes': lidar_bboxes[keep_flag],
        'labels': labels[keep_flag],
        'scores': scores[keep_flag],
        'bboxes2d': bboxes2d[keep_flag],
        'camera_bboxes': camera_bboxes[keep_flag]
    }
    return result

