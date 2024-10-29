"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
from data_utils import vis_pc

# --------------------------for pointnet++ ----------------------------#
import cv2
import pdb
from utils import setup_seed, read_points, read_calib, read_label, \
    keep_bbox_from_lidar_range, vis_pc, \
    bbox3d2corners_camera
from model import PointPillars

def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 
# --------------------------for pointnet++ ----------------------------#
    

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, default = 'pointnet2_part_seg_msg',required=True, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth', help='your checkpoint for kitti')
    parser.add_argument('--pc_path', default='dataset/demo_data/val/000134.bin',help='your point cloud path')
    parser.add_argument('--calib_path', default='', help='your calib file path')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    return parser.parse_args()


def main(args):
    # --------------------------from pointpillar------------------------------------------#
    CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
        }
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
    pcd_limit_range = np.array([-5.0, -5.0, -5.0, 5.0, 5.0, 5.0], dtype=np.float32)
    if not args.no_cuda:
        model_pillar = PointPillars(nclasses=len(CLASSES)).cuda()
        model_pillar.load_state_dict(torch.load(args.ckpt))
    else:
        model_pillar = PointPillars(nclasses=len(CLASSES))
        model_pillar.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu')))
    
    # ------------------------------------------------------------------------------------#
    
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/part_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    

    original_filenames = TEST_DATASET.original_filenames
    print(original_filenames)

    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()
        for batch_id, (original_point, points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                                      smoothing=0.9):
  
            original_point_values = original_point.cpu().data.numpy()
            print(np.shape(original_point_values))  # 应该是(17, 2048, 6)

            for point_values, filename in zip(original_point_values, original_filenames):
                output_file_path = os.path.join('ori_data', f'{filename}.txt')
                np.savetxt(output_file_path, point_values.reshape(-1, point_values.shape[-1]), fmt='%.6f')
                
                # # 这一部分主要是用于点云的可视化操作 读取点云数据的前三列[x,y,z]用于后续的目标检测阶段
                # data = np.loadtxt(output_file_path)
                # pc_loaded = data[:,:3]

                # pc_torch = torch.from_numpy(pc_loaded).float()
                # print(pc_loaded.shape)
                # vis_pc(pc_loaded)
                
                

            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            
            if not args.use_cpu:
                points, label, target = points.cuda(), label.cuda(), target.cuda()
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()
            if not args.use_cpu:
                vote_pool = vote_pool.cuda()

            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes)) #seg_predb 包含每个点分类概率的张量
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes #得到每个点的平均概率
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size): #遍历每个样本点进行分类
                cat = seg_label_to_cat[target[i, 0]] #target[i, 0]获得对应类别，查找该类别在seg_label_to_cat字典中的值
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0] #计算每个点的预测类别

            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=float))
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        # 将 cur_pred_val 的每个类别预测值添加到 ori_data 文件夹下的新文本文件
        ori_data_path = 'ori_data/'
        for point_values, filename in zip(cur_pred_val, original_filenames):  # 同时遍历预测值和文件名
            original_points_file = os.path.join(ori_data_path, f'{filename}.txt')
            new_file_path = os.path.join(ori_data_path, f'{filename}_pred.txt')  # 新的预测结果文件名
            original_points = np.loadtxt(original_points_file)  # 读取文本文件内容，假设形状为 (2048, 6)

            
            # 确保原始点数据的形状和 cur_pred_val 的形状匹配
            if original_points.shape[0] != point_values.shape[0]:
                raise ValueError(
                    f"Shape mismatch: {original_points.shape[0]} (from file) != {point_values.shape[0]} (cur_pred_val)")

            # 将 cur_pred_val 的当前类别预测值添加为最后一列
            updated_points = np.hstack((original_points, point_values[:, np.newaxis]))  # 添加最后一列
            # 保存更新后的数据到新文件
            np.savetxt(new_file_path, updated_points, fmt='%.6f')

            #当前在遍历测试集中的每个文件，在这个阶段同时进行目标检测
            # --------------------------from pointpillar------------------------------------------#
            pc_loaded = np.loadtxt(new_file_path)[:,:4]#z数据包含四列，分别是[x,y,z,激光雷达的反射强度]
            pc_torch = torch.from_numpy(pc_loaded).float()
            # 检测路径是否存在
            if os.path.exists(args.calib_path):
                calib_info = read_calib(args.calib_path)
            else:
                calib_info = None
            model_pillar.eval()
            # 此处是使用训练好的模型进行推理
            with torch.no_grad(): #因为在推理的时候不需要计算梯度，使用orch.no_grad()可以提高效率
                if not args.no_cuda:
                    pc_torch = pc_torch.cuda() #检测是否使用cuda
                
                result_filter = model_pillar(batched_pts=[pc_torch], 
                                    mode='test')[0] #将pc_torch点云数据作为输入，给模型进行推理
            result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
            lidar_bboxes = result_filter['lidar_bboxes']
            
            labels, scores = result_filter['labels'], result_filter['scores']
            print('point labels:{}'.format(result_filter['labels']))
            vis_pc(pc_loaded, bboxes=lidar_bboxes, labels=labels)
            # ------------------------------------------------------------------------------------#

    log_string('Accuracy is: %.5f' % test_metrics['accuracy'])
    log_string('Class avg accuracy is: %.5f' % test_metrics['class_avg_accuracy'])
    log_string('Class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])
    log_string('Inctance avg mIOU is: %.5f' % test_metrics['inctance_avg_iou'])


if __name__ == '__main__':
    args = parse_args()
    main(args)
