# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import json
import mmcv

from mmdet3d.apis import (inference_mono_3d_detector, init_model,
                          show_result_meshlab)

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
                Modified from NUSC

    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Config file',
                        default="configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py")
    parser.add_argument('--checkpoint', help='Checkpoint file',
                        default='checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth')
    parser.add_argument(
        "--input",
        default='/media/yuliangguo/data_ssd_4tb/Datasets/kitti/training/image_2',
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.png'",
    )
    parser.add_argument(
        "--output",
        default='/home/yuliangguo/Projects/nerf-auto-driving/data/third_party_det3D/fcos3d/result_kitti/training/label_2',
        help="A file or directory to save output results. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--label-dir",
        default='/media/yuliangguo/data_ssd_4tb/Datasets/kitti/training/calib',
        help="where nusc labels saved"
    )
    parser.add_argument(
        '--score-thr', type=float, default=0.15, help='bbox score threshold')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    return parser


if __name__ == "__main__":
    class_names = np.array(class_names)
    # mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # demo = VisualizationDemo(cfg)
    model = init_model(args.config, args.checkpoint, device='cuda:0')

    if args.input:
        input_files = glob.glob(args.input + '/*.png')
        for path in tqdm.tqdm(input_files, disable=not args.output):

            calib_file = os.path.join(args.label_dir, os.path.basename(path)[:-4] + '.txt')
            calibs = {}
            with open(calib_file, "r") as f:
                for line in f.readlines():
                    line = line.rstrip()
                    if len(line) == 0:
                        continue
                    key, value = line.split(":", 1)
                    # The only non-float values in these files are dates, which
                    # we don't care about anyway
                    try:
                        calibs[key] = np.array([float(x) for x in value.split()])
                    except ValueError:
                        pass
            P = calibs["P2"]
            P = np.reshape(P, [3, 4])
            json_data = {"images": [{"file_name": path, "cam_intrinsic": P[:, :3].tolist()}]}
            output_file = 'temp_dict.json'
            with open(output_file, 'w') as f:
                json.dump(json_data, f)

            start_time = time.time()
            result, data = inference_mono_3d_detector(model, path, output_file)

            out_label_file = os.path.join(args.output, os.path.basename(path)[:-4] + '.txt')
            keep_indices = np.where(result[0]['img_bbox']['scores_3d'] > args.score_thr)[0]

            with open(out_label_file, 'a') as FILE:
                # TODO: Critical to adapt the depth based on cross-dataset focal length ratio! --> follow DEVIANT
                result[0]['img_bbox']['boxes_3d'].corners[keep_indices] /= 1.361
                result[0]['img_bbox']['boxes_3d'].bottom_center[keep_indices] /= 1.361
                for obj_idx in keep_indices:
                    type = class_names[result[0]['img_bbox']['labels_3d'][obj_idx]]
                    truncated = 0
                    occluded = 0
                    alpha = result[0]['img_bbox']['boxes_3d'].local_yaw[obj_idx].numpy()
                    corners_3d = result[0]['img_bbox']['boxes_3d'].corners[obj_idx].numpy().T
                    corners_2d = view_points(corners_3d, P[:, :3], normalize=True)
                    bbox = np.round([corners_2d[0].min(), corners_2d[1].min(), corners_2d[0].max(), corners_2d[1].max()], 2)
                    dimensions = np.round(result[0]['img_bbox']['boxes_3d'].dims[obj_idx][[2, 1, 0]].numpy(), 2)
                    location = np.round(result[0]['img_bbox']['boxes_3d'].bottom_center[obj_idx].numpy(), 2)
                    rotation_y = np.round(result[0]['img_bbox']['boxes_3d'].yaw[obj_idx].numpy(), 2)
                    score = np.round(result[0]['img_bbox']['scores_3d'][obj_idx].numpy(), 2)

                    out_line = f'{type.capitalize()} {truncated} {occluded} {alpha}' \
                               f' {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}' \
                               f' {dimensions[0]} {dimensions[1]} {dimensions[2]}' \
                               f' {location[0]} {location[1]} {location[2]}' \
                               f' {rotation_y} {score}\n'
                    FILE.write(out_line)

            # out_dict = {}
            # out_dict['boxes_center'] = result[0]['img_bbox']['boxes_3d'].center[keep_indices].tolist()
            # out_dict['boxes_bottom_center'] = result[0]['img_bbox']['boxes_3d'].bottom_center[keep_indices].tolist()
            # out_dict['boxes_bottom_height'] = result[0]['img_bbox']['boxes_3d'].bottom_height[keep_indices].tolist()
            # out_dict['boxes_dims'] = result[0]['img_bbox']['boxes_3d'].dims[keep_indices].tolist()
            # out_dict['boxes_height'] = result[0]['img_bbox']['boxes_3d'].height[keep_indices].tolist()
            # out_dict['boxes_local_yaw'] = result[0]['img_bbox']['boxes_3d'].local_yaw[keep_indices].tolist()
            # out_dict['boxes_volume'] = result[0]['img_bbox']['boxes_3d'].volume[keep_indices].tolist()
            # out_dict['boxes_yaw'] = result[0]['img_bbox']['boxes_3d'].yaw[keep_indices].tolist()
            # out_dict['scores'] = result[0]['img_bbox']['scores_3d'][keep_indices].tolist()
            # out_dict['labels'] = result[0]['img_bbox']['labels_3d'][keep_indices].tolist()
            # out_dict['attrs'] = result[0]['img_bbox']['attrs_3d'][keep_indices].tolist()
            # out_dict['class_names'] = class_names

            if args.snapshot:  # the later one seems not effective, always save output
                show_result_meshlab(
                    data,
                    result,
                    args.output,
                    args.score_thr,
                    show=args.show,
                    snapshot=args.snapshot,
                    task='mono-det')

