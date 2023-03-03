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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Config file',
                        default="configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py")
    parser.add_argument('--checkpoint', help='Checkpoint file',
                        default='checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth')
    parser.add_argument(
        "--input",
        default='/mnt/LinuxDataFast/Datasets/NuScenes/v1.0-mini/samples/',
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='/mnt/LinuxDataFast/Datasets/NuScenes/v1.0-mini/pred_det3d/',
        help="A file or directory to save output results. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--label-dir",
        default='/mnt/LinuxDataFast/Datasets/NuScenes/v1.0-mini/v1.0-mini/',
        help="where nusc labels saved"
    )
    parser.add_argument('--camera-id',
                        help='the camera id from nuscenes',
                        default='CAM_FRONT',
                        type=str)
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
    args = parser.parse_args()
    return parser


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    args.input = os.path.join(args.input, args.camera_id)
    args.output = os.path.join(args.output, args.camera_id)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.snapshot:
        vis_dir = os.path.join(args.output, 'vis')
        os.makedirs(vis_dir)

    with open(os.path.join(args.label_dir, 'sensor.json'), 'r') as f:
        sensors = json.load(f)

    with open(os.path.join(args.label_dir, 'calibrated_sensor.json'), 'r') as f:
        calibrated_sensors = json.load(f)

    sensor_token = None
    for sensor in sensors:
        if sensor['channel'] == args.camera_id:
            sensor_token = sensor['token']
            break
    cam_intrinsic = None
    for calibrated_sensor in calibrated_sensors:
        if calibrated_sensor['sensor_token'] == sensor_token:
            cam_intrinsic = calibrated_sensor['camera_intrinsic']
            break

    # demo = VisualizationDemo(cfg)
    model = init_model(args.config, args.checkpoint, device='cuda:0')

    if args.input:
        input_files = glob.glob(args.input + '/*.jpg')
        for path in tqdm.tqdm(input_files, disable=not args.output):
            # use PIL, to be consistent with evaluation
            # img = read_image(path, format="BGR")

            json_data = {"images": [{"file_name": path, "cam_intrinsic": cam_intrinsic}]}
            output_file = 'temp_dict.json'
            with open(output_file, 'w') as f:
                json.dump(json_data, f)

            start_time = time.time()
            result, data = inference_mono_3d_detector(model, path, output_file)

            # TODO: save prediction results in economic way
            out_jsonfile = os.path.join(args.output, os.path.basename(path)[:-4]+'.json')
            out_dict = {}
            keep_indices = np.where(result[0]['img_bbox']['scores_3d'] > args.score_thr)[0]

            out_dict['boxes_center'] = result[0]['img_bbox']['boxes_3d'].center[keep_indices].tolist()
            out_dict['boxes_bottom_center'] = result[0]['img_bbox']['boxes_3d'].bottom_center[keep_indices].tolist()
            out_dict['boxes_bottom_height'] = result[0]['img_bbox']['boxes_3d'].bottom_height[keep_indices].tolist()
            out_dict['boxes_dims'] = result[0]['img_bbox']['boxes_3d'].dims[keep_indices].tolist()
            out_dict['boxes_height'] = result[0]['img_bbox']['boxes_3d'].height[keep_indices].tolist()
            out_dict['boxes_local_yaw'] = result[0]['img_bbox']['boxes_3d'].local_yaw[keep_indices].tolist()
            out_dict['boxes_volume'] = result[0]['img_bbox']['boxes_3d'].volume[keep_indices].tolist()
            out_dict['boxes_yaw'] = result[0]['img_bbox']['boxes_3d'].yaw[keep_indices].tolist()
            out_dict['scores'] = result[0]['img_bbox']['scores_3d'][keep_indices].tolist()
            out_dict['labels'] = result[0]['img_bbox']['labels_3d'][keep_indices].tolist()
            out_dict['attrs'] = result[0]['img_bbox']['attrs_3d'][keep_indices].tolist()
            out_dict['class_names'] = class_names

            with open(out_jsonfile, 'w') as f:
                json.dump(out_dict, f)

            show_result_meshlab(
                data,
                result,
                args.output,
                args.score_thr,
                show=args.show,
                snapshot=args.snapshot,
                task='mono-det')

