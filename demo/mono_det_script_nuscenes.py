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
        default='/media/yuliangguo/data_ssd_4tb/Datasets/nuscenes_yuliang/samples/',
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='/media/yuliangguo/data_ssd_4tb/Datasets/nuscenes_yuliang/pred_det3d/',
        help="A file or directory to save output results. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--label-dir",
        default='/media/yuliangguo/data_ssd_4tb/Datasets/nuscenes_yuliang/v1.0-trainval/',
        help="where nusc labels saved"
    )
    parser.add_argument('--camera-id',
                        help='the camera id from nuscenes',
                        default='CAM_BACK_LEFT',
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
    return parser


if __name__ == "__main__":
    class_names = np.array(class_names)
    # mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    args.input = os.path.join(args.input, args.camera_id)
    args.output = os.path.join(args.output, args.camera_id)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with open(os.path.join(args.label_dir, 'calibrated_sensor.json'), 'r') as f:
        calibrated_sensors = json.load(f)

    with open(os.path.join(args.label_dir, 'sample_data.json'), 'r') as f:
        sample_data = json.load(f)


    # demo = VisualizationDemo(cfg)
    model = init_model(args.config, args.checkpoint, device='cuda:0')

    if args.input:
        input_files = glob.glob(args.input + '/*.jpg')
        for path in tqdm.tqdm(input_files, disable=not args.output):
            img_name = os.path.basename(path)
            # if img_name != 'n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151613947405.jpg':
            #     continue

            # A complicated way to get camera intrinsics because there single cam-id may have multiple caliab sensors
            key1 = "filename"
            val1 = "samples/" + args.camera_id + "/" + img_name
            d1 = next((d1 for d1 in sample_data if d1.get(key1) == val1), None)
            key2 = 'token'
            val2 = d1['calibrated_sensor_token']
            d2 = next((d2 for d2 in calibrated_sensors if d2.get(key2) == val2), None)
            cam_intrinsic = d2['camera_intrinsic']
            json_data = {"images": [{"file_name": path, "cam_intrinsic": cam_intrinsic}]}
            output_file = 'temp_dict.json'
            with open(output_file, 'w') as f:
                json.dump(json_data, f)

            start_time = time.time()
            result, data = inference_mono_3d_detector(model, path, output_file)

            out_jsonfile = os.path.join(args.output, img_name[:-4]+'.json')
            out_dict = {}
            keep_indices = np.where(result[0]['img_bbox']['scores_3d'] > args.score_thr)[0]

            # TODO: Box center is strange to be same as bottom center
            boxes_center = result[0]['img_bbox']['boxes_3d'].center[keep_indices].numpy()
            boxes_dims = result[0]['img_bbox']['boxes_3d'].dims[keep_indices].numpy()
            # dims: l h w
            boxes_center[:, 1] = boxes_center[:, 1] - boxes_dims[:, 1]/2
            out_dict['boxes_center'] = boxes_center.tolist()
            out_dict['boxes_bottom_center'] = result[0]['img_bbox']['boxes_3d'].bottom_center[keep_indices].tolist()
            out_dict['boxes_gravity_center'] = result[0]['img_bbox']['boxes_3d'].gravity_center[keep_indices].tolist()
            out_dict['corners_3d'] = result[0]['img_bbox']['boxes_3d'].corners[keep_indices].numpy().tolist()
            # out_dict['boxes_bottom_height'] = result[0]['img_bbox']['boxes_3d'].bottom_height[keep_indices].tolist()
            out_dict['boxes_dims'] = boxes_dims.tolist()
            # out_dict['boxes_height'] = result[0]['img_bbox']['boxes_3d'].height[keep_indices].tolist()
            out_dict['boxes_local_yaw'] = result[0]['img_bbox']['boxes_3d'].local_yaw[keep_indices].tolist()
            # out_dict['boxes_volume'] = result[0]['img_bbox']['boxes_3d'].volume[keep_indices].tolist()
            out_dict['boxes_yaw'] = result[0]['img_bbox']['boxes_3d'].yaw[keep_indices].tolist()
            out_dict['scores'] = result[0]['img_bbox']['scores_3d'][keep_indices].tolist()
            # out_dict['attrs'] = result[0]['img_bbox']['attrs_3d'][keep_indices].tolist()
            label_indices = result[0]['img_bbox']['labels_3d'][keep_indices].tolist()
            out_dict['classes'] = class_names[label_indices].tolist()

            with open(out_jsonfile, 'w') as f:
                json.dump(out_dict, f)

            if args.snapshot:  # the later one seems not effective, always save output
                show_result_meshlab(
                    data,
                    result,
                    args.output,
                    args.score_thr,
                    show=args.show,
                    snapshot=args.snapshot,
                    task='mono-det')

