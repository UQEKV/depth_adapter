from operator import le
import os
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import (frame_utils, range_image_utils,
                                      transform_utils)
import numpy as np

def waymo_adapter(tfrecord_path,set_count,flag = True):
    count = 0
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')

    print('----------------Read Waymo Dataset----------------')
    
    for data in tqdm(dataset, desc='process '):
        frame = open_dataset.Frame()

        frame.ParseFromString(bytearray(data.numpy()))
        data_dict = frame_utils.convert_frame_to_dict(frame)
        (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
        # --- 点云 -- points

        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
          frame,
          range_images,
          camera_projections,
          range_image_top_pose,
          keep_polar_features=True)

        camera_calibration = frame.context.camera_calibrations[0]

        time = np.array(data_dict['TIMESTAMP'])
        points_all = np.concatenate(points, axis=0)
        pc_info_dict = dict()
        pc_info_dict['pc'] = points_all
        pc_info_dict['pose'] = frame.pose.transform
        pc_info_dict['laser_labels'] = frame.laser_labels
        pc_info_dict['camera_calibration'] = camera_calibration
        # -- 内参 --
        f_u   = camera_calibration.intrinsic[0]
        f_v   = camera_calibration.intrinsic[1]
        c_u   = camera_calibration.intrinsic[2]
        c_v   = camera_calibration.intrinsic[3]
        # -- 内参 --

        # -- 外参 --
        extrinsic = tf.reshape(camera_calibration.extrinsic.transform, [4, 4])
        vehicle_to_sensor = tf.linalg.inv(extrinsic)
        # -- 外参 --
        count +=1
    return points, camera_calibration