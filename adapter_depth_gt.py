from operator import le
import os
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import (frame_utils, range_image_utils,
                                      transform_utils)
import numpy as np
import itertools

import matplotlib.pyplot as plt
import matplotlib.patches as patches



def get_dirnames(base_dir):

  everything_in_folder = os.listdir(base_dir)
  all_dirs = map(lambda x: os.path.join(base_dir, x), everything_in_folder)
  dir_list = list(filter(os.path.isdir, all_dirs))

  return dir_list

def get_filename(file_folder):

  file_list = os.listdir(file_folder)
  file_list = map(lambda x: os.path.join(file_folder, x), file_list)
  file_list = list(filter(os.path.isfile, file_list))

  return file_list


def rgba(r):
  """Generates a color based on range.

  Args:
    r: the range value of a given point.
  Returns:
    The color for a given range
  """
  c = plt.get_cmap('jet')((r % 20.0) / 20.0)
  c = list(c)
  c[-1] = 0.5  # alpha
  return c

def plot_image(camera_image):
  """Plot a cmaera image."""
  plt.figure(figsize=(20, 12))
  plt.imshow(tf.image.decode_jpeg(camera_image.image))
  plt.grid("off")

def plot_points_on_image(projected_points, camera_image, rgba_func,
                         point_size=5.0):
  """Plots points on a camera image.

  Args:
    projected_points: [N, 3] numpy array. The inner dims are
      [camera_x, camera_y, range].
    camera_image: jpeg encoded camera image.
    rgba_func: a function that generates a color from a range value.
    point_size: the point size.

  """
  plot_image(camera_image)

  xs = []
  ys = []
  colors = []

  for point in projected_points:
    xs.append(point[0])  # width, col
    ys.append(point[1])  # height, row
    colors.append(rgba_func(point[2]))

  plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")


def pc2img_range(frame, points_all, cp_points_all, image):

    # images = sorted(frame.images, key=lambda i: i.name)
    ranges_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
    cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)
    mask = tf.equal(cp_points_all_tensor[:, 0], image.name)
    cp_points_all_tensor = tf.cast( tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32) 
    ranges_all_tensor = tf.gather_nd(ranges_all_tensor, tf.where(mask))
    projected_points_all_from_raw_data = tf.concat([cp_points_all_tensor[:, 1:3], ranges_all_tensor], axis=-1).numpy()

    return projected_points_all_from_raw_data

def pc2img_depth(frame, points_all, cp_points_all, image):

    # images = sorted(frame.images, key=lambda i: i.name)
    depth_all_tensor = points_all[:, 0]
    cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)
    mask = tf.equal(cp_points_all_tensor[:, 0], image.name)
    cp_points_all_tensor = tf.cast( tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32) 
    ranges_all_tensor = tf.gather_nd(ranges_all_tensor, tf.wherer(mask))
    projected_points_all_from_raw_data = tf.concat([cp_points_all_tensor[:, 1:3], ranges_all_tensor], axis=-1).numpy()

    return projected_points_all_from_raw_data

def waymo_adapter(tfrecord_path,flag = True):
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
        print(len(points))
        print('****************')
        points_all = np.concatenate(points, axis=0)
        points_all = points_all[:, 3:]
        depth = points_all[:, 0]
        cp_points_all = np.concatenate(cp_points, axis=0)
        print(cp_points_all)
        print('--------------------')

        images = sorted(frame.images, key=lambda i: i.name)
        projected_points_all_from_raw_data = pc2img_range(frame, points_all, cp_points_all, images[0])
        plot_points_on_image(projected_points_all_from_raw_data,
                     images[0], rgba, point_size=5.0)


        pc_info_dict = dict()
        pc_info_dict['pc'] = points_all
        pc_info_dict['depth'] = depth
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
        break
    return points, camera_calibration



if __name__ == '__main__':

  base_dir = '/mnt/lustre/share_data/PerceptionX/data/mono3d/waymo_v1.2/training'
  files = get_filename(base_dir)
  data = files[0]
  waymo_adapter(data)
  
