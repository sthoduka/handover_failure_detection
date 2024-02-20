#!/usr/bin/python

'''
This example shows how the sensor data (wrench, joint positions, joint velocities, etc.) are resampled
to the same frequency as the camera.
The main idea is to select the data points from the sensor data whose timestamp is closest
to the timestamps of each frame of the video.
'''

import os
import glob
import numpy as np

def resample_joint_data_to_camera_freq(data, data_ts, camera_ts):
    resampled_data = []
    for cam_ts in camera_ts:
        # find sensor ts closest to camera ts
        sensor_idx = np.argmin(np.abs(data_ts - cam_ts))
        resampled_data.append(data[sensor_idx])
    resampled_data = np.array(resampled_data)
    return resampled_data

def main():
    root = '/path/to/training_set'
    trials = sorted(glob.glob(root + '/*'))
    sensor_data_file = 'joint_positions.npy'
    resampled_sensor_data_file = 'joint_pos_resampled.npy'
    sensor_data_ts_file = 'joint_state_ts.npy'

    '''
    sensor_data_file = 'wrench.npy'
    resampled_sensor_data_file = 'wrench_resampled.npy'
    sensor_data_ts_file = 'wrench_ts.npy'
    '''
    for trial in trials:
        camera_ts = np.load(os.path.join(trial, 'head_cam_ts.npy'))
        data_ts = np.load(os.path.join(trial, sensor_data_ts_file))
        data = np.load(os.path.join(trial, sensor_data_file))
        resampled = resample_joint_data_to_camera_freq(data, data_ts, camera_ts)
        np.save(os.path.join(trial, resampled_sensor_data_file), resampled)

if __name__ == '__main__':
    main()
