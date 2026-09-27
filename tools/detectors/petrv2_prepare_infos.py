"""Build PETRv2 temporal info pkls for nuScenes (runs inside the petr container).

Steps:
  1. mmdet3d v0.17.1 `create_nuscenes_infos` -> {prefix}_infos_{train,val}.pkl
     (written into `--root`, which is a symlink farm under outputs/, so nothing
     is written into datasets/).
  2. Port of PETR's tools/generate_sweep_pkl.py (hard-coded paths/version made
     configurable) -> mmdet3d_nuscenes_30f_infos_{train,val}.pkl, which adds
     previous camera sweeps/key-frames to every info.

Usage:
  python tools/detectors/petrv2_prepare_infos.py --root outputs/petrv2/data/nuscenes \
      --version v1.0-mini --splits val
"""
import argparse
import os
import os.path as osp
import sys

import mmcv
import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion

SENSORS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
           'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']


def add_frame(nusc, data_root, sample_data, e2g_t, l2e_t, l2e_r_mat, e2g_r_mat):
    """Same as PETR's add_frame: camera sweep expressed in the key-frame lidar frame."""
    sweep_cam = dict()
    sweep_cam['is_key_frame'] = sample_data['is_key_frame']
    sweep_cam['data_path'] = os.path.join(data_root, sample_data['filename'])
    sweep_cam['type'] = 'camera'
    sweep_cam['timestamp'] = sample_data['timestamp']
    sweep_cam['sample_data_token'] = sample_data['sample_token']
    pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])
    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])

    l2e_r_s_mat = Quaternion(cs_record['rotation']).rotation_matrix
    e2g_r_s_mat = Quaternion(pose_record['rotation']).rotation_matrix
    l2e_t_s = cs_record['translation']
    e2g_t_s = pose_record['translation']
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep_cam['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep_cam['sensor2lidar_translation'] = T

    lidar2cam_r = np.linalg.inv(sweep_cam['sensor2lidar_rotation'])
    lidar2cam_t = sweep_cam['sensor2lidar_translation'] @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    intrinsic = np.array(cs_record['camera_intrinsic'])
    viewpad = np.eye(4)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
    sweep_cam['intrinsics'] = viewpad.astype(np.float32)
    sweep_cam['extrinsics'] = lidar2cam_rt.astype(np.float32)
    sweep_cam['lidar2img'] = lidar2img_rt.astype(np.float32)
    return sweep_cam


def make_paths_absolute(info):
    """mmdet3d stores paths relative to the cwd at conversion time; make them absolute
    so tools/test.py can be run from the PETR directory."""
    info['lidar_path'] = osp.abspath(info['lidar_path'])
    for cam_info in info['cams'].values():
        cam_info['data_path'] = osp.abspath(cam_info['data_path'])


def add_sweeps(nusc, data_root, key_infos, num_prev=5, num_sweep=5):
    """Same loop as PETR's generate_sweep_pkl.py."""
    for info in mmcv.track_iter_progress(key_infos['infos']):
        make_paths_absolute(info)
        e2g_t = info['ego2global_translation']
        l2e_t = info['lidar2ego_translation']
        l2e_r_mat = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        e2g_r_mat = Quaternion(info['ego2global_rotation']).rotation_matrix

        sample = nusc.get('sample', info['token'])
        current_cams = {cam: nusc.get('sample_data', sample['data'][cam]) for cam in SENSORS}
        sweep_lists = []
        for _ in range(num_prev):
            if sample['prev'] == '':  # first frame of the scene
                break
            # non-key camera sweeps between two key frames
            for _ in range(num_sweep):
                sweep_cams = dict()
                for cam in SENSORS:
                    if current_cams[cam]['prev'] == '':
                        sweep_cams = sweep_lists[-1]
                        break
                    sample_data = nusc.get('sample_data', current_cams[cam]['prev'])
                    sweep_cams[cam] = add_frame(nusc, data_root, sample_data, e2g_t, l2e_t,
                                                l2e_r_mat, e2g_r_mat)
                    current_cams[cam] = sample_data
                sweep_lists.append(sweep_cams)
            # previous key frame
            sample = nusc.get('sample', sample['prev'])
            sweep_cams = dict()
            for cam in SENSORS:
                sample_data = nusc.get('sample_data', sample['data'][cam])
                sweep_cams[cam] = add_frame(nusc, data_root, sample_data, e2g_t, l2e_t,
                                            l2e_r_mat, e2g_r_mat)
                current_cams[cam] = sample_data
            sweep_lists.append(sweep_cams)
        info['sweeps'] = sweep_lists
    return key_infos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='nuScenes data root (infos are written here)')
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--splits', nargs='+', default=['val'], choices=['train', 'val'])
    parser.add_argument('--mmdet3d-root', default='/opt/mmdetection3d')
    args = parser.parse_args()

    root = osp.abspath(args.root)
    prefix = 'nuscenes'
    sys.path.insert(0, args.mmdet3d_root)
    from tools.data_converter import nuscenes_converter  # mmdet3d v0.17.1

    base_files = [osp.join(root, f'{prefix}_infos_{s}.pkl') for s in ('train', 'val')]
    if not all(osp.exists(f) for f in base_files):
        nuscenes_converter.create_nuscenes_infos(root, prefix, version=args.version, max_sweeps=10)

    nusc = NuScenes(version=args.version, dataroot=root, verbose=False)
    for split in args.splits:
        key_infos = mmcv.load(osp.join(root, f'{prefix}_infos_{split}.pkl'))
        key_infos = add_sweeps(nusc, root, key_infos)
        out = osp.join(root, f'mmdet3d_nuscenes_30f_infos_{split}.pkl')
        mmcv.dump(key_infos, out)
        print(f'wrote {out} ({len(key_infos["infos"])} samples)')


if __name__ == '__main__':
    main()
