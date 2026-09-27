"""Build pseudo detections from nuScenes GT for pipeline sanity checks.

Each GT box is kept with probability `keep`, jittered, and given a random
score; false positives are added around the ego vehicle. The output follows
the nuScenes detection submission format (global frame, with velocities).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytetrack_v2.datasets.nuscenes import TRACKING_CLASSES  # noqa: E402

CATEGORY_TO_CLASS = {
    "vehicle.bicycle": "bicycle", "vehicle.bus.bendy": "bus", "vehicle.bus.rigid": "bus",
    "vehicle.car": "car", "vehicle.motorcycle": "motorcycle", "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian", "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian", "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", default="datasets/nuscenes")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--split", default="mini_val")
    p.add_argument("--out", default="outputs/pseudo_dets/mini_val.json")
    p.add_argument("--keep", type=float, default=0.9)
    p.add_argument("--pos-noise", type=float, default=0.2)
    p.add_argument("--vel-noise", type=float, default=0.5)
    p.add_argument("--fp-per-sample", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    rng = np.random.default_rng(args.seed)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    scenes = set(create_splits_scenes()[args.split])
    results = {}
    for sample in nusc.sample:
        if nusc.get("scene", sample["scene_token"])["name"] not in scenes:
            continue
        dets = []
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            cls = CATEGORY_TO_CLASS.get(ann["category_name"])
            if cls is None or ann["num_lidar_pts"] + ann["num_radar_pts"] == 0 or rng.random() > args.keep:
                continue
            vel = np.nan_to_num(nusc.box_velocity(ann_token)[:2])
            dets.append({
                "sample_token": sample["token"],
                "translation": (np.array(ann["translation"]) + rng.normal(0, args.pos_noise, 3)).tolist(),
                "size": ann["size"],
                "rotation": ann["rotation"],
                "velocity": (vel + rng.normal(0, args.vel_noise, 2)).tolist(),
                "detection_name": cls,
                "detection_score": float(rng.uniform(0.05, 1.0) ** 0.5),
                "attribute_name": "",
            })
        ego = nusc.get("ego_pose", nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["ego_pose_token"])
        for _ in range(args.fp_per_sample):
            xy = np.array(ego["translation"][:2]) + rng.uniform(-50, 50, 2)
            dets.append({
                "sample_token": sample["token"],
                "translation": [float(xy[0]), float(xy[1]), ego["translation"][2]],
                "size": [1.9, 4.5, 1.7], "rotation": [1.0, 0.0, 0.0, 0.0], "velocity": [0.0, 0.0],
                "detection_name": str(rng.choice(TRACKING_CLASSES)),
                "detection_score": float(rng.uniform(0.0, 0.4)),
                "attribute_name": "",
            })
        results[sample["token"]] = dets

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    meta = {"use_camera": False, "use_lidar": True, "use_radar": False, "use_map": False, "use_external": False}
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "results": results}, f)
    print(f"wrote {sum(len(v) for v in results.values())} detections for {len(results)} samples to {args.out}")


if __name__ == "__main__":
    main()
