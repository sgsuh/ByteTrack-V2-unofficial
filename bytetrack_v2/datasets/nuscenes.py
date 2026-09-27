"""nuScenes helpers: detection loading, scene iteration and result formatting.

nuScenes boxes use `translation` (center), `size` = [w, l, h], a `rotation`
quaternion [w, x, y, z] and `velocity` [vx, vy], all in the global frame.
Internally boxes are [x, y, z, yaw, l, w, h].
"""
import json
from collections import defaultdict

import numpy as np
from pyquaternion import Quaternion

TRACKING_CLASSES = ("bicycle", "bus", "car", "motorcycle", "pedestrian", "trailer", "truck")

# Class-wise 3D GIoU matching thresholds (ByteTrackV2, Sec. 5.1).
GIOU_THRESHOLDS = {
    "bicycle": -0.7, "bus": -0.2, "car": -0.1, "motorcycle": -0.5,
    "pedestrian": -0.7, "trailer": -0.4, "truck": -0.1,
}


def nusc_box_to_array(det):
    """nuScenes box dict -> ((7,) [x, y, z, yaw, l, w, h], (2,) velocity)."""
    x, y, z = det["translation"]
    w, l, h = det["size"]
    yaw = Quaternion(det["rotation"]).yaw_pitch_roll[0]
    vel = det.get("velocity", [0.0, 0.0])[:2]
    vel = np.nan_to_num(np.asarray(vel, dtype=np.float64))
    return np.array([x, y, z, yaw, l, w, h], dtype=np.float64), vel


def array_to_nusc_box(box):
    """(7,) [x, y, z, yaw, l, w, h] -> translation, size, rotation (nuScenes format)."""
    x, y, z, yaw, l, w, h = (float(v) for v in box)
    q = Quaternion(axis=[0, 0, 1], radians=yaw)
    return [x, y, z], [w, l, h], [q.w, q.x, q.y, q.z]


def load_detections(path, classes=TRACKING_CLASSES, sample_tokens=None):
    """Load a nuScenes detection json into per-sample, per-class arrays.

    Returns:
        {sample_token: {class: dict(boxes, scores, velocities, attrs)}}
    """
    with open(path) as f:
        results = json.load(f)["results"]
    out = {}
    for token, dets in results.items():
        if sample_tokens is not None and token not in sample_tokens:
            continue
        per_class = defaultdict(lambda: dict(boxes=[], scores=[], velocities=[], attrs=[]))
        for det in dets:
            name = det["detection_name"]
            if name not in classes:
                continue
            box, vel = nusc_box_to_array(det)
            d = per_class[name]
            d["boxes"].append(box)
            d["scores"].append(det["detection_score"])
            d["velocities"].append(vel)
            d["attrs"].append({"attribute_name": det.get("attribute_name", "")})
        out[token] = {
            c: dict(boxes=np.asarray(d["boxes"]).reshape(-1, 7), scores=np.asarray(d["scores"]),
                    velocities=np.asarray(d["velocities"]).reshape(-1, 2), attrs=d["attrs"])
            for c, d in per_class.items()
        }
    return out


def scene_samples(nusc, scene_names):
    """Yield (scene_name, [(sample_token, timestamp_seconds), ...]) in temporal order."""
    by_name = {s["name"]: s for s in nusc.scene}
    for name in scene_names:
        scene = by_name[name]
        samples, token = [], scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            samples.append((token, sample["timestamp"] * 1e-6))
            token = sample["next"]
        yield name, samples


def track_to_nusc(track, sample_token, class_name):
    """Format an STrack3D as a nuScenes tracking result."""
    translation, size, rotation = array_to_nusc_box(track.box)
    vx, vy = track.velocity[:2]
    return {
        "sample_token": sample_token,
        "translation": translation,
        "size": size,
        "rotation": rotation,
        "velocity": [float(vx), float(vy)],
        "tracking_id": str(track.track_id),
        "tracking_name": class_name,
        "tracking_score": float(track.score),
    }
