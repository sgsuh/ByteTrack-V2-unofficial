"""Run ByteTrackV2 (3D) on nuScenes detections and evaluate AMOTA.

Example (CenterPoint detections, LiDAR setting of the paper):
    python tools/track_nuscenes.py --dets <centerpoint_val.json> \
        --version v1.0-mini --split mini_val --output outputs/nusc_centerpoint
"""
import argparse
import json
import os
import sys

from loguru import logger
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytetrack_v2.core.basetrack import BaseTrack  # noqa: E402
from bytetrack_v2.datasets.nuscenes import (CUSTOM_SPLITS, GIOU_THRESHOLDS,  # noqa: E402
                                            TRACKING_CLASSES, load_detections, patch_eval_split,
                                            resolve_split, scene_samples, track_to_nusc)
from bytetrack_v2.tracker3d.byte_tracker import MOTION_MODES, MultiClassTracker3D  # noqa: E402

PRESETS = {
    # Sec. 5.1: tau = 0.2 for CenterPoint / PETRv2, alpha = 10 (LiDAR) / 100 (camera).
    "lidar": dict(track_thresh=0.2, score_alpha=10.0),
    "camera": dict(track_thresh=0.2, score_alpha=100.0),
}


def make_parser():
    p = argparse.ArgumentParser("ByteTrackV2 3D MOT on nuScenes")
    p.add_argument("--dets", required=True, help="nuScenes detection results json")
    p.add_argument("--dataroot", default="datasets/nuscenes")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--split", default="mini_val",
                   help=f"official split or one of {sorted(CUSTOM_SPLITS)}")
    p.add_argument("--output", default="outputs/nusc")
    p.add_argument("--modality", default="lidar", choices=list(PRESETS))
    p.add_argument("--track-thresh", type=float, default=None, help="tau (overrides preset)")
    p.add_argument("--low-thresh", type=float, default=0.1)
    p.add_argument("--giou-thresh", type=float, default=None,
                   help="single GIoU threshold for all classes (default: class-wise, Sec. 5.1)")
    p.add_argument("--max-time-lost", type=int, default=30)
    p.add_argument("--motion", default="integrated", choices=MOTION_MODES)
    p.add_argument("--score-alpha", type=float, default=None, help="alpha (overrides preset)")
    p.add_argument("--no-score-update", dest="score_update", action="store_false",
                   help="disable score-adaptive measurement noise (Eq. 10)")
    p.add_argument("--no-byte", dest="use_byte", action="store_false")
    p.add_argument("--no-eval", dest="eval", action="store_false")
    return p


def build_tracker(args):
    preset = PRESETS[args.modality]
    track_thresh = preset["track_thresh"] if args.track_thresh is None else args.track_thresh
    score_alpha = preset["score_alpha"] if args.score_alpha is None else args.score_alpha
    return MultiClassTracker3D(
        TRACKING_CLASSES,
        giou_thresh=GIOU_THRESHOLDS if args.giou_thresh is None else args.giou_thresh,
        track_thresh=track_thresh,
        low_thresh=args.low_thresh,
        max_time_lost=args.max_time_lost,
        motion=args.motion,
        score_alpha=score_alpha if args.score_update else None,
        use_byte=args.use_byte,
    )


def run(args, nusc, scene_names):
    scenes = list(scene_samples(nusc, scene_names))
    tokens = {tok for _, samples in scenes for tok, _ in samples}
    detections = load_detections(args.dets, sample_tokens=tokens)
    missing = tokens - detections.keys()
    if missing:
        logger.warning(f"{len(missing)} / {len(tokens)} samples have no detections")

    BaseTrack.reset_id()
    results = {}
    for _, samples in tqdm(scenes, desc="scenes"):
        tracker = build_tracker(args)
        prev_ts = None
        for token, ts in samples:
            dt = 0.5 if prev_ts is None else ts - prev_ts
            prev_ts = ts
            tracks = tracker.update(detections.get(token, {}), dt)
            results[token] = [track_to_nusc(t, token, c) for c, ts_ in tracks.items() for t in ts_]
    return results


def main():
    args = make_parser().parse_args()
    os.makedirs(args.output, exist_ok=True)
    logger.info(f"Args: {vars(args)}")

    from nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    eval_set, scene_names = resolve_split(args.split)
    results = run(args, nusc, scene_names)
    result_path = os.path.join(args.output, "tracking_results.json")
    meta = {"use_camera": args.modality == "camera", "use_lidar": args.modality == "lidar",
            "use_radar": False, "use_map": False, "use_external": False}
    with open(result_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f)
    logger.info(f"Saved {sum(len(v) for v in results.values())} boxes to {result_path}")

    if args.eval:
        from nuscenes.eval.common.config import config_factory
        from nuscenes.eval.tracking.evaluate import TrackingEval

        if args.split in CUSTOM_SPLITS:
            patch_eval_split(eval_set, scene_names)
        evaluator = TrackingEval(config_factory("tracking_nips_2019"), result_path, eval_set,
                                 os.path.join(args.output, "eval"), args.version, args.dataroot,
                                 verbose=False)
        metrics = evaluator.main()
        logger.info("AMOTA {amota:.3f} | AMOTP {amotp:.3f} | MOTA {mota:.3f} | IDS {ids:d} | "
                    "FRAG {frag:d} | RECALL {recall:.3f}".format(
                        amota=metrics["amota"], amotp=metrics["amotp"], mota=metrics["mota"],
                        ids=int(metrics["ids"]), frag=int(metrics["frag"]), recall=metrics["recall"]))
        for c in TRACKING_CLASSES:
            logger.info(f"  {c:<11s} AMOTA {metrics['label_metrics']['amota'][c]:.3f}")


if __name__ == "__main__":
    main()
