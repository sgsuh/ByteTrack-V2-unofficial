"""Run YOLOX + ByteTrack (2D) on MOTChallenge sequences and evaluate.

Stages: (1) detection, cached as MOT det files, (2) tracking, (3) evaluation.

Example (MOT17 val half, ByteTrack ablation model):
    python tools/track_mot.py --ckpt pretrained/bytetrack_ablation.pth.tar \
        --output outputs/mot17_val_half
"""
import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytetrack_v2.core.basetrack import BaseTrack  # noqa: E402
from bytetrack_v2.datasets.mot import list_sequences, write_results, write_split_gt  # noqa: E402
from bytetrack_v2.evaluation.mot_eval import clear_metrics, hota_metrics  # noqa: E402
from bytetrack_v2.tracker2d.byte_tracker import BYTETracker2D  # noqa: E402

# Per-sequence settings hard-coded in ByteTrack's MOT evaluator (mot_evaluator.py).
V1_TRACK_BUFFER = {"MOT17-05-FRCNN": 14, "MOT17-06-FRCNN": 14,
                   "MOT17-13-FRCNN": 25, "MOT17-14-FRCNN": 25}
V1_TRACK_THRESH = {"MOT17-01-FRCNN": 0.65, "MOT17-06-FRCNN": 0.65,
                   "MOT17-12-FRCNN": 0.7, "MOT17-14-FRCNN": 0.67,
                   "MOT20-06": 0.3, "MOT20-08": 0.3}


def make_parser():
    p = argparse.ArgumentParser("ByteTrackV2 2D MOT")
    p.add_argument("--data-root", default="datasets/MOT17")
    p.add_argument("--subset", default="train", choices=["train", "test"])
    p.add_argument("--split", default="val_half", choices=["train_half", "val_half", "full"])
    p.add_argument("--detector-filter", default="FRCNN",
                   help="keep one copy of each MOT17 sequence; empty for MOT20")
    p.add_argument("--output", default="outputs/mot17_val_half")
    # detector
    p.add_argument("--ckpt", default="pretrained/bytetrack_ablation.pth.tar")
    p.add_argument("--depth", type=float, default=1.33)
    p.add_argument("--width", type=float, default=1.25)
    p.add_argument("--tsize", type=int, nargs=2, default=[800, 1440], metavar=("H", "W"))
    p.add_argument("--conf", type=float, default=0.01)
    p.add_argument("--nms", type=float, default=0.7)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--redetect", action="store_true", help="ignore cached detections")
    # tracker
    p.add_argument("--track-thresh", type=float, default=0.6)
    p.add_argument("--match-thresh", type=float, default=0.9)
    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument("--min-box-area", type=float, default=100)
    p.add_argument("--no-fuse-score", dest="fuse_score", action="store_false",
                   help="disable score fusion in the first association (MOT20)")
    p.add_argument("--no-byte", dest="use_byte", action="store_false",
                   help="drop low-score detections (SORT-like ablation)")
    p.add_argument("--score-alpha", type=float, default=None,
                   help="enable score-adaptive KF measurement noise (Eq. 10)")
    p.add_argument("--no-v1-overrides", dest="v1_overrides", action="store_false",
                   help="disable ByteTrack's per-sequence track_buffer/track_thresh")
    p.add_argument("--no-eval", dest="eval", action="store_false")
    return p


def load_dets(path):
    """MOT det file -> {frame_id: (N, 5) [x1, y1, x2, y2, score]}."""
    per_frame = defaultdict(lambda: np.zeros((0, 5), dtype=np.float32))
    if os.path.getsize(path) == 0:
        return per_frame
    rows = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
    for f in np.unique(rows[:, 0]).astype(int):
        r = rows[rows[:, 0] == f]
        per_frame[f] = np.stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5], r[:, 6]], axis=1)
    return per_frame


def run_detection(args, seqs, det_dir):
    todo = [s for s in seqs if args.redetect or not os.path.exists(os.path.join(det_dir, f"{s.name}.txt"))]
    if not todo:
        return
    from bytetrack_v2.detectors.yolox_detector import YOLOXDetector

    detector = YOLOXDetector(args.ckpt, depth=args.depth, width=args.width, test_size=args.tsize,
                             conf_thresh=args.conf, nms_thresh=args.nms, fp16=args.fp16)
    for seq in todo:
        with open(os.path.join(det_dir, f"{seq.name}.txt"), "w") as f:
            for frame_id in tqdm(range(1, seq.num_frames + 1), desc=f"detect {seq.name}"):
                dets = detector(cv2.imread(seq.image_path(frame_id)))
                for x1, y1, x2, y2, s in dets:
                    f.write(f"{frame_id},-1,{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{s:.5f}\n")


def run_tracking(args, seq, dets):
    track_thresh, track_buffer = args.track_thresh, args.track_buffer
    if args.v1_overrides:
        track_thresh = V1_TRACK_THRESH.get(seq.name, track_thresh)
        track_buffer = V1_TRACK_BUFFER.get(seq.name, track_buffer)
    tracker = BYTETracker2D(track_thresh=track_thresh, match_thresh=args.match_thresh,
                            track_buffer=track_buffer, fuse_score=args.fuse_score,
                            score_alpha=args.score_alpha, use_byte=args.use_byte)
    results = []
    for frame_id in range(1, seq.num_frames + 1):
        frame_dets = dets[frame_id]
        if len(frame_dets) == 0:
            # ByteTrack skips the tracker update on frames without detections.
            continue
        tlwhs, ids, scores = [], [], []
        for t in tracker.update(frame_dets):
            tlwh = t.tlwh
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                tlwhs.append(tlwh)
                ids.append(t.track_id)
                scores.append(t.score)
        results.append((frame_id, tlwhs, ids, scores))
    return results


def main():
    args = make_parser().parse_args()
    det_dir = os.path.join(args.output, "detections")
    res_dir = os.path.join(args.output, "track_results")
    gt_dir = os.path.join(args.output, "gt")
    os.makedirs(det_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    logger.info(f"Args: {vars(args)}")

    seqs = list_sequences(args.data_root, args.subset, args.split, args.detector_filter)
    logger.info(f"{len(seqs)} sequences: {[s.name for s in seqs]}")

    run_detection(args, seqs, det_dir)

    for seq in seqs:
        BaseTrack.reset_id()
        results = run_tracking(args, seq, load_dets(os.path.join(det_dir, f"{seq.name}.txt")))
        write_results(os.path.join(res_dir, f"{seq.name}.txt"), results)

    if args.eval and args.subset == "train":
        for seq in seqs:
            write_split_gt(seq, gt_dir)
        _, text = clear_metrics(gt_dir, res_dir)
        logger.info("\n" + text)
        hota = hota_metrics(gt_dir, res_dir, [s.name for s in seqs])
        logger.info("HOTA {HOTA:.1f} | DetA {DetA:.1f} | AssA {AssA:.1f}".format(**hota))


if __name__ == "__main__":
    main()
