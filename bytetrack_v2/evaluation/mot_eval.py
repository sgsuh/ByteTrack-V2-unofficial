"""MOTChallenge evaluation: CLEAR/ID metrics via motmetrics and HOTA via TrackEval."""
import glob
import os
from collections import OrderedDict

import motmetrics as mm
import numpy as np


def clear_metrics(gt_dir, results_dir):
    """MOTA / IDF1 / IDs etc. following ByteTrack's `tools/track.py` protocol."""
    mm.lap.default_solver = "lap"
    gtfiles = sorted(glob.glob(os.path.join(gt_dir, "*", "gt", "gt.txt")))
    gt = OrderedDict((f.split(os.sep)[-3], mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1))
                     for f in gtfiles)
    ts = OrderedDict((os.path.splitext(os.path.basename(f))[0],
                      mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=-1))
                     for f in sorted(glob.glob(os.path.join(results_dir, "*.txt"))))
    accs, names = [], []
    for name, ts_df in ts.items():
        if name in gt:
            accs.append(mm.utils.compare_to_groundtruth(gt[name], ts_df, "iou", distth=0.5))
            names.append(name)
    mh = mm.metrics.create()
    metrics = mm.metrics.motchallenge_metrics + ["num_objects"]
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    text = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    return summary, text


def hota_metrics(gt_dir, results_dir, seqs):
    """HOTA / DetA / AssA with TrackEval (MOTChallenge 2D box, pedestrian class)."""
    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False,
                        "TIME_PROGRESS": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
                        "PLOT_CURVES": False})
    tracker_root = os.path.dirname(os.path.abspath(results_dir))
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": gt_dir,
        "TRACKERS_FOLDER": tracker_root,
        "TRACKERS_TO_EVAL": [""],
        "TRACKER_SUB_FOLDER": os.path.basename(os.path.abspath(results_dir)),
        "OUTPUT_FOLDER": os.path.join(tracker_root, "trackeval"),
        "SEQ_INFO": {s: None for s in seqs},
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
        "SKIP_SPLIT_FOL": True,
        "PRINT_CONFIG": False,
    })
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    res, _ = evaluator.evaluate([dataset], [trackeval.metrics.HOTA({"PRINT_CONFIG": False})])
    hota = res["MotChallenge2DBox"][""]["COMBINED_SEQ"]["pedestrian"]["HOTA"]
    return {k: float(np.mean(hota[k])) * 100 for k in ("HOTA", "DetA", "AssA")}
