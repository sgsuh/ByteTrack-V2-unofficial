"""Keep only the samples of one nuScenes split in a detection results json.

Example:
    python tools/filter_nusc_results.py \
        --src datasets/centerpoint_result/infos_val_10sweeps_withvelo_filter_True.json \
        --out outputs/centerpoint/mini_val.json --version v1.0-mini --split mini_val
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytetrack_v2.datasets.nuscenes import resolve_split  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, nargs="+", help="one or more detection jsons to merge")
    p.add_argument("--out", required=True)
    p.add_argument("--dataroot", default="datasets/nuscenes")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--split", default="mini_val")
    args = p.parse_args()

    from nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    scenes = set(resolve_split(args.split)[1])
    tokens = {s["token"] for s in nusc.sample if nusc.get("scene", s["scene_token"])["name"] in scenes}

    results, meta = {}, {}
    for src in args.src:
        with open(src) as f:
            data = json.load(f)
        meta = data.get("meta", meta)
        results.update({t: dets for t, dets in data["results"].items() if t in tokens})
    missing = tokens - results.keys()
    if missing:
        raise SystemExit(f"{len(missing)} / {len(tokens)} samples of {args.split} are missing in {args.src}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "results": results}, f)
    counts = Counter(d["detection_name"] for dets in results.values() for d in dets)
    print(f"meta: {meta}")
    print(f"kept {len(results)} samples, {sum(counts.values())} boxes -> {args.out}")
    print(dict(counts))


if __name__ == "__main__":
    main()
