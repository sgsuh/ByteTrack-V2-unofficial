"""MOTChallenge sequence utilities (MOT17 / MOT20 layout).

The half split follows ByteTrack / CenterTrack: for a sequence with n frames,
`train_half` covers frames [1, n // 2 + 1] and `val_half` covers frames
[n // 2 + 2, n], renumbered to start at 1.
"""
import configparser
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class MOTSequence:
    name: str
    root: str          # directory containing img1/, gt/, seqinfo.ini
    frame_rate: int
    width: int
    height: int
    first_frame: int   # 1-based original frame index of output frame 1
    num_frames: int

    def image_path(self, frame_id):
        """Path of the image for output frame `frame_id` (1-based)."""
        return os.path.join(self.root, "img1", f"{self.first_frame + frame_id - 1:06d}.jpg")


def read_seqinfo(seq_root):
    parser = configparser.ConfigParser()
    parser.read(os.path.join(seq_root, "seqinfo.ini"))
    info = parser["Sequence"]
    return {
        "frame_rate": int(info["frameRate"]),
        "length": int(info["seqLength"]),
        "width": int(info["imWidth"]),
        "height": int(info["imHeight"]),
    }


def half_range(num_images, split):
    """0-based inclusive image index range of a split (ByteTrack convention)."""
    if split == "train_half":
        return 0, num_images // 2
    if split == "val_half":
        return num_images // 2 + 1, num_images - 1
    return 0, num_images - 1


def list_sequences(data_root, subset="train", split="val_half", detector_filter="FRCNN"):
    """List MOT sequences of `data_root/subset`, restricted to one public detector copy."""
    subset_dir = os.path.join(data_root, subset)
    seqs = []
    for name in sorted(os.listdir(subset_dir)):
        if detector_filter and detector_filter not in name:
            continue
        root = os.path.join(subset_dir, name)
        info = read_seqinfo(root)
        num_images = len([f for f in os.listdir(os.path.join(root, "img1")) if f.endswith(".jpg")])
        start, end = half_range(num_images, split)
        seqs.append(MOTSequence(name, root, info["frame_rate"], info["width"], info["height"],
                                first_frame=start + 1, num_frames=end - start + 1))
    return seqs


def write_split_gt(seq, out_dir):
    """Write the split GT as `out_dir/<seq>/gt/gt.txt` (+ seqinfo.ini) in MOTChallenge layout."""
    anns = np.loadtxt(os.path.join(seq.root, "gt", "gt.txt"), dtype=np.float64, delimiter=",")
    keep = (anns[:, 0] >= seq.first_frame) & (anns[:, 0] < seq.first_frame + seq.num_frames)
    anns = anns[keep]
    anns[:, 0] -= seq.first_frame - 1

    seq_dir = os.path.join(out_dir, seq.name)
    os.makedirs(os.path.join(seq_dir, "gt"), exist_ok=True)
    with open(os.path.join(seq_dir, "gt", "gt.txt"), "w") as f:
        for o in anns:
            f.write("{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n".format(
                *(int(v) for v in o[:8]), o[8]))
    with open(os.path.join(seq_dir, "seqinfo.ini"), "w") as f:
        f.write("[Sequence]\n"
                f"name={seq.name}\nimDir=img1\nframeRate={seq.frame_rate}\n"
                f"seqLength={seq.num_frames}\nimWidth={seq.width}\nimHeight={seq.height}\n"
                "imExt=.jpg\n")


def write_results(filename, results):
    """Write MOTChallenge results: frame, id, x1, y1, w, h, score, -1, -1, -1."""
    with open(filename, "w") as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for (x1, y1, w, h), tid, s in zip(tlwhs, track_ids, scores):
                f.write(f"{frame_id},{tid},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{s:.2f},-1,-1,-1\n")
