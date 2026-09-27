# ByteTrack-V2-unofficial

Unofficial implementation of **ByteTrackV2: 2D and 3D Multi-Object Tracking by Associating Every Detection Box** ([arXiv:2303.15334](https://arxiv.org/abs/2303.15334)), built on top of the ByteTrack v1 code base ([FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack)).

## What is implemented

| Component | Paper | Code |
|---|---|---|
| BYTE hierarchical association (high-score first, low-score second), shared by 2D and 3D | Sec. 3.4, Alg. 1 | `bytetrack_v2/core/byte_tracker.py` |
| 2D Kalman filter `(u, v, a, h)` + IoU association | Sec. 3.2 | `bytetrack_v2/tracker2d/` |
| 3D Kalman filter `(x, y, z, θ, l, w, h, vx, vy, vz)` in world coordinates | Sec. 3.2 | `bytetrack_v2/tracker3d/kalman_filter.py` |
| Complementary 3D motion prediction (detected-velocity backward prediction + KF forward prediction) | Sec. 3.3, Eq. 7–8 | `bytetrack_v2/tracker3d/byte_tracker.py` |
| 3D GIoU similarity with class-wise thresholds | Eq. 9, Sec. 5.1 | `bytetrack_v2/tracker3d/giou3d.py` |
| Score-adaptive measurement noise `R̂ = α(1−s)²R` | Eq. 10 | `KalmanFilter3D.update`, `KalmanFilter2D.update` |
| YOLOX detector (vendored from ByteTrack) | Sec. 3.2 | `bytetrack_v2/detectors/yolox/` |

## Environment (Docker only)

```bash
docker/build.sh                   # builds bytetrack-v2:latest (CUDA 12.1, PyTorch 2.4)
docker/run.sh                     # interactive shell, repo mounted at /workspace
docker/run.sh python -m pytest -q tests
```

Datasets live in `datasets/` (git-ignored):

```
datasets/
├── MOT17/{train,test}/MOT17-XX-{DPM,FRCNN,SDP}/
├── nuscenes/{v1.0-mini,samples,sweeps,maps}/
└── detections/                   # third-party 3D detection results (json)
```

## 2D MOT (MOT17)

Download the ByteTrack ablation detector (YOLOX-X trained on CrowdHuman + MOT17 half train):

```bash
docker/run.sh gdown -q 1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob -O pretrained/bytetrack_ablation.pth.tar
docker/run.sh python tools/track_mot.py --ckpt pretrained/bytetrack_ablation.pth.tar \
    --output outputs/mot17_val_half
```

Detections are cached in `<output>/detections/`, so tracker ablations re-run in seconds.

MOT17 validation half (FRCNN copies of the 7 training sequences):

| Setting | MOTA | IDF1 | IDs | HOTA |
|---|---|---|---|---|
| Paper (Table 1/2, BYTE) | 76.6 | 79.3 | 159 | – |
| **Ours, BYTE** | **76.5** | **79.3** | 164 | 67.9 |
| Ours, w/o BYTE (`--no-byte`) | 76.0 | 77.3 | 175 | 66.5 |
| Ours, BYTE + score-adaptive R (`--score-alpha 10`) | 76.5 | 79.2 | 164 | 68.0 |

Like ByteTrack's evaluator, per-sequence `track_buffer` / `track_thresh` overrides are applied (disable with `--no-v1-overrides`).

## 3D MOT (nuScenes)

The tracker consumes detection results in the nuScenes submission format (global frame, with velocities):

```bash
docker/run.sh python tools/track_nuscenes.py --dets datasets/detections/<detections>.json \
    --version v1.0-mini --split mini_val --modality lidar --output outputs/nusc_lidar
```

Ablations of Table 7 / 8: `--motion {kalman,velocity,integrated}`, `--no-score-update`, `--no-byte`.

Defaults follow Sec. 5.1: τ = 0.2, class-wise GIoU thresholds (bicycle −0.7, bus −0.2, car −0.1, motorcycle −0.5, pedestrian −0.7, trailer −0.4, truck −0.1), lost tracks kept for 30 frames, α = 10 (LiDAR) / 100 (camera).

### CenterPoint (LiDAR) on nuScenes mini

Official CenterPoint val predictions (`infos_val_10sweeps_withvelo_filter_True.json` from the [CenterPoint model zoo](https://github.com/tianweiy/CenterPoint/tree/master/configs/nusc)) filtered to the mini scenes:

```bash
docker/run.sh python tools/filter_nusc_results.py \
    --src datasets/centerpoint_result/infos_val_10sweeps_withvelo_filter_True.json \
    --out outputs/centerpoint/mini_in_val.json --split mini_in_val
docker/run.sh python tools/track_nuscenes.py --dets outputs/centerpoint/mini_in_val.json \
    --split mini_in_val --output outputs/nusc_cp_inval/full
```

`mini_in_val` contains the 4 `v1.0-mini` scenes that belong to the official val split (scene-0103, -0553, -0796, -0916), so they are unseen by the detector. `mini_val` is the official 2-scene subset.

| Motion / association (Table 7 / 8 rows) | mini_val AMOTA | IDS | mini_in_val AMOTA | IDS |
|---|---|---|---|---|
| Kalman | 0.771 | 18 | 0.698 | 30 |
| Detected velocity | 0.782 | 10 | 0.708 | 11 |
| Integrated | 0.776 | 12 | 0.704 | 13 |
| Integrated + score update (Eq. 10) | 0.780 | 16 | 0.700 | 17 |
| Integrated + update + BYTE (full) | 0.792 | 13 | 0.697 | 16 |

Using detected velocities consistently cuts ID switches compared with the Kalman-only model (30 → 11 IDS), matching the trend of Table 7. The remaining ±0.01 AMOTA differences are within the noise of 2–4 scenes, so the gains of Eq. 10 and BYTE reported on the full val set (+0.2 to +0.7 AMOTA) cannot be confirmed at this scale. The Kalman filter noise parameters are untuned defaults.

### PETRv2 (camera) on nuScenes mini

The paper's camera setting uses PETRv2 (VoVNet, 1600×640), which is not public. The public PETRv2-VoVNet-800×320 checkpoint (NDS 50.3 on full val) is used instead. It runs in a separate detector image that follows PETR's stack (CUDA 11.1, torch 1.9, mmcv-full 1.4.0, mmdet3d 0.17.1):

```bash
docker/petr/build.sh
SPLITS="val train" tools/detectors/petrv2_infer.sh     # -> outputs/petrv2/mini_{val,train}_detections.json
docker/run.sh python tools/filter_nusc_results.py --split mini_in_val \
    --src outputs/petrv2/mini_val_detections.json outputs/petrv2/mini_train_detections.json \
    --out outputs/petrv2/mini_in_val.json
docker/run.sh python tools/track_nuscenes.py --dets outputs/petrv2/mini_in_val.json \
    --split mini_in_val --modality camera --output outputs/nusc_petr_mini_in_val/full
```

PETRv2-800×320 on mini_val scores mAP 38.9 / NDS 42.4 / mAVE 0.571.

| Motion / association (Table 7 / 8 rows) | mini_val AMOTA | IDS | mini_in_val AMOTA | IDS |
|---|---|---|---|---|
| Kalman | 0.549 | 100 | 0.563 | 120 |
| Detected velocity | 0.570 | 57 | 0.583 | 74 |
| Integrated | 0.566 | 57 | 0.571 | 79 |
| Integrated + score update (Eq. 10, α = 100) | 0.567 | 41 | 0.572 | 61 |
| Integrated + update + BYTE (full) | **0.580** | 73 | **0.586** | 89 |

With noisier camera detections, the trends of Table 7 / 8 are clearer:
- Detected velocities cut ID switches by about 40% compared with the Kalman-only model.
- The score-adaptive update reduces ID switches further.
- BYTE raises recall and gives the best AMOTA on both splits.

A sanity check with noisy ground-truth "detections" (no detector needed):

```bash
docker/run.sh python tools/make_pseudo_detections.py --out outputs/pseudo_dets/mini_val.json
docker/run.sh python tools/track_nuscenes.py --dets outputs/pseudo_dets/mini_val.json --output outputs/nusc_pseudo
```

> Only `v1.0-mini` is used here, so the numbers are not comparable with the paper's full validation set (CenterPoint: 72.4 AMOTA).

## Interpretation choices

The paper leaves a few details open; this implementation makes the following choices:

1. **Time step in Eq. 7.** Detected velocities are in m/s, so the backward prediction is `D̂_{t−1} = D_t − V_t · Δt`, with `Δt` from sample timestamps.
2. **Eq. 9 with lost tracks.** In the first association, tracks alive at `t−1` are compared as `GIoU(D̂_{t−1}, T_{t−1})` (backward prediction), and lost tracks as `GIoU(D_t, T̂_t)` using the Kalman forward prediction, all in one Hungarian problem. The second association and unconfirmed tracks use the same rule.
3. **Eq. 10 at `s → 1`.** The factor `α(1−s)²` is floored at `1e-2` to keep `R` positive definite.
4. **Per-class association.** One tracker runs per class, so associations never cross classes.
5. **3D track initialization.** The KF velocity is initialized from the detected velocity. Unconfirmed tracks are motion-predicted, unlike 2D ByteTrack, because nuScenes frames are 0.5 s apart.
6. **Low-score floor.** Detections with a score ≤ 0.1 are discarded in both 2D (as in ByteTrack) and 3D (`--low-thresh`).

## Acknowledgements and licenses

- [ByteTrack](https://github.com/FoundationVision/ByteTrack) (MIT, © 2021 Yifu Zhang): tracker logic, evaluation protocol and the vendored detector code.
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0, © Megvii Inc.): files under `bytetrack_v2/detectors/yolox/` keep their original headers.
- [PETR](https://github.com/megvii-research/PETR) (Apache-2.0) and [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) for PETRv2 inference (not vendored; cloned into the detector image).
- [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) and [TrackEval](https://github.com/JonathonLuiten/TrackEval) for evaluation.
