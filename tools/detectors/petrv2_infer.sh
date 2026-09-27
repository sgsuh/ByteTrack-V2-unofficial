#!/usr/bin/env bash
# PETRv2 (VoVNet-99, 800x320) inference on nuScenes v1.0-mini.
#
# SPLITS selects the info splits to run (default "val" = mini_val). "train"
# (= mini_train) is run with --format-only, since most of its scenes overlap
# PETRv2's training data; it is only needed for the two official-val scenes
# inside mini_train (see the `mini_in_val` split of tools/track_nuscenes.py).
#
# Run from the host (wraps itself in the petr container):
#   docker/petr/build.sh              # once
#   SPLITS="val train" tools/detectors/petrv2_infer.sh
# or directly inside the container (repo mounted at /workspace):
#   docker/petr/run.sh tools/detectors/petrv2_infer.sh
#
# Outputs (all under outputs/petrv2/, nothing is written into datasets/):
#   data/nuscenes/                                symlink farm + generated info pkls
#   petrv2_mini_<split>.py                        derived test config
#   eval_<split>/pts_bbox/results_nusc.json       raw mmdet3d format_results output
#   eval_val/pts_bbox/metrics_summary.json        nuScenes detection metrics (mini_val)
#   mini_<split>_detections.json                  detections (nuScenes submission format, global frame)
set -euo pipefail

if [ ! -d /opt/PETR ]; then
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    exec "${REPO_DIR}/docker/petr/run.sh" env SPLITS="${SPLITS:-val}" WORKERS="${WORKERS:-2}" \
        tools/detectors/petrv2_infer.sh "$@"
fi

WS=/workspace
OUT="${WS}/outputs/petrv2"
DATA="${OUT}/data/nuscenes"
CKPT="${WS}/pretrained/petrv2/petrv2_vovnet_gridmask_p4_800x320.pth"
CKPT_GDRIVE_ID=1tv_D8Ahp9tz5n4pFp4a64k-IrUZPu5Im
PETR_ROOT="${PETR_ROOT:-/opt/PETR}"
WORKERS="${WORKERS:-2}"
SPLITS="${SPLITS:-val}"

# 1) Data root: symlinks to the raw nuScenes mini data (infos are generated next to them).
mkdir -p "${DATA}"
for d in samples sweeps maps v1.0-mini; do
    [ -e "${DATA}/${d}" ] || ln -s "../../../../datasets/nuscenes/${d}" "${DATA}/${d}"
done

# 2) Info pkls: mmdet3d base infos + PETR temporal sweeps.
for split in ${SPLITS}; do
    if [ ! -f "${DATA}/mmdet3d_nuscenes_30f_infos_${split}.pkl" ]; then
        python "${WS}/tools/detectors/petrv2_prepare_infos.py" \
            --root "${DATA}" --version v1.0-mini --splits "${split}"
    fi
done

# 3) Checkpoint.
if [ ! -f "${CKPT}" ]; then
    mkdir -p "$(dirname "${CKPT}")"
    gdown -q "${CKPT_GDRIVE_ID}" -O "${CKPT}"
fi

# 4) Per split: derived config (official petrv2 800x320 config pointed at the
#    mini infos), then test. The eval set follows the info metadata (v1.0-mini -> mini_*).
cd "${PETR_ROOT}"
for split in ${SPLITS}; do
    CFG="${OUT}/petrv2_mini_${split}.py"
    cat > "${CFG}" <<CONFIG
_base_ = ['${PETR_ROOT}/projects/configs/petrv2/petrv2_vovnet_gridmask_p4_800x320.py']
data_root = '${DATA}/'
ann_file = data_root + 'mmdet3d_nuscenes_30f_infos_${split}.pkl'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=${WORKERS},
    val=dict(data_root=data_root, ann_file=ann_file),
    test=dict(data_root=data_root, ann_file=ann_file))
CONFIG

    if [ "${split}" = "val" ]; then
        MODE=(--eval bbox)
    else
        MODE=(--format-only)
    fi
    rm -rf "${OUT}/eval_${split}"
    PYTHONPATH="${PETR_ROOT}:${PYTHONPATH:-}" python tools/test.py "${CFG}" "${CKPT}" "${MODE[@]}" \
        --eval-options "jsonfile_prefix=${OUT}/eval_${split}" 2>&1 | tee "${OUT}/test_${split}.log"

    cp "${OUT}/eval_${split}/pts_bbox/results_nusc.json" "${OUT}/mini_${split}_detections.json"
    echo "Wrote ${OUT}/mini_${split}_detections.json"
done
