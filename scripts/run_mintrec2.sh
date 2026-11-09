#!/usr/bin/env bash
# Train + test the final 6-granularity dual-anchor fusion on MIntRec2.0.
set -e

# Set these to your local paths.
DATA_PATH="/path/to/data"                # root containing MIntRec2.0/
DESC_PATH="/path/to/description"         # root containing MIntRec2.0/compre_desc/

for s in 0 1 2 3 4; do
  echo "===== SEED $s ====="
  python run.py \
    --data_path "$DATA_PATH" \
    --desc_path "$DESC_PATH" \
    --dataset MIntRec2.0 \
    --method cesf \
    --train \
    --save_results \
    --seed "$s" \
    --gpu_id 0 \
    --video_feats_path swin_roi.pkl \
    --audio_feats_path wavlm_feats.pkl \
    --text_backbone bert-large-uncased \
    --config_file_name cesf_mintrec2 \
    --results_file_name results_cesf_mintrec2.csv
  echo "===== SEED $s DONE ====="
done

echo "ALL SEEDS COMPLETE"
