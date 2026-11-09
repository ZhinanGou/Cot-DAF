#!/usr/bin/env bash
# Train + test the final 6-granularity dual-anchor fusion on MIntRec.
set -e

# Set these to your local paths.
DATA_PATH="/path/to/data"                # root containing MIntRec/
DESC_PATH="/path/to/description"         # root containing MIntRec/compre_desc/

for s in 0 1 2 3 4; do
  echo "===== SEED $s ====="
  python run.py \
    --data_path "$DATA_PATH" \
    --desc_path "$DESC_PATH" \
    --dataset MIntRec \
    --method cesf \
    --train \
    --save_results \
    --seed "$s" \
    --gpu_id 0 \
    --video_feats_path video_feats.pkl \
    --audio_feats_path audio_feats.pkl \
    --text_backbone bert-large-uncased \
    --config_file_name cesf_mintrec \
    --results_file_name results_cesf_mintrec.csv
  echo "===== SEED $s DONE ====="
done

echo "ALL SEEDS COMPLETE"
