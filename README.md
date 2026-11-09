Dual-Anchor shallow-to-deep Fusion (DAF) for multimodal intent recognition.

In real-world multimodal intent recognition, the textual, visual, and acoustic modalities frequently convey conflicting rather than complementary information — a speaker may complain while smiling and speaking cheerfully.
This repository implements the fusion stage of a two-stage framework: given the utterance text, its video/audio features, and a pre-distilled Core Description that carries cross-modal conflict-arbitration cues, DAF produces the intent prediction.

- The  Core Description is treated as a semantic anchor, complementary to  the literal anchor formed by the original text.
- Four cross-attention operations re-express video/audio from each anchor's   viewpoint.
- Six multi-granularity representations interact at the rep level through a  Transformer encoder.
- A capacity-controlled classifier produces the intent prediction.

Note. This repository contains only the trainable fusion stage. The > upstream semantic-distillation stage that produces the Core Description from raw video/audio is not included here; its output is consumed directly as a 3-column TSV (see below).


Repository layout

```
├── run.py                 # training / evaluation entry point
├── configs/               # hyper-parameter configs (mintrec / mintrec2)
├── models/                # dual-anchor fusion (cesf.py, subnets.py)
├── utils/                 # metrics, early-stopping, io helpers
├── data/                  # preprocessing pipeline (text/video/audio/desc)
├── scripts/               # one-shot train+test scripts
└── docs/                  # methodology + results
```

Setup

```bash
pip install -r requirements.txt
```

The text backbone defaults to `bert-large-uncased` (downloaded from the Hugging Face Hub). To run offline, set the path in `configs/__init__.py` under `pretrained_models_path`.

## Input data

Training requires four inputs per sample:

1. **Utterance text** — `train.tsv` / `dev.tsv` / `test.tsv` under    `DATA_PATH/<Dataset>/`.
2. **Video features** — a `.pkl` dictionary (`video_feats.pkl` for MIntRec,   `swin_roi.pkl` for MIntRec2.0).
3. **Audio features** — a `.pkl` dictionary (`audio_feats.pkl` for MIntRec,   `wavlm_feats.pkl` for MIntRec2.0).
4. **Core Description** — a 3-column TSV under   `DESC_PATH/<Dataset>/compre_desc/`:

   | Column | Field | Meaning |
   |--------|-------|---------|
   | 1 | `Dialogue_id` | dialogue index (ignored by the model) |
   | 2 | `Utterance_id` | utterance index (ignored by the model) |
   | 3 | `Core_Description` | comprehensive semantics (the semantic anchor) |

   The default file name is `<split>.core.tsv`   (`train.core.tsv`, `dev.core.tsv`, `test.core.tsv`); override with
   `--desc_suffix`, e.g. `--desc_suffix ".core.qwen.tsv"`. Only the third   column is read — no intermediate reasoning output is required.

## Train and evaluate

```bash
# edit DATA_PATH / DESC_PATH in the script, then:
bash scripts/run_mintrec2.sh
bash scripts/run_mintrec.sh
```

Or directly:

```bash
python run.py \
  --data_path /path/to/data \
  --desc_path /path/to/description \
  --dataset MIntRec2.0 \
  --method cesf \
  --train --save_results \
  --seed 0 --gpu_id 0 \
  --video_feats_path swin_roi.pkl \
  --audio_feats_path wavlm_feats.pkl \
  --text_backbone bert-large-uncased \
  --config_file_name cesf_mintrec2 \
  --results_file_name results_cesf_mintrec2.csv
```

`--dataset` is `MIntRec` (video `video_feats.pkl`, audio `audio_feats.pkl`) or`MIntRec2.0` (video `swin_roi.pkl`, audio `wavlm_feats.pkl`).

## Citation

```bibtex
@misc{daf,
  title   = {Dual-Anchor Shallow-to-Deep Fusion for Multimodal Intent Recognition},
  author  = {},
  year    = {2026},
}
```

## License

[MIT](LICENSE)
