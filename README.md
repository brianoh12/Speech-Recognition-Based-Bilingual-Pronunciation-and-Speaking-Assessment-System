# Spoken Pronunciation Assessment Project

This repository packages a pronunciation-assessment project codebase for portfolio review.

## What This Repository Shows
- ASR preprocessing/training/evaluation workflow
- Pronunciation scoring feature extraction pipeline
- Phoneme-level feedback generation logic
- Realtime demo notebook workflow (outputs removed)

## Project Structure
```bash
.
├── src/
│   ├── asr/                        # ASR pipeline scripts
│   │   ├── preprocess_dataset.py
│   │   ├── train_ctc_phone_recognizer.py
│   │   └── evaluate_ctc_phone_recognizer.py
│   ├── core/                       # core scoring and feedback modules
│   │   ├── pronunciation_model_ko.py
│   │   ├── extract_features_ko.py
│   │   ├── phoneme_feedback_rules_ko.py
│   │   └── asr_alignment_visualizer_ko.py
│   └── reference_baseline/         # reference baseline modules used during development
├── demo/                           # sanitized demo notebooks (no result outputs)
└── README.md
```

## Notes
- Project-specific model checkpoints and private datasets are excluded.
- Notebook outputs are removed.
- Some scripts still contain local path assumptions and should be parameterized for standalone execution.
