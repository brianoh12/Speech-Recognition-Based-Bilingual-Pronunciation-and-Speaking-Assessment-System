# Speech Recognition-Based Bilingual Pronunciation and Speaking Assessment System

This repository contains two independent pronunciation-assessment projects:
- English speech assessment for Korean learners of English
- Korean speech assessment for foreign learners of Korean

Both projects implement the same evaluation framework with four scoring dimensions:
`Accuracy`, `Fluency`, `Prosody`, and `Completeness`, plus phoneme-level feedback and alignment-based analysis.

## Project Structure
```bash
.
├── projects/
│   ├── english_project/
│   │   ├── src/
│   │   │   ├── feature_extraction.py
│   │   │   ├── pronunciation_model.py
│   │   │   ├── phoneme_feedback.py
│   │   │   ├── asr_alignment_visualization.py
│   │   │   └── requirements.txt
│   │   └── demo/
│   │       └── realtime_evaluate.ipynb
│   └── korean_project/
│       ├── src/
│       │   ├── feature_extraction.py
│       │   ├── pronunciation_model.py
│       │   ├── phoneme_feedback.py
│       │   ├── asr_alignment_visualization.py
│       │   └── requirements.txt
│       └── demo/
│           └── realtime_evaluate_ko.ipynb
├── src/
│   └── asr/
│       ├── preprocess_dataset.py
│       ├── train_ctc_phone_recognizer.py
│       └── evaluate_ctc_phone_recognizer.py
└── demo/
    ├── realtime_pronunciation_demo_en.ipynb
    └── realtime_pronunciation_demo_ko.ipynb
```

## English Project (Korean Learners' English Speech)
- Phone recognizer + ASR outputs are used to compute pronunciation-related metrics.
- `pronunciation_model.py` predicts fluency and prosody from acoustic/prosodic features.
- `feature_extraction.py` computes pronunciation features including consonant/vowel correctness and rhythm/pitch features.
- `phoneme_feedback.py` generates rule-based feedback from phoneme-level error patterns.
- `asr_alignment_visualization.py` provides alignment/spectrogram-centered analysis for interpretability.

## Korean Project (Foreign Learners' Korean Speech)
- Built with the same scoring philosophy for Korean pronunciation and speaking assessment.
- `feature_extraction.py` extracts Korean-phoneme-based and rhythm/pitch-related features.
- `pronunciation_model.py` predicts fluency and prosody scores via multitask-style heads.
- `phoneme_feedback.py` provides bilingual (KR/EN) feedback messages for common phoneme-level errors.
- `asr_alignment_visualization.py` supports token/word alignment and segment-level inspection.

## Implementation Focus
- End-to-end pronunciation assessment workflow design
- Multidimensional score calculation (`Accuracy`, `Fluency`, `Prosody`, `Completeness`)
- Error-type taxonomy and actionable feedback logic
- ASR alignment-based interpretation layer for assessment transparency
