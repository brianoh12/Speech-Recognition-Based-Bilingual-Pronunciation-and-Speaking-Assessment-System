# Speech Recognition-Based Bilingual Pronunciation and Speaking Assessment System

This repository contains the implementation of a speech assessment project that combines ASR modeling and pronunciation scoring for bilingual speaking evaluation.

## Project Scope
- End-to-end ASR data preprocessing, model training, and evaluation scripts
- English pronunciation assessment modules for Korean learners
- Phoneme-level feedback generation and ASR alignment visualization
- Realtime notebook demo pipeline for pronunciation assessment workflows

## Project Structure
```bash
.
├── src/
│   ├── asr/
│   │   ├── preprocess_dataset.py
│   │   ├── train_ctc_phone_recognizer.py
│   │   └── evaluate_ctc_phone_recognizer.py
│   ├── english_pronunciation_assessment/
│   │   ├── feature_extraction.py
│   │   ├── pronunciation_model.py
│   │   ├── phoneme_feedback.py
│   │   ├── asr_alignment_visualization.py
│   │   └── requirements.txt
│   ├── core/
│   └── reference_baseline/
├── demo/
│   ├── realtime_pronunciation_assessment.ipynb
│   ├── realtime_pronunciation_demo_ko.ipynb
│   └── gbc_realtime_demo_ko.ipynb
└── README.md
```

## Focus Areas Shown in Code
- Acoustic and linguistic feature engineering for pronunciation scoring
- Multi-score speaking assessment model components (accuracy, fluency, prosody, completeness)
- Error pattern analysis and feedback rule design at phoneme level
- ASR alignment analysis for interpretable learner feedback
