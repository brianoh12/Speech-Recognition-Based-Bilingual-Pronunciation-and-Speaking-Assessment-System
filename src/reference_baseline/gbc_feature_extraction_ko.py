import librosa
import numpy as np
import torch
import torch.nn as nn
from Levenshtein import ratio
import Levenshtein as Lev
from itertools import zip_longest
from difflib import SequenceMatcher

# 한국어 음소 세트 정의
KOR_VOWELS = {
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ",
    "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"
}

KOR_CONSONANTS = {
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ",
    "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
}

# 모음 여부 확인
def is_vowel(phoneme):
    return phoneme in KOR_VOWELS

# 자음 여부 확인
def is_consonant(phoneme):
    return phoneme in KOR_CONSONANTS

def phoneme_similarity_score(predicted, reference):
    return SequenceMatcher(None, predicted, reference).ratio()


def extract_accuracy_features(audio_path, correct_phoneme, processor, model_phoneme, device):
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        logits = model_phoneme(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_phonemes = processor.batch_decode(predicted_ids.cpu())[0].lower().split()
    
    if not isinstance(correct_phoneme, str):
        print(f"Warning: correct_phoneme is not a string. Received: {correct_phoneme}")
        return [0.0, 0.0, 0.0]
    
    correct_phonemes = correct_phoneme.lower().split()

    # 전체 유사도 기반 점수
    sim = SequenceMatcher(None, predicted_phonemes, correct_phonemes).ratio()

    # 자음, 모음 분리 후 각각 유사도 계산
    pred_consonants = [p for p in predicted_phonemes if is_consonant(p)]
    pred_vowels = [p for p in predicted_phonemes if is_vowel(p)]
    corr_consonants = [p for p in correct_phonemes if is_consonant(p)]
    corr_vowels = [p for p in correct_phonemes if is_vowel(p)]

    pcc = SequenceMatcher(None, pred_consonants, corr_consonants).ratio()
    pcv = SequenceMatcher(None, pred_vowels, corr_vowels).ratio()
    pct = SequenceMatcher(None, predicted_phonemes, correct_phonemes).ratio()

    return [pcc, pcv, pct]


##### function to extract fluency features #####
def estimate_syllable_count(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    peaks = np.where((rms[1:-1] > rms[:-2]) & (rms[1:-1] > rms[2:]))[0]
    return len(peaks)

def calculate_voice_breaks(y, sr):
    intervals = librosa.effects.split(y, top_db=30)
    num_breaks = len(intervals)
    silent_frames = sum((end - start) for start, end in intervals)
    total_frames = len(y)
    ratio_pct = (silent_frames / total_frames) * 100 if total_frames > 0 else 0
    return num_breaks, ratio_pct

def extract_fluency_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    duration_s = librosa.get_duration(y=y, sr=sr)
    syllable_count = estimate_syllable_count(audio_path)
    speaking_rate = syllable_count / duration_s if duration_s > 0 else 0

    intervals = librosa.effects.split(y, top_db=30)
    pause_dur = sum((end - start) / sr for start, end in intervals)
    voiced_dur = max(duration_s - pause_dur, 1e-6)
    articulation_rate = syllable_count / voiced_dur

    num_breaks, voice_breaks_ratio = calculate_voice_breaks(y, sr)

    return [
        speaking_rate,
        voice_breaks_ratio,
        num_breaks
    ]


##### function to extract prosody features #####
def extract_prosody_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    
    # F0 (pitch) mean
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=400)
    f0_mean = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0
    
    # Intensity (energy) mean
    rms = librosa.feature.rms(y=y)[0]
    intensity_mean = np.mean(rms)
    
    # Duration (초)
    duration_s = librosa.get_duration(y=y, sr=sr)
    
    return [
        f0_mean,
        intensity_mean,
        duration_s
    ]

