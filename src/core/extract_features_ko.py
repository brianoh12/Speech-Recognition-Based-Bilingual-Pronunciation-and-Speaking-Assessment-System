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

"""

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
    total_phonemes = len(correct_phonemes)
    total_consonants = sum(1 for p in correct_phonemes if is_consonant(p))
    total_vowels = sum(1 for p in correct_phonemes if is_vowel(p))

    correct_consonants_count = 0
    correct_vowels_count = 0
    correct_total_count = 0
    err_penalty=0.7
    
    for p, c in zip_longest(predicted_phonemes, correct_phonemes):
        
        if p == c:
            score = 1.0  # 정확히 일치
        else:
            score = 0.0  # 완전히 틀린 경우
        
        if is_consonant(c):
            correct_consonants_count += score
        elif is_vowel(c):
            correct_vowels_count += score
        
        correct_total_count += score
    
    pcc = (correct_consonants_count / total_consonants) if total_consonants > 0 else 0
    pcv = (correct_vowels_count / total_vowels) if total_vowels > 0 else 0
    pct = (correct_total_count / total_phonemes) if total_phonemes > 0 else 0
    
    return [pcc, pcv, pct]

"""

##### function to extract fluency features #####
# 음절 수를 자동으로 추정하는 함수 (초당 7개 음절 보정 적용, 피크 감지 강화)
def estimate_syllable_count(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    
    # STFT 기반 RMS 에너지 분석
    D = librosa.stft(y)
    spectrogram = np.abs(D)
    rms = librosa.feature.rms(S=spectrogram)[0]
    
    # 음절 경계 감지 (더 많은 피크를 감지하도록 조정)
    peaks = librosa.util.peak_pick(rms, pre_max=1, post_max=1, pre_avg=2, post_avg=2, delta=0.001, wait=1)
    syllable_count = len(peaks)
    
    # 초당 7개 음절 기준으로 보정
    duration = librosa.get_duration(y=y, sr=sr)
    expected_syllables = max(1, int(duration * 7))  # 초당 7개 음절 가정
    syllable_count = min(syllable_count, expected_syllables)
    
    return syllable_count

# Voice Breaks (무음 구간) 계산 함수
def calculate_voice_breaks(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    
    # 무음 구간 탐지 (20ms 이상의 무음만 카운트)
    silent_intervals = librosa.effects.split(y, top_db=30)  # 음성이 아닌 구간 탐지
    
    # 전체 무음 프레임 개수 계산
    silent_frames = sum((end - start) for start, end in silent_intervals)
    total_frames = len(y)
    voice_breaks_ratio = (silent_frames / total_frames) * 100 if total_frames > 0 else 0
    
    return len(silent_intervals), voice_breaks_ratio

# Fluency 분석 함수
def extract_fluency_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    intervals = librosa.effects.split(y, top_db=20)
    voiced_duration = sum(i[1] - i[0] for i in intervals) / sr
    total_duration = len(y) / sr
    
    speaking_rate = estimate_syllable_count(audio_path) / total_duration
    articulation_rate = speaking_rate * (1 - (calculate_voice_breaks(audio_path)[1] / 100))
    
    voice_breaks = len(intervals) - 1
    voice_breaks_duration = total_duration - voiced_duration
    voice_breaks_ratio = voice_breaks_duration / total_duration
    
    return [speaking_rate, articulation_rate, voice_breaks, voice_breaks_ratio]


##### function to extract prosody features #####

def extract_prosody_features(audio_path):

    y, sr = librosa.load(audio_path, sr=None)

    # 무음 제거
    non_silent_intervals = librosa.effects.split(y, top_db=20)
    y_voiced = np.concatenate([y[start:end] for start, end in non_silent_intervals])

    # 음성이 너무 짧으면 분석 제외 (최소 길이 0.1초)
    if len(y_voiced) / sr < 0.1:
        return None  

    # RMS 에너지 계산 (보간 적용)
    energy = librosa.feature.rms(y=y)[0]
    energy = np.interp(np.arange(len(y)), np.linspace(0, len(y), num=len(energy)), energy)

    # 무음 판별
    threshold = np.mean(energy) * 0.6
    voiced_segments = energy > threshold

    # F0 (기본 주파수) 계산
    f0, _, _ = librosa.pyin(y_voiced, fmin=50, fmax=350, sr=sr)

    # F0 검출 실패한 경우 제외
    if f0 is None or np.isnan(f0).sum() / len(f0) > 0.9:
        return None

    f0 = np.nan_to_num(f0, nan=0)
    f0_filtered = f0[f0 > 0]

    f0_mean = np.mean(f0_filtered) if len(f0_filtered) > 0 else 0
    f0_std = np.std(f0_filtered) if len(f0_filtered) > 0 else 0
    f0_min = np.min(f0_filtered) if len(f0_filtered) > 0 else 0
    f0_max = np.max(f0_filtered) if len(f0_filtered) > 0 else 0

    # 음절 기반 운율 자질 계산
    syllable_boundaries = np.where(voiced_segments)[0]
    syllable_durations = np.diff(syllable_boundaries) / sr if len(syllable_boundaries) > 1 else 0

    varco_v = np.std(syllable_durations) / np.mean(syllable_durations) * 50 if len(syllable_durations) > 1 else 0
    varco_c = np.std(energy) / np.mean(energy) * 70 if np.mean(energy) > 0 else 0

    rPVI_v = np.sum(np.abs(np.diff(syllable_durations))) if len(syllable_durations) > 1 else 0
    nPVI_v = np.mean(np.abs(np.diff(syllable_durations)) / (syllable_durations[:-1] + syllable_durations[1:])) * 130 if len(syllable_durations) > 2 else 0

    percent_v = np.sum(voiced_segments) / len(voiced_segments) * 100 if len(voiced_segments) > 0 else 0
    
    return [
        f0_mean, f0_std, f0_min, f0_max,
        varco_v, varco_c,
        rPVI_v, nPVI_v,
        percent_v
    ]

