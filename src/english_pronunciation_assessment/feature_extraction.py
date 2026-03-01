import librosa
import numpy as np
import torch
import torch.nn as nn
from Levenshtein import ratio

# CMU 음소 세트 정의
CMU_VOWELS = {
    "aa", "ae", "ah", "ao", "aw", "ax", "axr", "ay", "eh", "er", "ey", 
    "ih", "ix", "iy", "ow", "oy", "uh", "uw", "ux"
}

CMU_CONSONANTS = {
    "b", "ch", "d", "dh", "dx", "el", "em", "en", "f", "g", "hh", "jh", "k", "l", "m",
    "n", "ng", "nx", "p", "q", "r", "s", "sh", "t", "th", "v", "w", "wh", "y", "z", "zh"
}

# Function to check if a phoneme is a vowel based on CMU phoneme set
def is_vowel(phoneme):
    return phoneme.strip("012") in CMU_VOWELS  # Strip stress markers (e.g., 'AH0', 'IH1')

def is_consonant(phoneme):
    return phoneme.strip("012") in CMU_CONSONANTS

def clean_phoneme(phoneme):
    return phoneme.replace("_err", "")

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
    
    for p, c in zip(predicted_phonemes, correct_phonemes):
        clean_p = clean_phoneme(p)
        clean_c = clean_phoneme(c)
        
        if p == c:
            score = 1.0  # 정확히 일치
        elif "_err" in p:
            score = err_penalty  # 오류가 포함되면 감점
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

def extract_fluency_features(audio_path):
    y, sr = librosa.load(audio_path)
    intervals = librosa.effects.split(y, top_db=20)
    voiced_duration = sum(i[1] - i[0] for i in intervals) / sr
    total_duration = len(y) / sr
    
    speaking_rate = len(intervals) / total_duration
    articulation_rate = len(intervals) / voiced_duration if voiced_duration > 0 else 0
    
    voice_breaks = len(intervals) - 1
    voice_breaks_duration = total_duration - voiced_duration
    voice_breaks_ratio = voice_breaks_duration / total_duration
    
    return [speaking_rate, articulation_rate, voice_breaks, voice_breaks_ratio]

def extract_prosody_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    energy = librosa.feature.rms(y=y)[0]
    threshold = np.mean(energy)
    voiced_segments = energy > threshold
    
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=300, sr=sr)
    if f0 is not None:
        f0_mean = np.nanmean(f0) if not np.all(np.isnan(f0)) else 0
        f0_std = np.nanstd(f0) if not np.all(np.isnan(f0)) else 0
        f0_min = np.nanmin(f0) if not np.all(np.isnan(f0)) else 0
        f0_max = np.nanmax(f0) if not np.all(np.isnan(f0)) else 0
        f0_range = f0_max - f0_min
    else:
        f0_mean, f0_std, f0_min, f0_max, f0_range = 0, 0, 0, 0, 0
    
    syllable_boundaries = np.where(voiced_segments)[0]
    syllable_durations = np.diff(syllable_boundaries) / sr if len(syllable_boundaries) > 1 else [0]
    varco_v = np.std(syllable_durations) / np.mean(syllable_durations) * 100 if len(syllable_durations) > 1 else 0
    varco_c = np.std(energy) / np.mean(energy) * 100 if np.mean(energy) > 0 else 0
    
    rPVI_v = np.sum(np.abs(np.diff(syllable_durations))) if len(syllable_durations) > 1 else 0
    nPVI_v = (
        np.mean(np.abs(np.diff(syllable_durations)) / (syllable_durations[:-1] + syllable_durations[1:])) * 100
        if len(syllable_durations) > 2 else 0
    )
    rPVI_c, nPVI_c = rPVI_v, nPVI_v
    
    percent_v = np.sum(voiced_segments) / len(voiced_segments) * 100 if len(voiced_segments) > 0 else 0
    
    return [
        f0_mean, f0_std, f0_min, f0_max, f0_range,
        varco_v, varco_c,
        rPVI_v, nPVI_v, rPVI_c, nPVI_c,
        percent_v
    ]

