"""Evaluate a trained CTC phone recognizer on a test split."""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import evaluate
from tqdm import tqdm
from datasets import load_from_disk


# Paths can be overridden via environment variables for reproducible runs.
MODEL_PATH = Path(
    os.environ.get("ASR_MODEL_PATH", "./trainer/korean_by_foreigner_word_recognition")
)
DATASET_PATH = Path(os.environ.get("ASR_DATASET_PATH", "./data/new_hf_datasets"))
OUTPUT_FILE = Path(os.environ.get("ASR_EVAL_OUTPUT", "./results/test_result.txt"))

# 1. GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 저장된 모델과 프로세서 로드
model_path = str(MODEL_PATH)
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
model.eval()

# 3. 테스트셋 로드
dataset_path = str(DATASET_PATH)
test_ds = load_from_disk(dataset_path)["test"]

# 4. 데이터 Collator 정의
class DataCollatorCTCWithPadding:
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        batch["labels"] = [feature["words"] for feature in features]
        return batch


def prepare_batch(batch):
    audio_array, sampling_rate = librosa.load(batch["audio"], sr=16000)
    input_values = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_values[0]
    return {"input_values": input_values, "words": batch["words"]}


# Preprocess the dataset to extract input values
print("Preparing test dataset...")
test_ds = test_ds.map(prepare_batch)

# Create a DataLoader for batched processing
batch_size = 8
data_collator = DataCollatorCTCWithPadding(processor)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

# 5. 배치 단위로 테스트셋 평가
predictions = []
references = []

print("Evaluating on the test set...")
for batch in tqdm(test_loader, desc="Processing Test Set"):
    input_values = batch["input_values"].to(device)

    # 모델 예측
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # 디코딩
    predicted_phonemes = processor.batch_decode(predicted_ids)
    predictions.extend(predicted_phonemes)
    references.extend(batch["labels"])

# === 여기에 추가 ===
def clean_text(text):
    return text.replace("<unk>", "").replace("?", "").strip()

predictions = [clean_text(p) for p in predictions]
references = [clean_text(r) for r in references]

# 6. WER, CER, PER 계산
wer_metric = evaluate.load("wer")
wer = wer_metric.compute(predictions=predictions, references=references)

cer_metric = evaluate.load("cer")
cer = cer_metric.compute(predictions=predictions, references=references)

# PER(Phoneme Error Rate) 계산 (WER과 같은 방식으로 계산)
def per(preds, refs):
    total_phonemes = sum(len(ref.split()) for ref in refs)
    total_errors = sum(len(set(ref.split()) - set(pred.split())) for ref, pred in zip(refs, preds))
    return total_errors / total_phonemes

per_score = per(predictions, references)

# 7. 결과를 파일로 저장
output_file = OUTPUT_FILE
output_file.parent.mkdir(parents=True, exist_ok=True)
with output_file.open("w") as f:
    # 7.1. WER, CER, PER 기록
    f.write(f"Word Error Rate (WER) on the test set: {wer:.4f}\n")
    f.write(f"Character Error Rate (CER) on the test set: {cer:.4f}\n")
    f.write(f"Phoneme Error Rate (PER) on the test set: {per_score:.4f}\n")
    f.write("\n")

    # 7.2. 샘플별 결과 기록
    for i, (ref, pred) in enumerate(zip(references, predictions)):
        f.write(f"Sample {i + 1}:\n")
        f.write(f"Correct Sentence: {ref}\n")
        f.write(f"Predicted Sentence: {pred}\n")
        f.write("\n")

# 8. 터미널 출력
print(f"Word Error Rate (WER) on the test set: {wer:.4f}")
print(f"Character Error Rate (CER) on the test set: {cer:.4f}")
print(f"Phoneme Error Rate (PER) on the test set: {per_score:.4f}")

# 8.2. 첫 5개 샘플 결과 출력
for i in range(5):
    print(f"Sample {i + 1}:")
    print(f"True Sentence: {references[i]}")
    print(f"Predicted Sentence: {predictions[i]}")
    print("-" * 50)

print(f"Test results saved to {output_file}")
