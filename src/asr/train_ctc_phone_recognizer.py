import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import librosa

import evaluate
import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    HubertForCTC,
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    WavLMForCTC,
)


def prepare_arguments():
    parser = argparse.ArgumentParser(description="Train the SSL on phone recognition.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_name_or_path", type=str, default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--group_by_length", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--ctc_loss_reduction", type=str, default="mean")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--metric_for_best_model", type=str, default="wer")
    parser.add_argument("--greater_is_better", action="store_true")
    parser.add_argument("--exp_prefix", type=str, default="korean_by_foreigner_word_recognition")
    parser.add_argument("--freeze_feature_extractor", type=bool, default=True)

    args = parser.parse_args()
    args.exp_name = f"{args.exp_prefix}_ep{args.num_train_epochs}_lr{args.learning_rate}_warm{args.warmup_ratio}_type-{args.lr_scheduler_type}"
    args.save_dir_path = "./trainer/" + args.exp_name
    args.save_log_path = "./logs/" + args.exp_name
    os.makedirs(args.save_dir_path, exist_ok=True)
    os.makedirs(args.save_log_path, exist_ok=True)

    with open(os.path.join(args.save_dir_path, "args.json"), "w") as args_file:
        json.dump(vars(args), args_file, ensure_ascii=False)

    return args


def prepare_dataset(batch):
    # Load the audio file
    audio_path = batch["audio"]

    if audio_path is None or not os.path.isfile(audio_path):
        raise ValueError(f"Invalid audio file path: {audio_path}")

    audio_array, sampling_rate = librosa.load(audio_path, sr=16000)

    # Process the audio data
    batch["input_values"] = processor(audio_array, sampling_rate=sampling_rate).input_values[0]

    # Process the words
    with processor.as_target_processor():
        batch["labels"] = processor(batch["words"]).input_ids

    return batch



@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch


def compute_wer(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}


def prepare_trainer(args, processor, train_ds, test_ds):
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name_or_path,
        ctc_loss_reduction=args.ctc_loss_reduction,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    if args.freeze_feature_extractor:
        model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir=args.save_dir_path,
        logging_dir=args.save_log_path,
        report_to=["tensorboard"],
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        # eval_accumulation_steps=4,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
    )


    data_collator = DataCollatorCTCWithPadding(processor=processor)

    return Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_wer,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=processor.feature_extractor,
    )



if __name__ == "__main__":
    args = prepare_arguments()

    #외국인의 한국어 발음 음성인식기 학습시
    dataset_path = "/home/brian/joint-apa-mdd-mtl/data/korean_data/for_word_ASR/new_hf_datasets"
    vocab_path = "/home/brian/joint-apa-mdd-mtl/data/korean_data/for_word_ASR/new_hf_datasets/vocab.json"

    # 데이터 로드
    dataset = load_from_disk(dataset_path)  # 데이터셋 로드 추가
    train_ds = dataset["train"]
    valid_ds = dataset["valid"] 
    test_ds = dataset["test"] 

    # 토크나이저 및 프로세서 준비
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path, unk_token="<unk>", pad_token="<pad>", word_delimiter_token=" ", do_phonemize=False
    )
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name_or_path)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(args.save_dir_path)

    # # 학습 데이터와 검증 데이터에 대해 전처리 적용
    train_ds = train_ds.map(prepare_dataset)
    valid_ds = valid_ds.map(prepare_dataset)
    test_ds = test_ds.map(prepare_dataset)

    # 이미 map(prepare_dataset) + save_to_disk 했다면 아래 코드 실행
    # train_ds = load_from_disk("/home/brian/joint-apa-mdd-mtl/data/korean_data/for_word_ASR/new_hf_datasets/train")
    # valid_ds = load_from_disk("/home/brian/joint-apa-mdd-mtl/data/korean_data/for_word_ASR/new_hf_datasets/valid")
    # test_ds  = load_from_disk("/home/brian/joint-apa-mdd-mtl/data/korean_data/for_word_ASR/new_hf_datasets/test")


    # WER 평가 메트릭 준비
    wer_metric = evaluate.load("wer")

    # Trainer 생성
    trainer = prepare_trainer(args, processor, train_ds=train_ds, test_ds=valid_ds)

    # 학습 시작
    trainer.train()

    # 모델 저장
    trainer.save_model(args.save_dir_path)

    # 검증 데이터셋으로 평가
    metrics = trainer.evaluate()
    print(metrics)
