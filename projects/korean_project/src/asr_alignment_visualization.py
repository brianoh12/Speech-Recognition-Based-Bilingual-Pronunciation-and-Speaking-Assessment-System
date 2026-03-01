"""ASR alignment and spectrogram visualization helpers (Korean project)."""

import torch
import torchaudio
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import IPython
from dataclasses import dataclass
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import tempfile
import soundfile as sf
import os
import shutil
import atexit

import matplotlib
matplotlib.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

asr_processor = Wav2Vec2Processor.from_pretrained("slplab/wav2vec2-xls-r_Korean_ASR_by_foreigners")
asr_model = Wav2Vec2ForCTC.from_pretrained("slplab/wav2vec2-xls-r_Korean_ASR_by_foreigners").to(device)

def visualize_asr_analysis(file_path, transcript, emissions):
    """음성 파일을 분석하고 ASR 정렬 시각화를 수행"""
    wav, sample_rate = torchaudio.load(file_path)
    wav = wav.to(device)

    emission = emissions[0].cpu().detach()
    labels = asr_processor.tokenizer.get_vocab()
    label_list = [key for key, _ in sorted(labels.items(), key=lambda item: item[1])]

    # 시각화 함수 정의
    def plot_emission(emission):
        fig, ax = plt.subplots(figsize=(15, 5))
        img = ax.imshow(emission.T, aspect='auto', origin='lower')
        ax.set_title("Frame-wise class probability")
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Labels")
        fig.colorbar(img, ax=ax, shrink=0.6, location="bottom")
        plt.show()

    # print("\n**ASR Alignment Visualization**")
    # print(f"Transcribed Sentence: {transcript}")
    # IPython.display.display(IPython.display.Audio(file_path))
    # plot_emission(emission)

    # 2 단계 Generate alignment probability (trellis) + Visualization
    transcript = "|" + "|".join(transcript.split()) + "|"

    dictionary = {c: i for i, c in enumerate(label_list)}
    tokens = asr_processor.tokenizer(transcript, return_tensors="pt")["input_ids"].squeeze(0).tolist()


    def get_trellis(emission, tokens, blank_id=0):
        num_frame = emission.size(0)
        num_tokens = len(tokens)
        trellis = torch.zeros((num_frame, num_tokens))
        trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
        trellis[0, 1:] = -float("inf")
        trellis[:, 0] = torch.cumsum(emission[:, blank_id], 0)

        for t in range(num_frame - 1):
            trellis[t + 1, 1:] = torch.maximum(
                trellis[t, 1:] + emission[t, blank_id],
                trellis[t, :-1] + emission[t, tokens[1:]],
            )
        return trellis

    trellis = get_trellis(emission, tokens)

    def plot_trellis(trellis):
        fig, ax = plt.subplots()
        img = ax.imshow(trellis.T, origin="lower")
        fig.colorbar(img, ax=ax, shrink=0.6, location="bottom")
        plt.show()

    # plot_trellis(trellis)

    # 3단계 Find the most likely path (backtracking) + Visualization
    @dataclass
    class Point:
        token_index: int
        time_index: int
        score: float

    def backtrack(trellis, emission, tokens, blank_id=0):
        t, j = trellis.size(0) - 1, trellis.size(1) - 1
        path = [Point(j, t, emission[t, blank_id].exp().item())]
        while j > 0:
            # Should not happen but just in case
            assert t > 0

            # 1. Figure out if the current position was stay or change
            p_stay = emission[t - 1, blank_id]
            p_change = emission[t - 1, tokens[j]]

            # Context-aware score for stay vs change
            stayed = trellis[t - 1, j] + p_stay
            changed = trellis[t - 1, j - 1] + p_change

             # Update position
            t -= 1
            if changed > stayed:
                j -= 1
            
            # Store the path with frame-wise probability.
            prob = (p_change if changed > stayed else p_stay).exp().item()
            path.append(Point(j, t, prob))

        # Now j == 0, which means, it reached the SoS.
        # Fill up the rest for the sake of visualization
        while t > 0:
            prob = emission[t - 1, blank_id].exp().item()
            path.append(Point(j, t - 1, prob))
            t -= 1

        return path[::-1]

    path = backtrack(trellis, emission, tokens)


    # Trellis와 경로 시각화 함수 정의
    def plot_trellis_with_path(trellis, path):
        trellis_with_path = trellis.clone()
        for p in path:
            trellis_with_path[p.time_index, p.token_index] = float("nan")
        plt.imshow(trellis_with_path.T, origin="lower")
        plt.title("The path found by backtracking")
        plt.tight_layout()
        plt.show()

    # plot_trellis_with_path(trellis, path)

    # 4 단계 Segment the path + Visualization
    @dataclass
    class Segment:
        label: str
        start: int
        end: int
        score: float

        def __repr__(self):
            return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

        @property
        def length(self):
            return self.end - self.start

    # 반복되는 라벨 병합 함수
    def merge_repeats(path):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    transcript[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )
            i1 = i2
        return segments

    # 경로에서 라벨 병합
    segments = merge_repeats(path)

    # Trellis와 병합된 라벨 시각화 함수 정의
    def plot_trellis_with_segments(trellis, segments, transcript):
        # To plot trellis with path, we take advantage of 'nan' value
        trellis_with_path = trellis.clone()
        for i, seg in enumerate(segments):
            if seg.label != "|":
                trellis_with_path[seg.start : seg.end, i] = float("nan")

        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(20,10))
        ax1.set_title("Path, label and probability for each label")
        ax1.imshow(trellis_with_path.T, origin="lower", aspect="auto")

        for i, seg in enumerate(segments):
            if seg.label != "|":
                ax1.annotate(seg.label, (seg.start, i - 0.7), size="small")
                ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 3), size="small")

        ax2.set_title("Label probability with and without repetation")
        xs, hs, ws = [], [], []
        for seg in segments:
            if seg.label != "|":
                xs.append((seg.end + seg.start) / 2 + 0.4)
                hs.append(seg.score)
                ws.append(seg.end - seg.start)
                ax2.annotate(seg.label, (seg.start + 0.8, -0.07))
        ax2.bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")

        xs, hs = [], []
        for p in path:
            label = transcript[p.token_index]
            if label != "|":
                xs.append(p.time_index + 1)
                hs.append(p.score)

        ax2.bar(xs, hs, width=0.5, alpha=0.5)
        ax2.axhline(0, color="black")
        ax2.grid(True, axis="y")
        ax2.set_ylim(-0.1, 1.1)
        fig.tight_layout()
        plt.show()

    # 시각화
    # plot_trellis_with_segments(trellis, segments, transcript)

    # 5 단계 Merge the segments into words + Visualization
    def merge_words(segments, separator="|"):
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words

    word_segments = merge_words(segments)
    

    # 정렬 시각화 함수 정의
    def plot_alignments(trellis, segments, word_segments, waveform, sample_rate=16000):
        trellis_with_path = trellis.clone()
        for i, seg in enumerate(segments):
            if seg.label != "|":
                trellis_with_path[seg.start : seg.end, i] = float("nan")

        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 10))

        ax1.imshow(trellis_with_path.T, origin="lower", aspect="auto")
        ax1.set_facecolor("lightgray")
        ax1.set_xticks([])
        ax1.set_yticks([])

        for word in word_segments:
            ax1.axvspan(word.start - 0.5, word.end - 0.5, edgecolor="white", facecolor="none")

        for i, seg in enumerate(segments):
            if seg.label != "|":
                ax1.annotate(seg.label, (seg.start, i - 0.7), size="small")
                ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 3), size="small")

        # 원본 waveform 시각화
        ratio = waveform.size(0) / sample_rate / trellis.size(0)
        eps = 1e-10  # Small constant to prevent log(0)
        # ax2.specgram(waveform.cpu().numpy() + eps, Fs=sample_rate)
        ax2.specgram(waveform.cpu().numpy() + eps, Fs=sample_rate, mode="magnitude")

        for word in word_segments:
            x0 = ratio * word.start
            x1 = ratio * word.end
            ax2.axvspan(x0, x1, facecolor="none", edgecolor="white", hatch="/")
            ax2.annotate(f"{word.score:.2f}", (x0, sample_rate * 0.51), annotation_clip=False)

        for seg in segments:
            if seg.label != "|":
                ax2.annotate(seg.label, (seg.start * ratio, sample_rate * 0.55), annotation_clip=False)
        ax2.set_xlabel("time [second]")
        ax2.set_yticks([])
        fig.tight_layout()

        return fig


    # 단어별 음성 영구 저장할 디렉토리 생성 (존재하지 않으면 자동 생성)
    # save_dir = "./saved_audio"
    # os.makedirs(save_dir, exist_ok=True)

    # 임시 디렉토리 생성: 평가 세션 동안만 사용할 디렉트리
    temp_audio_dir = tempfile.mkdtemp(prefix="temp_saved_audio")
    atexit.register(lambda: shutil.rmtree(temp_audio_dir, ignore_errors=True))
    save_dir = temp_audio_dir

    # **6단계: 모든 세그먼트 오디오 출력 함수 정의
    def display_all_segments():

        word_audio_segments = []  # New list to collect word audio info
        ratio = wav.size(1) / trellis.size(0)
        for i, word in enumerate(word_segments):
            x0 = int(ratio * word.start)
            x1 = int(ratio * word.end)

            ###########
            # **Waveform 범위 초과 방지**
            x0 = max(0, x0)  # 시작점이 0보다 작아지지 않도록
            x1 = min(wav.size(1), x1)  # 종료점이 전체 길이를 넘지 않도록

            # **길이가 0이면 스킵**
            if x1 <= x0:
                continue
            ###########

            print(f"{word.label} ({word.score:.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
            segment = wav[:, x0:x1]
            
            ############################# 추가된 부분 ############################
            # Save the segment to a temporary file
            temp_segment_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=save_dir)
            sf.write(temp_segment_file.name, segment.cpu().numpy().squeeze(), sample_rate)
            
            # Append a tuple: (label, score, file_path)
            word_audio_segments.append((word.label, word.score, temp_segment_file.name))

        return word_audio_segments
            ############################# 추가된 부분 ############################
        
            # IPython.display.display(IPython.display.Audio(segment.cpu().numpy(), rate=sample_rate)) # 각 단어 분리 음성

     # 5 단계 & 6단계 시각화
    print(transcript)
    # IPython.display.display(IPython.display.Audio(file_path))
    # display_all_segments()

    word_segments_audio = display_all_segments()
    
    figure = plot_alignments(trellis, segments, word_segments, wav[0])
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    figure.savefig(tmpfile.name)  # 그림을 파일로 저장
    plt.close(figure) # 그림 닫기

    return tmpfile.name, word_segments_audio


