# English Pronunciation Assessment with ASR Visualization

이 프로젝트는 Gradio 인터페이스를 통해 사용자의 영어 발음을 평가하고, ASR(Automatic Speech Recognition) 전사 및 정렬 시각화를 제공하는 시스템입니다.  
발음 평가는 Accuracy, Fluency, Prosody, Completeness 네 가지 점수로 산출되며, 추가적으로 음소 예측 및 피드백 기능을 포함합니다.


## 주요 기능

- **발음 평가**
  - **Accuracy Score:** 허깅페이스에 저장된 finetuned wav2vec2 모델을 사용하여 추출한 예측 음소와 정답 음소의 일치도를 기반으로 계산
  - **Fluency Score:** 음성의 유창성을 평가
  - **Prosody Score:** 억양 및 리듬 등 발음의 운율 평가
  - **Completeness Score:** 허깅페이서에서 저장된 finetuned wav2vec2 모델을 사용하여 ASR 전사 결과와 정답 문장의 유사도를 기반으로 계산
  - **Final Score:** 위 네 가지 점수를 가중 평균하여 산출

- **ASR 전사 및 시각화**
  - Wav2Vec2 기반 ASR 모델을 활용하여 음성을 텍스트로 변환
  - 전사 결과를 기반으로 음성 정렬(Alignment) 시각화를 제공

- **음소 예측 및 피드백**
  - 사전학습된 모델을 통해 음소를 예측하고, 정답 음소와의 비교를 통해 상세 피드백 제공


## 실행 환경

- **OS:** Ubuntu 20.04.3 2022.02.18 LTS (Cubic)
- **커널:** 5.11.0-27-generic
- **Python:** 3.9.12
- **CUDA:** 
  - **nvcc:** Cuda compilation tools, release 10.1, V10.1.243  
    (※ nvcc 버전은 CUDA 컴파일러 버전을 의미합니다.)
  - **NVIDIA 드라이버:** 525.125.06 (nvidia-smi에서 확인된 CUDA 버전은 12.0)
- **GPU:** NVIDIA A100 80G (두 개의 GPU가 장착되어 있음)



## 파일 구조

```bash
/Eng_Pron_AI_Model
├── realtime_evaluate_ko.ipynb                             # 메인 실행 파일 (Gradio 인터페이스 및 평가 함수 포함)
├── feature_extraction_ko.py                               # Accuracy, Fluency, Prosody 특성 추출 함수 모듈
├── PronunciationModel_ko.py                               # 발음 평가 모델 클래스 정의
├── asr_visualization_for_realtime_gradio_ko.py            # ASR 정렬 및 시각화 모듈
├── phoneme_feedback_ko.py                                 # 음소 피드백 생성 모듈
├── best_model.pth                                         # 사전학습된 발음 평가 모델 가중치
├── requirements.txt                                       # 필수 라이브러리 목록
└── README.md                                              # 이 파일

## 설치 및 실행 방법

1. **가상환경(선택사항) 구성**  
   - 가상환경을 사용하면 다른 프로젝트와 라이브러리 충돌을 방지할 수 있습니다.
   - 예시(venv):
     ```bash
     python -m venv venv
     source venv/bin/activate  # (Windows: venv\Scripts\activate)
     ```
   - 예시(conda):
     ```bash
     conda create -n eng_pron_env python=3.9
     conda activate eng_pron_env
     ```

2. **필수 라이브러리 설치**  
   - 프로젝트 폴더로 이동한 뒤, 아래 명령어로 `requirements.txt`에 명시된 라이브러리를 설치합니다.
     ```bash
     pip install -r requirements.txt
     ```
   - `requirements.txt`에는 발음 평가 및 ASR 전사에 필요한 필수 라이브러리(예: PyTorch, librosa, gradio 등)와 Jupyter 노트북 실행을 위한 라이브러리(예: jupyter, notebook, ipykernel, ipywidgets 등)가 포함되어 있습니다.

3. **Jupyter 노트북 실행**  
   - 설치가 완료되면, Jupyter 노트북을 실행하여 프로젝트를 확인하고 코드를 실행할 수 있습니다.
     ```bash
     jupyter notebook
     ```
   - 또는
     ```bash
     jupyter lab
     ```
   - 명령어 실행 후 링크 클릭하여 브라우저 열리면, 해당 디렉토리 안에 있는 `realtime_evaluate.ipynb` 파일을 찾아서 열어주세요.

4. **노트북 셀 실행**  
   - `realtime_evaluate_ko.ipynb` 파일이 열리면, 위에서부터 순서대로 셀을 실행합니다. 
   - 각 셀에서는 다음과 같은 작업을 수행합니다:
     - 필요 라이브러리 임포트
     - 음성 평가 모델 및 ASR 모델 로드
     - Gradio 인터페이스 실행 (음성 입력, 결과 평가, 정렬 시각화 등)

5. **발음 평가 테스트**  
   - `realtime_evaluate_ko.ipynb`의 마지막 셀까지 실행하면 Gradio UI가 로컬 서버 형태로 실행됩니다.
   - 출력된 Gradio 인터페이스 혹은 Gradio 링크를 통해 인터페이스를 사용할 수 있습니다.
   - 마이크 입력 혹은 업로드 기능을 통해 음성을 입력하면, Accuracy, Fluency, Prosody, Completeness 점수와 ASR 전사 결과 및 피드백이 출력됩니다.
   - 단어별 음성을 새로운 경로에 영구 저장 하고 싶다면 asr_visualization_for_realtime_gradio.py 파일에서 "# 단어별 음성 영구 저장할 디렉토리 생성 (존재하지 않으면 자동 생성)" 에 해당하는 부분을 주석 해제하시고, 단어별 음성을 해당 세션에서만 유지하고자 한다면 첨부된 파일을 그대로 사용하시면 됩니다.

6. **추가 주의사항** 
  - 샘플링 레이트:
     - 내부적으로 입력 오디오는 16kHz로 변환되어 처리됩니다.
     - 입력 오디오가 16kHz가 아닌 경우, 자동 변환 시 음질 저하가 생길 수 있으므로 가급적 16kHz 오디오를 사용하세요.
   - GPU 사용:
     - CUDA 환경이 구성되어 있다면 GPU를 사용하며, 그렇지 않을 경우 CPU로 실행됩니다.

