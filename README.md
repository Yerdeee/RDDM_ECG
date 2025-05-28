<h1 align="center"> 
Region-Disentangled Diffusion Model for High-Fidelity PPG-to-ECG Translation
</h1>

RDDM 구조를 활용해서 ecg lead1 -> lead 생성 후 분류 실험을 위한 repo  

### 논문 사이트 (https://arxiv.org/abs/2308.13568)

---

## 2025.04.16 수정  

1. metrics.py -> calculate_FD 수정  
- 크기 reshape 512 고정 -> 1280 변경 (실험 세팅에 맞춰서)

2. data_withdiffusion 추가
- diffusion모델로 lead 생성 후 dataloader로 전환하는 함수 추가

### 사용예시

```python
from data_withdiffusion import get_dataset_withdiffusion
train_loader, val_loader, test_loader = get_dataset_withdiffusion(DATA_PATH = '/cap/RDDM-main/datasets/', MODEL_PATH='/cap/RDDM-main/hsh/ECG2ECG_FINAL/LEAD1TO' ,lead_num=[2], only_one=False)
```
---

## 2025.05.08 수정  

1. metrics.py -> calculate_FD 재수정  
- 크기 reshape 1280 고정 -> window_size에 맞춰서 변경

2. RDDM_classification.ipynb 추가  
- 생성된 신호로 1d/2d cnn 기반 분류 성능 확인 ipynb 파일 추가  

3. std_eval.py 수정
- naive Diffusion, RDDM 둘다 한번에 성능 비교 가능하도록 코드 수정


## 2025.05.28 수정

1. train.py : fftloss / fftcond 추가
- config 변수 활용하여 fftloss, fftcond 사용 할지 안할지 선택 가능

2. model.py : fft condition 모델 추가
- ConditionNetWithFFT class 활용하면 사용 가능
  


