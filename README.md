<h1 align="center"> 
Region-Disentangled Diffusion Model for High-Fidelity PPG-to-ECG Translation
</h1>

<h3>
  RDDM 구조를 활용해서 ecg lead1 -> lead 생성 후 분류 실험을 위한 repo
</h3>

### 논문 사이트 (https://arxiv.org/abs/2308.13568)


## 2025.04.16 수정  

metrics.py 수정  
- 크기 reshape 512 고정 -> 1280 변경 (실험 세팅에 맞춰서)
data_withdiffusion 추가
- diffusion모델로 lead 생성 후 dataloader로 전환하는 함수 추가
  
### 사용예시

```python
from data_withdiffusion import get_dataset_withdiffusion
train_loader, val_loader, test_loader = get_dataset_withdiffusion(DATA_PATH = '/cap/RDDM-main/datasets/', MODEL_PATH='/cap/RDDM-main/hsh/ECG2ECG_FINAL/LEAD1TO' ,lead_num=[2], only_one=False)
```


