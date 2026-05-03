---
layout: post
title: "YoNoSplat"
date: 2026-05-03 11:00:00 +0900
categories: [Paper Review]
tags: [3DGS, 3D Gaussian Splatting]
---

# YoNoSplat [Paper](https://arxiv.org/pdf/2511.07321) [Code](https://github.com/cvg/YoNoSplat)

![YoNoSplat Teaser](/assets/images/yonosplat-teaser.png)

Botao Ye · Boqi Chen · Haofei Xu · Daniel Barath · Marc Pollefeys
{: .text-muted }

YoNoSplat reconstructs 3D Gaussian splats directly from unposed and uncalibrated images, while flexibly leveraging ground-truth camera poses or intrinsics when available.
{: .text-muted }

## 📌 Basic Information

| 항목 | 내용 |
|------|------|
| **Title** | YoNoSplat: You Only Need One Model for Feedforward 3D Gaussian Splatting |
| **Authors & Affiliations** | Botao Ye, Boqi Chen, Haofei Xu, Daniel Barath, Marc Pollefeys (ETH Zurich, ETH AI Center, Microsoft) |
| **Year / Venue** | 2025 / arXiv Preprint |
| **arXiv / DOI** | arXiv:2511.07321v1 [cs.CV] |
| **Code / Project Page** | botaoye.github.io/yonosplat/ |

---

## 1. 🔭 Research Direction

- 이 논문이 해결하려는 **핵심 문제(core problem)**는 무엇인가?
  - Unstructured 이미지 모음으로부터 빠르고 유연하게 고품질의 3D Gaussian Splatting(3DGS) scene을 reconstruction하는 문제를 해결하고자 한다.
- 기존 연구(prior work)의 어떤 **한계(limitation)**를 지적하는가?
  - 기존의 feedforward 모델들은 정확한 camera pose를 요구하거나, 캘리브레이션된 intrinsics를 필요로 하거나, 고정되고 제한된 개수의 입력 뷰(view)만 처리할 수 있다는 엄격한 제약 조건을 가진다.
  - 최근의 pose-free 방법들은 2~4개의 적은 뷰에서는 잘 동작하지만, unified canonical space로 직접 예측하는 구조 때문에 뷰 수가 많아지면 확장성(scalability)이 떨어져 성능이 하락한다.
- 이 연구가 속한 **task / domain**은 무엇인가?
  - 3D Scene Reconstruction 및 Novel View Synthesis (NVS).
- 연구의 **최종 목표(goal)**와 **동기(motivation)**는?
  - 임의의 개수($V$)의 이미지(posed/unposed, calibrated/uncalibrated)를 입력받아 하나의 모델로 유연하게 3D scene을 재구성하는 범용적인 feedforward 모델을 제안하는 것이다.
  - Pose와 3D geometry의 학습이 심하게 얽혀있는 문제(entanglement)와 scale ambiguity 문제를 해결하여 모델의 안정성과 확장성을 극대화하는 것을 목표로 한다.

---

## 2. 🔁 Model Input & Output

| 구분 | 내용 | 형태(shape / format) |
|------|------|----------------------|
| **Input** | 임의의 개수($V$)의 Unposed & Uncalibrated 이미지 (선택적으로 ground-truth pose 및 intrinsics 제공 가능) | $\{I^v\}_{v=1}^V, \quad I^v \in \mathbb{R}^{3 \times H \times W}$ |
| **Output** | 각 입력 뷰에 대한 Local 3D Gaussian 파라미터(center, opacity, rotation, scale, color), Camera Poses, Camera Intrinsics | $\{ \cup (\mu_j^v, \alpha_j^v, r_j^v, s_j^v, c_j^v), k^v, p^v \}_{j=1,\dots,H\times W}^{v=1,\dots,V}$ |
| **Task 유형** | Feedforward 3D Reconstruction, Novel View Synthesis, Camera Pose Estimation | Feedforward Network Inference |
| **Supervision** | Supervised (Ground-truth pose, intrinsic, target views rendering loss 활용) | |

---

## 3. ⚙️ Method — Architecture & Mathematical Formulation

### 3-1. Overall Pipeline Overview

YoNoSplat은 입력 이미지를 DINOv2 기반의 인코더로 특징을 추출함과 동시에 camera intrinsic을 예측하고, 이를 Intrinsic Condition Embedding (ICE) 모듈을 통해 ray 정보로 변환하여 특징맵에 더해준다. 이후 Local-Global Attention을 통해 여러 뷰 간의 특징을 융합하며, 분리된 Head들을 통해 각 뷰에 대한 Local 3D Gaussian 파라미터와 Camera pose를 예측한 뒤, 주어진(또는 예측된) pose를 사용하여 global 3D representation으로 통합한다.

| Step | Module / Component | Input | Output |
|------|--------------------|-------|--------|
| ① | **Encoder & Intrinsic Head** | 입력 이미지 $\{I^v\}_{v=1}^V$, Learnable Intrinsic Token | Image Features, Predicted Intrinsics $k^v$ |
| ② | **Intrinsic Condition Embedding (ICE)** | Image Features, Intrinsics $k^v$ | Intrinsic-Conditioned Features |
| ③ | **Local-Global Attention Decoder** | Intrinsic-Conditioned Features | Multi-view Fused Features $\mathbf{F}_{\text{fused}}$ |
| ④ | **Gaussian Heads** | Upsampled Fused Features, Input Image Skip Connection | Local Gaussian Parameters $(\mu, \alpha, r, s, c)$ |
| ⑤ | **Pose Head & Global Aggregation** | Multi-view Fused Features | Predicted Poses $p^v$, Global 3D Gaussians |

---

### 3-2. Step-by-Step Architecture Flow

#### Step ① — Encoder & Intrinsic Head

**Input:**  
$\{I^v \in \mathbb{R}^{3 \times H \times W}\}_{v=1}^V$, Intrinsic Token

**Process:**  
입력 이미지를 patch 단위로 나누어 토큰화하고 learnable intrinsic token과 결합하여 DINOv2 아키텍처 기반의 Vision Transformer (ViT) 인코더에 통과시킨다. 인코딩 단계에서 추출된 intrinsic token은 별도의 MLP 레이어를 통과하여 카메라 intrinsics를 예측한다.

$$
k^v = \text{MLP}(\text{Intrinsic\_Token}^v)
$$

- 각 기호 설명: $k^v$ = 예측된 카메라 intrinsics (focal length 등).
- 직관적 의미: 단일 이미지로부터 scale 정보를 유추하기 위해 개별 뷰 레벨에서 카메라 내부 파라미터를 추정한다.

**Output:**  
Image Feature Tokens, Predicted Intrinsics $k^v$

---

#### Step ② — Intrinsic Condition Embedding (ICE)

**Input:**  
Image Feature Tokens, Predicted (or GT) Intrinsics $k^v$

**Process:**  
Scale ambiguity 문제를 해결하기 위해, 추정된(또는 제공된 GT) intrinsics를 camera ray로 변환한 뒤 선형 레이어를 통과시켜 임베딩을 얻고, 이를 기존 이미지 특징에 더해준다 (Residual connection). 

$$
\mathbf{F}_{\text{conditioned}}^v = \mathbf{F}_{\text{img}}^v + \text{Linear}(\text{Rays}(k^v))
$$

- 기호 설명: $\mathbf{F}_{\text{img}}^v$ = 이미지 특징맵, $\text{Rays}(\cdot)$ = intrinsics를 기반으로 한 카메라 ray 생성 함수.
- 직관적 의미: 네트워크가 3D 공간의 scale과 원근감을 올바르게 인지하도록 기하학적 단서를 주입하는 과정이다.

**Output:**  
$\mathbf{F}_{\text{conditioned}}^v$

---

#### Step ③ — Local-Global Attention Decoder

**Input:**  
$\mathbf{F}_{\text{conditioned}}^v$

**Process:**  
$N$개의 alternating attention block을 거친다. 각 블록은 개별 프레임의 특징을 정제하는 per-frame self-attention 레이어와, 모든 뷰의 토큰을 병합하여 프레임 간 정보 교환을 돕는 global concatenated self-attention 레이어로 구성된다.

$$
\mathbf{F}_{\text{fused}} = \text{GlobalSelfAttention}(\text{LocalSelfAttention}(\mathbf{F}_{\text{conditioned}}))
$$

- 직관적 의미: 단일 뷰에서 볼 수 없는 3D 기하학적 구조를 파악하기 위해 여러 카메라 뷰 간의 특징을 매칭하고 융합한다.

**Output:**  
$\mathbf{F}_{\text{fused}}$ (Multi-view fused features)

---

#### Step ④ — Gaussian & Pose Heads

**Input:**  
$\mathbf{F}_{\text{fused}}$

**Process:**  
- **Gaussian Heads**: 특징맵을 2배 업샘플링하고 입력 이미지의 skip connection을 추가한 뒤, 두 개의 분리된 헤드(center 헤드 및 나머지 파라미터 헤드)를 통해 Local 3D Gaussian 파라미터를 예측한다.
- **Pose Head**: 특징맵을 MLP와 average pooling에 통과시켜 12D 카메라 벡터(3D translation $t^v$와 9D rotation 표현)를 예측한 뒤 SVD 직교화로 회전 행렬 $R^v$를 얻는다.

$$
f_\theta: \{ I^v \}_{v=1}^V \mapsto \{ \cup (\mu_j^v, \alpha_j^v, r_j^v, s_j^v, c_j^v), k^v, p^v \}_{j=1,\dots,H\times W}^{v=1,\dots,V}
$$

- 기호 설명: $f_\theta$ = 전체 네트워크, $\mu_j^v, \alpha_j^v, r_j^v, s_j^v, c_j^v$ = 뷰 $v$의 $j$번째 픽셀에 대한 Gaussian 속성(위치, 투명도, 회전, 스케일, 색상), $p^v = [R^v, t^v]$ = 카메라 외부 파라미터.
- 직관적 의미: 네트워크가 픽셀 단위로 로컬 3D Gaussian을 생성하고, 이를 전역 공간으로 배치하기 위한 카메라 포즈를 동시에 산출한다.

**Output:**  
Local Gaussians, $p^v$ (Camera Poses)

---

### 3-3. Key Technical Contribution — Core Equations

#### (A) Mix-forcing Training Strategy

논문은 pose와 geometry 학습의 entanglement 문제를 풀기 위해 Mix-forcing이라는 학습 스케줄을 사용한다.

| 기호 | 의미 | 비고 |
|------|------|------|
| $t_{\text{start}}$ | Mix-forcing이 시작되는 학습 스텝 | 80k로 설정 |
| $t_{\text{end}}$ | 최종 섞임 비율에 도달하는 학습 스텝 | 100k로 설정 |
| $r$ | 모델 예측 pose의 사용 비율 (mixing ratio) | 최종 0.1 적용 |

- 이 전략이 해결하는 문제: 예측된 pose만 사용하면(self-forcing) 학습이 불안정해지고, GT pose만 사용하면(teacher-forcing) 추론 시 노출 편향(exposure bias)이 발생하여 성능이 저하되는 딜레마를 해결한다.
- 기존 방법과의 차이: 초기에는 GT pose로 기하학적 기초를 단단히 다지고, 점진적으로 예측 pose의 비율을 늘려 모델이 자체 예측 오류에 강건해지도록 유도한다.

#### (B) Max Pairwise Distance Normalization

$$
\hat{c}_i = \frac{c_i}{s}, \quad \text{where} \quad s = \max_{i,j} \| c_i - c_j \|_2
$$

| 기호 | 의미 | 비고 |
|------|------|------|
| $c_i, c_j$ | $i$번째, $j$번째 카메라 중심(center) 위치 | |
| $s$ | 카메라 중심들 간의 최대 유클리디안 거리 | Scale factor |
| $\hat{c}_i$ | 정규화된 카메라 중심 | |

- 이 수식이 해결하는 문제: SfM 데이터셋의 scale 모호성(scale ambiguity)을 제거한다.
- 기존 방법과의 차이: Ground-truth depth를 기반으로 정규화하는 기존 방법들과 달리 카메라 거리만을 사용하여 정규화를 수행하며, relative pose loss와 완벽히 호환된다.

---

### 3-4. Loss Function & Training Strategy

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{image}} + \lambda_{\text{intrin}} \mathcal{L}_{\text{intrin}} + \lambda_{\text{pose}} \mathcal{L}_{\text{pose}} + \lambda_{\text{opacity}} \mathcal{L}_{\text{opacity}}
$$

| Loss 항 | 수식 (또는 구성) | 역할 |
|---------|------|------|
| $\mathcal{L}_{\text{image}}$ | $\text{MSE} + \text{LPIPS}$ | 목표 뷰에서의 렌더링 품질을 최적화 (Gaussian 학습) |
| $\mathcal{L}_{\text{intrin}}$ | $\| \hat{k}^v - k_{\text{GT}}^v \|_2$ | 예측된 focal length와 GT 간의 L2 distance |
| $\mathcal{L}_{\text{pose}}$ | $\frac{1}{N(N-1)}\sum_{i\ne j}(\mathcal{L}_R(i,j) + \lambda_t\mathcal{L}_t(i,j))$ | 입력 이미지 쌍에 대한 상대적(relative) pose 예측 학습 (순서 불변성 유지) |
| $\mathcal{L}_{\text{opacity}}$ | $\frac{1}{M}\sum_{i=1}^M |o_i|$ | 투명도 기반 희소성(sparsity) 정규화로 불필요한 Gaussian 억제 |

- 학습 스케줄 / optimizer: AdamW 옵티마이저 사용, Backbone LR $2\times10^{-5}$, 나머지 $2\times10^{-4}$.
- 하이퍼파라미터 세팅: $\lambda_{\text{intrin}} = 0.5$, $\lambda_{\text{pose}} = 0.1$, $\lambda_{\text{opacity}} = 0.01$. Gaussian pruning 임계값 $o_i < 0.005$.

---

### 3-5. Architecture Highlights

- **Backbone / Pretrained model:** DINOv2 Large 구조를 사용하며, $\pi^3$ 모델의 가중치로 초기화하여 fine-tuning을 진행한다.
- **Novel module / design choice:**
  1. 임의 개수의 뷰 확장에 유리한 Local Gaussian 예측 및 Global Aggregation 구조.
  2. Intrinsic Condition Embedding (ICE) 모듈을 통한 캘리브레이션 불필요 기능.
  3. Mix-forcing 훈련 스케줄.
- **Inference 시 특이사항:** Pose-free, Intrinsic-free 조건에서도 feedforward 추론이 가능하며, 선택적으로 200 iteration의 빠른 post-optimization을 적용하여 성능을 더욱 끌어올릴 수 있다.

---

## 4. 📊 Experimental Results

### 4-1. Datasets & Splits

| Dataset | 용도 | Split |
|---------|------|-------|
| RealEstate10K | 실내 환경 훈련 및 평가 | Train 67,477 / Test 7,289 비디오 (테스트는 프레임 200 이상인 1,580 시퀀스) |
| DL3DV-10K | 실외 환경 훈련 및 평가 | Train 10,000 / Test 140 비디오 |
| ScanNet++ | Zero-shot Generalization 평가 | DL3DV에서 훈련된 모델로 평가 (32, 64, 128 뷰 샘플링) |

### 4-2. Baselines & Evaluation Metrics

- **Baselines:** 
  - Optimization-based: InstantSplat.
  - Pose-dependent: MVSplat, DepthSplat.
  - Pose-free: NoPoSplat, AnySplat.
- **Metrics:** NVS 성능 평가를 위한 PSNR, SSIM, LPIPS. Pose 추정 평가를 위한 AUC ($5^\circ, 10^\circ, 20^\circ$ threshold).

### 4-3. Quantitative Results

(DL3DV 데이터셋, 24 View 기준 NVS 성능 요약)

| Method | PSNR (↑) | SSIM (↑) | LPIPS (↓) |
|--------|----------|----------|----------|
| **Ours (Pose-free, Intrinsic-free)** | **22.174** | **0.720** | **0.209** |
| Ours (w/ Pose & Intrinsic) | 22.664 | 0.758 | 0.192 |
| DepthSplat (Pose-dependent) | 20.088 | 0.690 | 0.240 |
| AnySplat (Pose-free) | 19.703 | 0.596 | 0.249 |
| InstantSplat (Opt-based) | 18.493 | 0.510 | 0.381 |

- 제안 방법이 앞서는 시나리오: Pose와 Intrinsic이 주어지지 않은 가장 제약이 심한 상황에서도 기존 SOTA Pose-dependent 모델(DepthSplat)을 큰 격차로 압도함. 뷰 수가 늘어날수록 확장성 측면에서 Canonical space 기반 모델(AnySplat 등) 대비 월등한 성능을 보임.
- 제안 방법이 뒤처지는 시나리오: 제시된 실험군 내에서 지속적으로 우세를 점하였으나, 모델 최적화(Post-optimization)를 진행한 자체 결과 대비 feedforward 단일 추론 결과는 수치적으로 약간 더 낮음.

### 4-4. Qualitative Results

- Visualization에서 눈에 띄는 점: 다른 방법론(AnySplat, DepthSplat 등)에 비해 멀티 뷰 컨텐츠 융합이 훨씬 일관성 있게 이루어져 깨짐(artifact)이나 부정확한 지오메트리 렌더링이 현저히 적음. ScanNet++ 제로샷 일반화 평가에서도 훨씬 선명한 렌더링 품질을 보여줌.

### 4-5. Ablation Study

| Ablation 대상 | 제거 시 성능 변화 | 결론 |
|--------------|-----------------|------|
| **Mix-forcing 전략** | Self-forcing 단독 시 PSNR 24.150 하락, Teacher-forcing 시 평가 세팅에 따라 노출 편향 발생 | Mix-forcing(25.587)이 두 세팅 간의 최적 균형을 이끌어냄 |
| **Max pairwise distance 정규화** | 정규화 미적용 시 PSNR이 22.662로 폭락 (Max-pairwise 적용 시 25.212) | 상대적 포즈 손실함수와 일치하는 가장 안정적인 스케일 정규화 방식임 |
| **ICE 모듈 (Intrinsic)** | Intrinsic 미사용 시 PSNR 24.481로 하락 (예측 Intrinsic 사용 시 24.711) | Scale ambiguity를 줄이는 데 ICE 모듈이 결정적인 역할을 함 |

---

## 5. 💡 Contributions

- **저자가 주장하는 contributions (논문 원문 기준):**
  1. 임의의 입력 뷰 수에서 Pose-free와 Pose-dependent 설정 모두에서 SOTA 성능을 달성한 최초의 feedforward 모델, YoNoSplat을 제안함.
  2. Pose와 Geometry 학습의 얽힘(entanglement) 문제를 Mix-forcing 학습 전략을 통해 효과적으로 완화하여 학습 불안정과 노출 편향을 방지함.
  3. Intrinsic 예측 기반의 컨디셔닝 파이프라인(ICE)과 Max pairwise distance 정규화 기법을 도입하여 Uncalibrated 이미지에서 발생하는 Scale ambiguity 문제를 해결함.

- **리뷰어 관점에서 가장 실질적인 contribution:**
  입력 데이터의 제약(Pose 유무, Camera 캘리브레이션 유무, 이미지 장수)을 하나의 파이프라인 안에서 모두 유연하게 소화할 수 있도록 설계한 'Local 예측 + Global 통합(Mix-forcing)' 디자인 초이스가 가장 훌륭하며, 실무적인 사용성을 크게 높인 점.

---

## 6. 🧱 Related Work & Positioning

- 직접 비교 / 참조하는 주요 논문들: NoPoSplat, AnySplat, DepthSplat, MVSplat, DUSt3R, $\pi^3$.
- 이 논문이 속한 연구 계보 (lineage): 단일/소수 뷰 기반의 Feedforward 3D Gaussian Splatting 연구 흐름과 $\pi^3$, VGGT 등으로 대변되는 Feedforward Point Cloud Pose 추정 연구의 교집합.
- 기존 접근 방식 대비 패러다임 변화 여부: 기존 Pose-free 3DGS 모델들이 Canonical Space에 직접 Gaussian을 투영하여 뷰 수 확장에 실패했던 패러다임에서 벗어나, Local 예측 후 Pose 추정을 거쳐 Global 공간으로 매핑하는 전통적 파이프라인을 딥러닝 기반 Feedforward 학습으로 매끄럽게 통합함.

---

## 7. ✅ Strengths

1. **Extreme Versatility:** Pose, Intrinsic, 이미지 장수에 구애받지 않고 유연하게 작동하는 범용적 프레임워크.
2. **Innovative Training Strategy:** Mix-forcing이라는 직관적이고 효과적인 커리큘럼 러닝을 도입해 End-to-end Pose-Geometry 동시 학습의 한계를 극복함.
3. **Superior Performance & Scalability:** 100장이 넘는 뷰(view)에서도 단 몇 초 만에 고품질 3D Reconstruction이 가능하며, 다른 도메인(ScanNet++)에서도 강건한 Zero-shot 일반화 성능을 입증함.

---

## 8. ⚠️ Limitations & Weaknesses

- **저자가 인정한 한계:** 
  입력 가능한 최대 뷰(view)의 수가 GPU 메모리 용량에 의해 직접적인 제약을 받음. 또한 Post-optimization을 수행할 경우 여전히 눈에 띄는 성능 향상이 존재하므로 Feedforward 예측 모델 자체로서의 발전 여지가 남아있음.
- **리뷰어가 판단한 추가 한계:**
  - 실험 설계상 아쉬운 점: DINOv2 Large 모델과 32장의 GH200 GPU를 학습에 동원해야 하는 등 막대한 컴퓨팅 리소스가 필요하여 진입 장벽이 매우 높음.
  - 실용성 문제: 모바일이나 엣지 디바이스에서 실시간 추론(Inference)을 수행하기에는 파라미터와 메모리 소모량이 거대함.

---

## 9. 🔮 Future Work

- **저자가 제시한 future work:** GPU 메모리 제약을 극복하기 위한 Incremental feedforward reconstruction (순차적 방식의 확장) 연구.
- **리뷰어가 생각하는 확장 가능성:** 고정된 해상도 연산이나 뷰 제약을 완화할 수 있도록 Linear Attention 기반의 시퀀스 모델(Mamba 등)을 Decoder에 결합하여 극단적으로 많은 뷰를 $O(N)$으로 처리하는 아키텍처 개선.

---

## 10. 🗒️ Overall Assessment

| 항목 | 내용 |
|------|------|
| **한 줄 요약** | 단일 모델로 임의 개수의 Unposed/Uncalibrated 이미지 처리를 정복한 가장 진보된 Feedforward 3DGS. |
| **예상 임팩트** | 3D Reconstruction 파이프라인에서 COLMAP과 같은 무거운 SfM 모듈을 완전히 대체할 수 있는 실질적 대안으로 자리매김할 가능성이 큼. |
| **추천 독자층** | NeRF, 3DGS, SLAM, 대규모 3D Vision 모델 연구자 및 엔지니어 |
| **개인 평점** | ⭐⭐⭐⭐⭐ / 5 |