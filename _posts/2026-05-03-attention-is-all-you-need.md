---
layout: post
title: "Attention Is All You Need 논문 핵심 요약"
date: 2026-05-03 10:00:00 +0900
categories: ["Paper Review"]
tags: ["NLP", "Transformer", "DeepLearning"]
description: "Transformer 아키텍처의 Self-Attention 메커니즘을 수학적 수식과 함께 상세히 뜯어봅니다."
---

이 글에서는 구글 브레인이 2017년에 발표한 기념비적인 논문, *Attention Is All You Need*의 핵심 아이디어를 정리합니다.

## 1. Introduction

기존의 RNN이나 CNN 구조를 완전히 배제하고, 오직 **Attention 메커니즘**만을 사용하여 병렬 처리를 극대화한 모델입니다.

> "We propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output."

## 2. Model Architecture

인코더와 디코더 구조를 가지며, Multi-Head Attention이 핵심 역할을 수행합니다. 코드로 구현하면 다음과 같은 형태가 됩니다.

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # 초기화 코드 생략...
```

앞으로 제 개인 프로젝트에서도 이 구조를 적극 활용해 볼 계획입니다.
