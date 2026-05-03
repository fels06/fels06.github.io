---
layout: post
title: "문서 기반 RAG 챗봇 구축 회고"
date: 2026-04-28 10:00:00 +0900
categories: ["Project"]
tags: ["LLM", "RAG", "LangChain"]
description: "LangChain을 활용하여 개인화된 논문 검색 및 질의응답 시스템을 만든 경험을 공유합니다."
---

이번 프로젝트에서는 LangChain을 활용하여 개인화된 논문 검색 및 질의응답 시스템을 구축했습니다.

RAG(Retrieval-Augmented Generation) 방식을 통해 LLM의 할루시네이션(Hallucination)을 줄이고, 특정 도메인 논문에 특화된 정확한 답변을 생성하는 것이 목표였습니다.

## 핵심 구현 내용
- PDF 문서 파싱 및 텍스트 청크 분할
- Vector Store(ChromaDB)를 이용한 임베딩 검색
- LangChain을 통한 프롬프트 엔지니어링 및 답변 생성 파이프라인 구축

다음번에는 더 다양한 논문 포맷을 지원하도록 기능을 확장할 예정입니다.
