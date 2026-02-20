# 🚀 CHILLGRAM: AI 기반 콘텐츠 생성 파이프라인

> **Enterprise-grade Content Automation Workflow**
> 최신 Generative AI와 효율적인 클라우드 인프라를 결합하여, 텍스트 구조화부터 고품질 비디오 생성까지 전 과정을 자동화합니다.

---

## 📌 개요 (Overview)
본 프로젝트는 반복적인 콘텐츠 제작 공정을 혁신하기 위한 **End-to-End 자동화 파이프라인**입니다. 마케팅, 광고, SNS 등 시각적 콘텐츠가 즉각적으로 필요한 비즈니스 환경에서 제작 효율성을 극대화하고 품질의 일관성을 보장합니다.

---

## 🎯 문제 정의 및 해결 (Problem & Solution)

| 현행 문제점 (Pain Points) | AI 파이프라인의 해결책 (Solution) |
| :--- | :--- |
| **비용 및 시간 소요** | AI 자동화를 통한 제작 공정 **80% 이상 단축** |
| **스타일 불일치** | 브랜드 가이드라인 기반의 **일관된 AI 스타일 유지** |
| **확장성 한계** | Queue 기반 병렬 처리를 통한 **대량 제작 대응** |
| **반복 작업의 비효율** | 단순 업무 자동화로 **인적 리소스 최적화** |

---

## 🏗️ 시스템 아키텍처 (Architecture)

### **설계 원칙**
* **비동기 및 디커플링:** `RabbitMQ`를 통해 API와 워커를 분리, 시스템 안정성 확보
* **고성능 처리:** `Python asyncio` 기반 워커로 다수의 AI I/O 작업을 병렬 처리
* **모듈화 서비스:** 서비스별(배너, 패키지, SNS, 비디오) 독립 모듈 구성으로 유지보수 용이
* **클라우드 네이티브:** GCS(저장)와 BigQuery(분석)를 활용한 확장성 있는 인프라

### **시스템 구조도**
```mermaid
graph TD
    A[User/Client] --> B(API Gateway);
    B --> C{RabbitMQ};
    C --> D1[Worker: Banner];
    C --> D2[Worker: Package];
    C --> D3[Worker: SNS/Video];
    D1 & D2 & D3 --> E(AI Engine: Gemini / Image Models);
    E --> F[(Cloud Storage)];
    E --> G[(BigQuery)];
