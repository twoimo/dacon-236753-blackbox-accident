# autoresearch 브랜치 안내

> DACON 236753 **블랙박스 영상 기반 지능형 고의사고 분석** 대회의 **리더보드 1등**을 목표로 한
> 근거 기반 자료조사(리서치) + 다기기 재현 실험 환경 브랜치.
> 이 브랜치는 **팀 공유 전용**이며, 실제 모델 실험은 여기서 분기한 별도 브랜치에서 진행한다.

- 대회: https://dacon.io/competitions/official/236753 (상금 4,200만 원, 리더보드 마감 2026-09-29)
- 저장소: https://github.com/twoimo/dacon-236753-blackbox-accident
- 브랜치: `autoresearch`

---

## 0. 30초 요약 (TL;DR)

- **무엇을**: 3개 Stage(재녹화 판별 / 사고 시점·상황 / 차량 거동) 각각에 대해 세계 최상위 학회·저널
  논문을 **인용수와 게재처로 검증**해 정리하고, 팀원이 어느 기기에서든 같은 환경으로 실험하도록
  Docker/conda 환경을 구성했다.
- **왜 이 브랜치인가**: 가재코드 `/skill:autoresearch` 방식(웹 문헌 + 로컬 데이터/제약을 교차하여
  구조화된 verdict로 마무리)을 따라, **자료조사·전략·환경만** 담는다. 코드로 모델을 구현하지 않는다.
- **어디서 시작**: 아래 [3. 팀원 온보딩] → `research/README.md` → `env/README.md` 순으로 읽는다.

---

## 1. 대회 핵심 (근거: `docs/`, `catalog/`)

| 항목 | 내용 |
|---|---|
| 과업 | 하나의 코드 제출로 3개 Stage 결과 동시 산출 |
| Stage 1 | 원본 vs 재녹화(화면 재촬영) 이진 분류 · Macro-F1 · **가중치 0.2** |
| Stage 2 | 충돌/진입 프레임 + 회피공간·진입방향 · 시각오차+범주정확도 · **가중치 0.4** |
| Stage 3 | 0.1초 단위 가감속·조향 범주 · 정확도 · **가중치 0.4** |
| 평가 서버 | NVIDIA **L40S 44.7GiB**, 7 vCPU, 60GB RAM, **추론 60분**, **인터넷 차단** |
| 학습 데이터 | **공식 미제공** → 법적 제한 없는 데이터 자체 구성(CCD, comma2k19 등) |
| 진출 | 1차 상위 15팀 → 2차 '모델 개발 보고서' + '학습데이터 구성 보고서' |

---

## 2. 이 브랜치가 담은 것

```
research/                      ★ 근거 기반 문헌 리서치 + 전략
├── README.md                     읽는 순서·신빙성 원칙 색인
├── references/README.md          논문 신빙성 정본 (학회/저널 등급 + 인용수 출처 병기)
├── synthesis/README.md           종합 전략 + autoresearch verdict(evidence/caveats)
├── 01-stage1-recapture/          재녹화 판별 (재촬영 포렌식, 코덱 누설 방어)
├── 02-stage2-anticipation/       사고 시점·상황 (시간 국소화 + 약지도)
├── 03-stage3-egomotion/          가감속·조향 (ego-motion, 20→10Hz, 임계값)
└── 04-backbones-and-constraints/ 백본 선택 + L40S/60분/오프라인 제약

env/                           ★ 다기기 재현 실험 환경 (팀 온보딩)
├── README.md                     온보딩 가이드
├── configs/                      Stage별 실험 설정 YAML (지표 계약 포함)
│   ├── common.yaml                   디바이스/AMP/시간예산/오프라인 규약
│   ├── stage1.yaml / stage2.yaml / stage3.yaml
│   └── requirements-train.txt        평가서버 기설치본과 핀 정합
├── docker/                       평가 서버 근사 CUDA 12.8 이미지 + compose
└── scripts/                      check_env.py(자기진단)·setup_gpu.sh·run_docker.sh·harness_template.sh
```

---

## 3. 팀원 온보딩 (3분)

```bash
git clone https://github.com/twoimo/dacon-236753-blackbox-accident.git
cd dacon-236753-blackbox-accident
git checkout autoresearch

# (A) 데이터/카탈로그·문서만 볼 때 — GPU 불필요
make setup && make check

# (B) 실제 학습/추론 — CUDA GPU 기기
bash env/scripts/setup_gpu.sh      # conda 또는 venv + torch 2.8.0+cu128
python env/scripts/check_env.py    # 환경 자기진단 (METRIC env_ok=1 이면 통과)

# (C) 평가 서버 근사 컨테이너
bash env/scripts/run_docker.sh              # GPU 컨테이너
bash env/scripts/run_docker.sh --offline    # 인터넷 차단 모사(가중치 동봉 검증)
```

> macOS(Apple Silicon)는 CUDA가 없어 전처리/후처리·라벨 파이프라인 검증만 가능하다.
> 학습·CUDA 추론은 GPU 기기에서 한다(근거: `docs/03` 6절).

---

## 4. Stage별 전략 한 줄 요약 (상세는 `research/0X-*/`)

- **Stage 1 (재녹화)**: 잔차+시간 2-스트림으로 **물리적 재촬영 흔적**(모아레·엣지·잔상) 학습,
  **두 클래스 동일 재인코딩**으로 코덱 누설 차단, **leave-one-device-out** 검증.
- **Stage 2 (사고 시점)**: CCD 프레임 0/1 주석으로 **충돌 시간 국소화** 직접 학습(가장 확실),
  진입/방향/회피는 추적+기하 **약지도** 구성.
- **Stage 3 (거동)**: **20→10Hz 재샘플 정합** 먼저, ego-motion 특징 + 옵티컬 플로우 베이스라인,
  **조향 임계값 스윕**. comma2k19 전체로 라벨 확장.
- **백본/제약**: 60분 예산이 지배 → 베이스라인 `mvit_v2_s` 대비 **X3D** 경량화 병행,
  `weights=None`+가중치 동봉으로 인터넷 차단 대응.

---

## 5. 근거 신빙성 원칙 (이 브랜치의 차별점)

많은 팀이 무검증 SOTA 주장을 좇다 무너진다. 이 브랜치는:

1. 핵심 근거를 **최상위 학회/저널 + 고인용**으로만 채택. 정본은 `research/references/README.md`.
2. 인용수는 **플랫폼(GS/S2/OpenAlex/IEEE)별로 다르므로 출처를 병기**하고 자릿수로 판단.
3. 약탈적 저널·저자 자체 보고 성능만 있는 자료는 **배제**, 저인용 논문은 "문제 정의"용으로 격리.
4. arxiv.org / alphaxiv.org 를 1차 확인 창구로 사용.

### 검증된 핵심 앵커 논문 (인용수 실측 병기)

| 역할 | 논문 | 게재처 | 인용수 (출처) |
|---|---|---|---|
| Stage 2 데이터 원조 | Bao et al. CCD | ACM MM 2020 | 약 235(S2)/315(ACM DL) |
| Stage 2 정초 | Chan et al. DSA | ACCV 2016 | 약 410 (GS) |
| Stage 3 ego-motion 정초 | Zhou et al. SfMLearner | CVPR 2017 (Oral) | 약 2,321(IEEE)/2,869(S2) |
| Stage 3 self-sup depth | Godard et al. Monodepth2 | ICCV 2019 | 수천 회 (GS) |
| Stage 3 스케일 일관성 | Bian et al. SC-SfMLearner | NeurIPS 2019 | 수백 회 (GS) |
| 백본(경량) | Feichtenhofer X3D | CVPR 2020 (Oral) | 약 1,880 (IEEE CSDL) |
| 백본(베이스라인) | Li et al. MViTv2 | CVPR 2022 | 1,000+ (GS) |
| Stage 1 재촬영 포렌식 | Thongkamwitoon et al. | IEEE TIFS 2015 | 약 115 (GS) |
| Stage 3 데이터 원조 | comma2k19 | arXiv 2018 (comma.ai) | 수백 회 (GS) |

---

## 6. 실험 브랜치와의 분리 (협업 규약)

- **`autoresearch` (이 브랜치)** = 자료조사·전략·환경만. 모델 코드를 넣지 않는다.
- **실제 실험** = 여기서 분기한 별도 브랜치에서.
  ```bash
  git checkout -b exp/stage3-egomotion autoresearch
  ```
- 각 실험은 `env/configs/*.yaml` 의 설정과 하니스 계약(`METRIC <name>=<value>`)을 따르고,
  baseline/keep/discard 규율로 기록한다(`env/scripts/harness_template.sh` 참고).

---

## 7. 리더보드 1등 로드맵 (권장 순서)

1. 환경 온보딩 검증(`env/`, 모든 기기 동일 재현).
2. **Stage 3 라벨 파이프라인 정확화**(20→10Hz, 조향 부호/임계값) — 가중치 0.4, 라벨 경로 명확.
3. **Stage 2 충돌 국소화**(CCD 주석 직접 지도) — 가중치 0.4의 핵심 축.
4. **Stage 1 코덱-무누설 재촬영 합성 + LODO 검증** — 가중치 0.2 안정 확보.
5. Stage 2 진입/방향/회피 약지도 개선 + Stage 3 임계값 스윕.
6. 추론 시간 프로파일링 → 60분 내 최대 정확도 앙상블.
7. 2차 평가용 '모델 개발 보고서' + '학습데이터 구성 보고서' 초안을 실험과 동시에 축적.

---

## 8. autoresearch verdict (요약)

- **status**: 근거 수집 완료, 실행 전략 확정. 실제 리더보드 점수는 미측정(이 브랜치는 리서치 전용, 비종결).
- **evidence**: 각 Stage 최상위 학회/저널 논문 + 데이터 원조 논문, 인용수 병기(`research/references/`).
- **caveats**:
  - Stage 2 `entry/evasion/side` 는 공식 라벨 부재 → 약지도 품질이 성능 상한을 결정.
  - Stage 3 조향 임계값(±1.0°)은 대회 정답 규칙 미공개로 추정치 → 스윕 필수.
  - 재녹화 예제는 진짜 재촬영본이 아님(합성 도메인 갭이 최대 리스크).
  - 비공개 평가 데이터 분포는 직접 확인 불가 → 모든 결론은 공개 예제/스펙/유래 기반 추론.
- **evaluator**: autoresearch (self-issued). 실험 브랜치에서 별도 크리틱 패스 권장.

---

## 9. 커밋 이력 (이 브랜치)

- `docs(autoresearch)`: 근거 기반 문헌 리서치 + 다기기 실험 환경 구축 (research/ + env/, 21 파일).
- `review(autoresearch)`: 독립 점검(arxiv/alphaxiv 대조)으로 인용수 정밀화 (Bao CCD 235~315, Zeng ~120 + arXiv ID).
- `fix(autoresearch)`: 문서 깨진 글자(mojibake) 2건 수정.

---

## 10. 참고 (투명성)

가재코드 `gjc autoresearch` CLI는 이 실행 환경에서 TTY/런타임 게이팅으로 headless 실행이 되지 않아,
스킬의 **방법론**(웹 문헌 + 로컬 데이터/제약 교차 → 구조화된 verdict)을 그대로 적용해 산출물을
`research/` 트리에 남겼다. autoresearch 원칙상 `.gjc/` 내부 상태는 직접 편집하지 않으며,
팀 공유 대상은 `research/` 와 `env/` 다.
