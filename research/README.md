# research/ — 문헌 리서치 & 전략 (autoresearch 브랜치)

이 폴더는 DACON 236753 리더보드 1등을 목표로 한 **근거 기반 리서치 정본**이다.
`가재코드 /skill:autoresearch` 워크플로 방식(웹 문헌 + 로컬 데이터/제약을 교차하여 구조화된
verdict로 마무리)을 따랐다.

> **범위 경계**: 이 브랜치는 **자료·전략·재현환경 공유 전용**이다. 실제 모델 코드/실험은
> **별도 실험 브랜치**에서 수행한다. autoresearch 산출물은 findings + evidence + verdict이지
> 제품 코드가 아니다.

## 읽는 순서 (팀원 온보딩)

1. **`references/README.md`** — 모든 근거의 정본. 논문 신빙성(학회/저널 등급 + 인용수) 판정 규칙.
   여기 없는 논문은 "핵심 근거"로 인용하지 않는다.
2. **`synthesis/README.md`** — 종합 전략, 점수 우선순위, autoresearch 최종 verdict.
3. Stage별 상세:
   - `01-stage1-recapture/` — 재녹화 판별 (가중치 0.2)
   - `02-stage2-anticipation/` — 사고 시점·상황 (가중치 0.4)
   - `03-stage3-egomotion/` — 가감속·조향 (가중치 0.4)
   - `04-backbones-and-constraints/` — 백본 선택 + L40S/60분/오프라인 제약

## 신빙성 원칙 (요약)

- 최상위 학회(CVPR/ICCV/ECCV/NeurIPS)·저널(TPAMI/TIFS/TIP/TITS) + 충분한 인용수 = 최우선 근거.
- 인용수는 플랫폼(GS/S2/OpenAlex)별로 다르므로 **출처를 병기**하고 자릿수로 판단.
- 약탈적 저널·무검증 성능 수치는 배제. 저인용 논문은 "문제 정의"용으로만 제한 사용.
- arxiv.org / alphaxiv.org 를 1차 확인 창구로 사용.

## 대회 핵심 사실 (근거: `docs/`, `catalog/`)

- 3-Stage 단일 코드 제출. 가중치 0.2 / 0.4 / 0.4.
- 평가 서버: NVIDIA L40S 44.7GiB, 7 vCPU, 60GB RAM, **추론 60분**, **인터넷 차단**.
- **공식 학습셋 미제공** → 법적 제한 없는 데이터 자체 구성(CCD, comma2k19 등).
- 상위 15팀 → 2차 평가('모델 개발 보고서' + '학습데이터 구성 보고서').

## 이 리서치를 실험으로 잇는 법

각 Stage 문서 끝의 "검증 가능한 실험 후크"는 `env/configs/*.yaml` 의 설정 키와 하니스가
출력할 `METRIC <name>=<value>` 라인을 정의한다. 실험 브랜치에서 이 계약을 그대로 구현하면
리서치 → 실험이 매끄럽게 연결된다.
