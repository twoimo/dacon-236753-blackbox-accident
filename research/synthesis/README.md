# 종합 전략 & 오토리서치 Verdict (Synthesis)

`autoresearch` 미션의 최종 종합. 개별 Stage 근거는 `research/0X-*/README.md`,
논문 신빙성은 `research/references/README.md` 참조.

---

## 0. 미션 개요 (autoresearch 방식)

- **목표**: DACON 236753 리더보드 1등을 위한, 근거 기반 모델 개발 전략 수립.
- **mode**: `mixed` (웹 문헌 조사 + 로컬 데이터/제약 분석 상호교차).
- **산출물**: 이 브랜치의 `research/` 문헌·전략 + `env/` 재현 환경. **제품 코드는 실험 브랜치에서.**
- **원칙**: 모든 기술 주장은 중첩 근거(논문 + 대회 데이터/스펙)로 뒷받침. 저명 학회/저널 +
  인용수로 신빙성 필터링. 약탈적/무검증 자료 배제.

## 1. 점수 배분과 우선순위

| Stage | 과업 | 가중치 | 라벨 상황 | 난이도/확실성 |
|---|---|---|---|---|
| 1 | 재녹화 판별 (Macro-F1) | **0.2** | 합성 필요 | 중 (코덱 누설만 피하면 안정) |
| 2 | 사고 시점·상황 | **0.4** | 충돌시점만 있음 | collision 확실 / 나머지 약지도 |
| 3 | 가감속·조향 | **0.4** | comma2k19 파생 | 임계값·프레임률이 관건 |

**투자 우선순위 (기대 점수 × 확실성):**
1. **Stage 2 collision_frame** — 0.4 비중 + 라벨 확보됨 → 가장 확실한 점수원.
2. **Stage 3 프레임률/임계값 정합** — 0.4 비중, 여기서 틀리면 대량 오답.
3. **Stage 1** — 0.2지만 코덱 누설만 방어하면 저비용 고안정.
4. Stage 2의 entry/evasion/side — 약지도, 보고서용 실험 기록 병행.

## 2. Stage별 한 줄 결론 (근거)

- **Stage 1**: 물리적으로 재촬영한 dashcam 클립으로 **잔차+시간 2-스트림** 분류기 학습,
  **두 클래스 동일 랜덤 재인코딩**, **leave-one-device-out** 평가.
  근거: `thongkamwitoon2015recapture`(엣지), `sun2018moire`(모아레), `dai2022videodemoire`(시공간),
  `wangfarid2006double`·`jiang2019hevc`(코덱 누설 위험).
- **Stage 2**: CCD 프레임 0/1 주석으로 **시간 국소화**(peak/change-point)로 collision 학습,
  나머지 3항목은 추적+기하 **약지도**. 근거: `bao2020ccd`(데이터), `chan2016anticipating`,
  `zeng2017agentcentric`.
- **Stage 3**: **20→10Hz 재샘플 정합** 먼저, self-supervised **ego-motion 특징**
  (`zhou2017sfmlearner`, `godard2019monodepth2`, `bian2019scsfm`) + 옵티컬 플로우 베이스라인,
  **조향 임계값 스윕**. comma2k19 전체로 라벨 확장.
- **백본/제약**: 60분 예산이 지배. 베이스라인 `mvit_v2_s`(`li2022mvitv2`) 대비 **X3D**
  (`feichtenhofer2020x3d`) 경량화 병행. `weights=None`+가중치 동봉으로 인터넷 차단 대응.

## 3. 공통 리스크 (전 Stage 관통)

1. **분포 시프트**: 공개 예제 ≠ 비공개 평가 분포. In-domain 지표 과신 금지 → 홀드아웃/교차검증.
2. **누설(leakage)**: 코덱(S1), 프레임 재번호(S2), 프레임률(S3), 파일 ID 기반 split.
3. **제출 포맷 오답**: 범주 화이트리스트/정수화/범위 clip 후처리 필수 (제출오류는 횟수 차감).
4. **시간 예산**: 3-Stage 합산 60분. 시간 프로파일링을 지표로 상시 추적.

## 4. autoresearch Verdict

```json
{
  "status": {
    "disposition": "conclusive",
    "scope": "research-strategy-only",
    "downstream": "implementation goes to a separate experiment branch"
  },
  "evidence": [
    "Stage 1: recapture forensics는 엣지/모아레/시공간 아티팩트 근거가 최상위 저널(TIFS/TIP)+CVPR로 확립; 코덱 누설 위험은 double-compression 문헌(TIFS 2019 등)이 입증",
    "Stage 2: 대회 데이터(CCD)는 bao2020ccd(ACM MM 2020)에서 유래, 충돌시점 라벨 확보 → 시간 국소화가 확실한 점수원; entry/evasion/side는 라벨 부재로 약지도",
    "Stage 3: ego-motion은 self-supervised depth+pose(SfMLearner CVPR17 ~2.3k+ cites, Monodepth2 ICCV19) 골격; comma2k19가 데이터·라벨 원천; 20→10Hz/임계값이 최대 함정",
    "백본: X3D(CVPR20 ~1.9k cites)/MViTv2(CVPR22)가 60분·44.7GB 제약에 부합; 인터넷 차단은 weights=None+가중치 동봉으로 해결"
  ],
  "caveats": [
    "entry_frame/evasion_space/entry_side는 CCD에 정답 라벨이 없어 약지도로만 구성 가능 — 성능 상한 불확실",
    "Stage 3 조향 임계값(±1.0°)은 대회 정답 규칙 미공개로 추정치 — 튜닝 없이는 분포가 크게 어긋남",
    "비공개 평가 데이터 분포를 직접 확인 불가 — 모든 결론은 공개 예제/스펙/유래 기반 추론",
    "일부 재촬영 검출 도메인일반화 논문(luo2021sadg 등)은 인용수 낮아 문제정의용으로만 사용"
  ],
  "evaluator": "autoresearch-mission-agent (Aside)",
  "confidence": "high (strategy) / medium (label-scarce subtasks)"
}
```

## 5. 다음 단계 (팀 온보딩)
1. `env/README.md` 로 환경 세팅(도커/conda) — 여러 기기 재현.
2. 실제 실험은 **별도 실험 브랜치**에서 (`autoresearch`는 자료/전략 공유 전용).
3. 각 실험은 `METRIC <name>=<value>` 로그와 baseline/keep/discard 규율로 기록.
