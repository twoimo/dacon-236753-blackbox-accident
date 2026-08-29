# 참고문헌 마스터 색인 (Reference Master Index)

DACON 236753 (블랙박스 영상 기반 지능형 고의사고 분석) 리서치의 **근거 정본**이다.
이 파일에 없는 논문은 `research/` 어디에서도 "핵심 근거"로 인용하지 않는다.

## 신빙성 원칙 (Credibility Policy)

많은 논문이 약한 근거 위에 세워진다. 이 프로젝트는 다음을 강제한다.

1. **저명 학회/저널 우선.** CVPR / ICCV / ECCV / NeurIPS / TPAMI / IEEE TIFS / IEEE TIP /
   WACV / ACM MM / ICLR 등 최상위 검증 채널을 1순위로 둔다.
2. **인용수로 신빙성 보증.** 각 논문에 대해 대략적 인용수와 그 출처(Google Scholar /
   Semantic Scholar / IEEE Xplore / OpenAlex)를 병기한다. 지표 간 차이가 크므로 출처를
   반드시 명시하고, 확인 실패 시 `[미검증]` 으로 표기한다.
3. **약탈적/저신뢰 저널 필터링.** MDPI 일부·OMICS·Hilaris 계열 등은 "핵심 근거"에서 제외하고,
   문제 정의·배경 용도로만 제한적으로 쓴다. 사용 시 그 사실을 명시한다.
4. **1차 출처 우선.** arXiv 원문 + 최종 게재본(학회/저널)을 함께 확인한다.
   arxiv.org / alphaxiv.org 를 1차 확인 창구로 쓴다.

> 인용수는 조회 시점(2026-08-29)의 근사치다. Google Scholar > Semantic Scholar > OpenAlex
> 순으로 값이 크게 나오는 경향이 있으므로, 절대값이 아니라 **자릿수(order of magnitude)** 로
> 신빙성을 판단한다.

---

## Tier S — 대회 데이터의 직접 근거 (반드시 읽을 것)

| Key | 논문 | 학회/저널 | arXiv | 인용수 (출처) | 관련 |
|---|---|---|---|---|---|
| `bao2020ccd` | Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning — Bao, Yu, Kong | **ACM MM 2020** | 2008.00334 | 수백 회 (GS) | Stage 2 (CCD 데이터 원조) |
| `schafer2018comma2k19` | A Commute in Data: The comma2k19 Dataset — Schafer, Santana, Haden, Biasini | arXiv 2018 (comma.ai 기술보고) | 1812.05752 | 수백 회 (GS) | Stage 3 (데이터 원조) |
| `chan2016anticipating` | Anticipating Accidents in Dashcam Videos — Chan, Chen, Xiang, Sun | **ACCV 2016** | — | **약 410 (GS)** | Stage 2 (분야 정초 논문) |

- `bao2020ccd`: 우리가 쓰는 **Car Crash Dataset(CCD)** ��� 만든 논문. Cascade R-CNN으로 위험
  영역 후보를 뽑고 GCN+RNN+베이지안 신경망으로 사고확률 시계열과 불확실성을 함께 예측한다.
  프레임별 0/1 주석이 Stage 2 충돌시점 정의의 근거다.
- `schafer2018comma2k19`: 33시간·2,019 세그먼트 캘리포니아 280 고속도로 주행 데이터. 카메라 +
  9축 IMU + CAN + raw GNSS. Stage 3 라벨(가감속·조향)의 CAN 정답 출처.
- `chan2016anticipating`: dashcam 사고 예측 분야를 연 논문(DAD 데이터셋). 사고 약 2초 전
  예측을 보고. 인용 약 410회로 이 분야 최다 인용 축에 속한다.

---

## Tier A — 백본 (검증된 고인용, 60분/44.7GB 제약에 적합)

| Key | 논문 | 학회 | arXiv | 인용수 (출처) | 비고 |
|---|---|---|---|---|---|
| `feichtenhofer2020x3d` | X3D: Expanding Architectures for Efficient Video Recognition | **CVPR 2020 (Oral)** | 2004.04730 | **약 1,880 (computer.org/GS)** | 초경량, 추론시간 유리 |
| `fan2021mvit` | Multiscale Vision Transformers (MViT v1) | **ICCV 2021** | 2104.11227 | 1,000+ (GS) | 베이스라인 백본 계열 |
| `li2022mvitv2` | MViTv2: Improved Multiscale Vision Transformers | **CVPR 2022** | 2112.01526 | 1,000+ (GS) | 베이스라인 `mvit_v2_s` 근거 |
| `liu2021videoswin` | Video Swin Transformer | **CVPR 2022** (arXiv 2021) | 2106.13230 | 1,000+ (GS) | 지역성 inductive bias |
| `kondratyuk2021movinets` | MoViNets: Mobile Video Networks | **CVPR 2021** | 2103.11511 | 수백 회 (GS) | 스트리밍/저메모리 추론 |

- 대회 베이스라인이 `mvit_v2_s` 를 Stage 1·3에서 쓰므로 `li2022mvitv2` / `fan2021mvit` 가
  1차 근거. 추론 60분 제약 때문에 **X3D(-S/-M)** 로 경량화하는 경로를 병행 검토한다.

---

## Tier A — Stage 1 (재녹화/재촬영 포렌식)

| Key | 논문 | 학회/저널 | arXiv | 인용수 (출처) | 역할 |
|---|---|---|---|---|---|
| `thongkamwitoon2015recapture` | Image Recapture Detection Based on Learning Dictionaries of Edge Profiles | **IEEE TIFS 2015** | — | ~100+ (GS) / 74 (OpenAlex) | 재촬영=엣지/MTF 변화 |
| `sun2018moire` | Moiré Photo Restoration Using Multiresolution CNN | **IEEE TIP 2018** | 1805.02996 | ~150 (OpenAlex) | 모아레의 물리·데이터 |
| `patel2015moire` | Live vs Spoof Face: Moiré Patterns to Detect Replay Video Attacks | **ICB 2015** | — | ~100 (OpenAlex) | 영상 모아레 단서 |
| `wangfarid2006double` | Exposing Digital Forgeries in Video by Detecting Double MPEG Compression | **ACM MM&Sec 2006** | — | ~267 (OpenAlex) | 이중압축=코덱 누설 위험 |
| `dai2022videodemoire` | Video Demoiréing with Relation-Based Temporal Consistency | **CVPR 2022** | 2204.02957 | ~50 (OpenReview) | 재촬영=시공간 아티팩트 |
| `jiang2019hevc` | Detection of HEVC Double Compression With Same Coding Parameters | **IEEE TIFS 2019** | — | ~40 (OpenAlex) | 동일 파라미터도 흔적 남음 |
| `yang2017laplacian` | Recapture Image Forensics Based on Laplacian CNN | IWDW 2017 (LNCS) | — | ~66 (OpenAlex) | 고역통과 잔차 스트림 |
| `luo2021sadg` | Scale Invariant Domain Generalization Image Recapture Detection | ICONIP 2021 (LNCS) | 2110.03496 | 낮음(문제정의용) | 도메인 시프트 붕괴 경고 |

> `luo2021sadg` 는 인용수가 낮으므로 "고인용 근거"가 아니라 **문제 정의(도메인 일반화 붕괴)**
> 용도로만 인용한다. 이 구분을 지키는 것이 신빙성 원칙의 핵심이다.

**제외/강등 목록 (핵심 근거로 쓰지 않음):**
- Mahdian et al., *J. Forensic Research* 2015 (cyclostationary LCD): OMICS/Hilaris 계열 저널 → 제외.
- Mehta et al., *SIVP* 2025 (Swin DG recapture): 인용 극소·중위 저널 → 포인터로만.

---

## Tier A — Stage 3 (단안 ego-motion / depth)

| Key | 논문 | 학회 | arXiv | 인용수 (출처) | 역할 |
|---|---|---|---|---|---|
| `zhou2017sfmlearner` | Unsupervised Learning of Depth and Ego-Motion from Video (SfMLearner) | **CVPR 2017 (Oral)** | 1704.07813 | **~2,321 (IEEE) / ~2,869 (S2)** | self-sup ego-motion 정초 |
| `godard2019monodepth2` | Digging Into Self-Supervised Monocular Depth Estimation (Monodepth2) | **ICCV 2019** | 1806.01260 | 수천 회 (GS) | 멀티스케일·오클루전 처리 |
| `bian2019scsfm` | Unsupervised Scale-Consistent Depth and Ego-Motion (SC-SfMLearner) | **NeurIPS 2019** | 1908.10553 | 수백 회 (GS) | 스케일 일관성 |

- Stage 3는 CAN 없이 **영상만으로** 가감속·조향을 추정한다. self-supervised depth+pose 계열이
  ego-motion(속도·회전) 추정의 검증된 골격이다. 다만 대회는 **범주 분류**(4-class accel / 3-class
  steer)이므로, pose regression을 그대로 쓰기보다 특징 추출기로 활용 + 분류 헤드를 얹는다.

---

## Tier B — Stage 2 (시간 국소화 / 사고 예측 확장)

| Key | 논문 | 학회/저널 | arXiv | 인용수 (출처) | 역할 |
|---|---|---|---|---|---|
| `liao2024accnet` | Real-time Accident Anticipation via Monocular Depth-Enhanced 3D Modeling (AccNet) | **Accident Analysis & Prevention 2024** | — | ~45 (GS) | depth 보강 3D 모델링 |
| `zeng2017agentcentric` | Agent-Centric Risk Assessment: Accident Anticipation and Risky Region Localization | **CVPR 2017** | — | 수백 회 (GS) | 공간+시간 위험 국소화 |

- 대회 Stage 2는 사고 "예측"이 아니라 이미 일어난 사고의 **충돌/진입 프레임 국소화**에 가깝다.
  따라서 사고 예측(anticipation) 문헌은 특징 설계·주석 정의 참고용이고, 실제 과업은 프레임 단위
  시계열 회귀/분류(temporal localization)에 가깝다는 점을 유의한다.

---

## 인용 표기 규칙

- 본문에서는 `bao2020ccd` 같은 **Key** 로 인용한다.
- 새 논문을 추가할 때는 반드시 이 표의 형식(학회/저널·arXiv·인용수·출처)을 채운다.
- 인용수를 확인하지 못하면 `[미검증]` 으로 남기고 절대 지어내지 않는다.
