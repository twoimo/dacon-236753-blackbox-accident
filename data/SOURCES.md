# 공개 예제 출처

- Stage 1·2 원본: CCD(Car Crash Dataset), `Crash-1500.zip`의 000001~000005
  - https://github.com/Cogito2012/CarCrashDataset
- Stage 1 재녹화 예제: 위 공개 원본 5건에 화면 재촬영 시 나타날 수 있는 리샘플링·노이즈 특성을 적용한 베이스라인용 파생본
- Stage 2 충돌 구간: CCD 공식 `Crash-1500.txt` 주석의 첫 positive frame
- Stage 3: comma2k19 공개 Example 데이터에서 생성한 5개 주행 클립과 공개 CAN 기반 라벨
  - https://github.com/commaai/comma2k19

Stage 2의 `-1`은 공개 정답이 없어 학습 손실에서 제외되는 항목입니다.
