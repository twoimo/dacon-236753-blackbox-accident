# DACON 236753 — 블랙박스 영상 기반 지능형 고의사고 분석
#
# 주요 흐름
#   make setup           도구용 가상환경 구성
#   make data            (클론 직후) 구글 드라이브에서 대용량 데이터 복원
#   make labels          라벨 재생성 (stage2 복원 + comma2k19 파생)
#   make stage2-images   Stage2 평가 레이아웃 프레임 이미지 생성
#   make catalog         기계판독 카탈로그 재생성
#   make verify          카탈로그 대조 검증
#   make check           verify + catalog 무결성 요약 (CI 용)
#   make release         배포 아카이브 생성
#   make publish         구글 드라이브 업로드 + 공개 링크 생성

SHELL := /bin/bash
PY    := .venv/bin/python
ROOT  := $(shell pwd)

.DEFAULT_GOAL := help
.PHONY: help setup data labels stage2-images catalog catalog-fast verify verify-full check release publish clean-derived tree

help:
	@grep -E '^[a-zA-Z][a-zA-Z0-9_-]*:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1;36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## 도구용 가상환경(.venv) 구성
	python3.11 -m venv .venv 2>/dev/null || python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements-tools.txt
	@$(PY) -c "import numpy,PIL;print('ok: numpy',numpy.__version__,'pillow',PIL.__version__)"

data: ## 구글 드라이브 공개 링크에서 대용량 데이터 복원
	bash scripts/fetch_data.sh

labels: ## 라벨 재생성 (stage2 복원 + comma2k19 파생)
	$(PY) scripts/make_stage2_labels.py
	$(PY) scripts/make_stage3_labels_from_comma2k19.py

stage2-images: ## Stage2 평가서버 입력 레이아웃(프레임 이미지 폴더) 생성
	$(PY) scripts/make_stage2_images.py

catalog: ## 카탈로그 전체 재생성 (sha256 포함, ~1분)
	$(PY) scripts/build_catalog.py

catalog-fast: ## 카탈로그 재생성 (sha256 생략)
	$(PY) scripts/build_catalog.py --skip-hash

verify: ## 카탈로그 대조 검증 (크기 기준, CrashBest 표본 2000장)
	$(PY) scripts/verify_integrity.py --sample 2000

verify-full: ## 카탈로그 대조 검증 (sha256 전수)
	$(PY) scripts/verify_integrity.py --hash

check: verify ## 검증 + 무결성 요약 출력
	@$(PY) -c "import json;d=json.load(open('catalog/integrity.json'));print('무결성:',d['summary']);\
[print(f\"  [{f['severity']}] {f['kind']}: {f['detail'][:150]}\") for f in d['findings'] if f['severity'] in ('error','warning')]"

release: ## 배포 아카이브 생성 (dist/)
	bash scripts/package_release.sh

publish: release ## 구글 드라이브 업로드 + 공개 링크 생성
	bash scripts/gdrive_upload.sh

clean-derived: ## 재생성 가능한 산출물 삭제
	rm -rf data/stage2/images dist

tree: ## 데이터 레이아웃 요약
	@find data -maxdepth 3 \
	  -not -path 'data/external/CrashBest/*' \
	  -not -path 'data/external/comma2k19/*' | sort
