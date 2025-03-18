#!/bin/sh

# conda 환경 활성화
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mlx-ui-env

# 애플리케이션 실행
streamlit run app.py -- "$@"
