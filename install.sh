#!/bin/sh

# conda 초기화 확인 및 실행
if [ ! -f "$HOME/.conda/environments.txt" ]; then
    echo '⏳ Initializing conda...'
    conda init zsh
    source "$HOME/.zshrc"
fi

# remove existing conda env
if [ -d "mlx-ui-env" ]; then
    echo '⏳ Recreating conda env..'
    conda deactivate
    conda env remove -n mlx-ui-env
fi

# create conda env with specific python version
echo '⏳ Creating conda environment...'
conda create -n mlx-ui-env python=3.10 -y

# activate conda env
echo '⏳ Activating conda environment...'
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mlx-ui-env

if [ "$1" = "refresh" ]; then
    echo '⏳ Refreshing requirements..'

    # install deps with specific versions to avoid conflicts
    conda install -c conda-forge -y \
        numpy=1.24.3 \
        pandas=2.0.3 \
        tiktoken=0.5.1 \
        sentencepiece=0.1.99 \
        streamlit=1.24.0 \
        watchdog=3.0.0

    # MLX는 conda-forge에서 제공하지 않으므로 pip로 설치
    pip install mlx-lm

    # requirements.txt 생성
    pip freeze > requirements.txt
else
    # install deps
    echo '⏳ Installing requirements..'
    pip install -r requirements.txt
fi

echo '✅ Installation complete. You can use ./run.sh to launch the app'
