#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=PAS1957
#SBATCH --output=output/%j.log
#SBATCH --mail-type=FAIL
#SBATCH --ntasks=48
#SBATCH --gpus-per-node=2
#SBATCH --gpu_cmode=exclusive
#SBATCH -p gpu
#SBATCH -N 1



ENV_DIR=.openvoiceenv

# Remove the old environment
conda env remove -n ${ENV_DIR}

# Create a new conda environment with Python 3.9
conda create -y -n ${ENV_DIR} python=3.9

# Activate the conda environment
source activate ${ENV_DIR}

# Ensure pip, setuptools, and wheel are upgraded
pip install --upgrade pip setuptools wheel

# Install the package in editable mode
pip install -e .

# Download Unidic dictionary
python -m unidic download

# Install required packages
pip install \
    librosa==0.9.1 \
    faster-whisper==0.9.0 \
    pydub==0.25.1 \
    wavmark==0.0.3 \
    numpy==1.22.0 \
    eng_to_ipa==0.0.2 \
    inflect==7.0.0 \
    unidecode==1.3.7 \
    whisper-timestamped==1.14.2 \
    openai \
    python-dotenv \
    pypinyin==0.50.0 \
    cn2an==0.5.22 \
    jieba==0.42.1 \
    gradio==3.48.0 \
    langid==1.1.6 \
    txtsplit \
    torch \
    torchaudio \
    cached_path \
    transformers==4.27.4 \
    mecab-python3==1.0.5 \
    num2words==0.5.12 \
    unidic_lite==1.0.8 \
    unidic==1.1.0 \
    pykakasi==2.2.1 \
    fugashi==1.3.0 \
    g2p_en==2.1.0 \
    anyascii==0.3.2 \
    jamo==0.4.1 \
    gruut[de,es,fr]==2.2.3 \
    g2pkk>=0.1.1 \
    tqdm \
    tensorboard==2.16.2 \
    loguru==0.7.2

# If needed, uncomment these lines to install additional packages
# pip install pytorch_lightning
# pip install OmegaConf
# pip install unidecode
# pip install pypinyin
# pip install inflect