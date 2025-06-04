#!/bin/bash

# R1-Omni Setup and Inference Script (v9 - Targeted Patches v2)
# -----------------------------------------------------------------
# v9: Uses more targeted sed patterns for patching humanomni_arch.py.

# --- Configuration Variables ---
BASE_DIR=~/models
HF_MODEL_DIR=~/hf_models # IMPORTANT: Replace with your actual desired path!
R1_OMNI_WEIGHTS_DIR=$BASE_DIR/R1-Omni/R1-Omni-0.5B
VIDEO_DIR=~/videos
VIDEO_FILE=fedtry.mp4 # IMPORTANT: Replace with your actual video file name!
CONDA_ENV_NAME=r1omni_env
PYTHON_VERSION=3.11
# Define expected CUDA version
EXPECTED_CUDA_MAJOR_MINOR="12.4"

# --- Prerequisite Checks ---
echo "# Checking prerequisites..."
# Check for nvcc in PATH
if ! command -v nvcc &> /dev/null; then
    echo "##############################################################################"
    echo "# ERROR: nvcc (NVIDIA CUDA Compiler) not found in PATH."
    echo "#"
    echo "# Please Copy the ENTIRE block below (from 'sudo apt-get --purge...' to 'source ~/.bashrc && \\')"
    echo "# and paste it into your terminal to install CUDA Toolkit ${EXPECTED_CUDA_MAJOR_MINOR} and set up the environment:"
    echo "# --- Start Copy ---"
    echo "# Part 1: Install CUDA Toolkit ${EXPECTED_CUDA_MAJOR_MINOR}"
    echo "sudo apt-get --purge remove \"*cublas*\" \"*cufft*\" \"*curand*\" \"*cusolver*\" \"*cusparse*\" \"*npp*\" \"*nvjpeg*\" \"cuda*\" \"nsight*\" -y && \\"
    echo "sudo apt-get autoremove -y && \\"
    echo "sudo apt-get clean && \\"
    echo "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \\"
    echo "sudo dpkg -i cuda-keyring_1.1-1_all.deb && \\"
    echo "sudo apt-get update && \\"
    echo "sudo apt-get install cuda-toolkit-${EXPECTED_CUDA_MAJOR_MINOR//./-} -y && \\"
    echo "rm cuda-keyring_1.1-1_all.deb && \\"
    echo "echo '# CUDA Toolkit installation finished.'"
    echo ""
    echo "# Part 2: Add CUDA paths to ~/.bashrc for future sessions (if not already present)"
    echo "grep -qxF 'export PATH=/usr/local/cuda-${EXPECTED_CUDA_MAJOR_MINOR}/bin:\$PATH' ~/.bashrc || echo 'export PATH=/usr/local/cuda-${EXPECTED_CUDA_MAJOR_MINOR}/bin:\$PATH' >> ~/.bashrc && \\"
    echo "grep -qxF 'export LD_LIBRARY_PATH=/usr/local/cuda-${EXPECTED_CUDA_MAJOR_MINOR}/lib64:\$LD_LIBRARY_PATH' ~/.bashrc || echo 'export LD_LIBRARY_PATH=/usr/local/cuda-${EXPECTED_CUDA_MAJOR_MINOR}/lib64:\$LD_LIBRARY_PATH' >> ~/.bashrc && \\"
    echo "echo 'Environment configuration added to ~/.bashrc (if needed).'"
    echo ""
    echo "# Part 3: Apply environment changes to the CURRENT shell session"
    echo "source ~/.bashrc && \\"
    echo "echo 'Environment changes applied to current session.'"
    echo "# --- End Copy ---"
    echo "#"
    echo "# After successfully running the block above, re-run this setup script:"
    echo "# ./setup_r1_omni.sh"
    echo "##############################################################################"
    exit 1
fi
# Get CUDA path from nvcc location
NVCC_PATH=$(which nvcc)
DETECTED_CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
echo "# Found nvcc: $NVCC_PATH"
echo "# Detected CUDA installation: $DETECTED_CUDA_HOME"
# Check CUDA version (optional but recommended)
NVCC_VERSION_OUTPUT=$(nvcc --version)
if ! echo "$NVCC_VERSION_OUTPUT" | grep -q "release ${EXPECTED_CUDA_MAJOR_MINOR}"; then
    echo "##############################################################################"
    echo "WARNING: Found CUDA Toolkit, but it might not be version ${EXPECTED_CUDA_MAJOR_MINOR}."
    echo "$NVCC_VERSION_OUTPUT"
    echo "Script will proceed, but compatibility issues might arise."
    echo "##############################################################################"
fi
# Set CUDA environment variables for this script's execution
export CUDA_HOME=$DETECTED_CUDA_HOME
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
echo "# Using CUDA_HOME=$CUDA_HOME for script execution."

# --- 0. Install System Prerequisites ---
echo "# Installing prerequisites (git, wget, build-essential)..."
sudo apt-get update -y && sudo apt-get install -y git wget build-essential || { echo "ERROR: Failed to install prerequisites."; exit 1; }

# --- 1. Install Miniconda (if not present) ---
if ! command -v conda &> /dev/null; then
    echo "# Conda not found. Downloading and installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $HOME/miniconda && \
    rm ~/miniconda.sh || { echo "ERROR: Miniconda installation failed."; exit 1; }
    export PATH="$HOME/miniconda/bin:$PATH"
    echo "# Initializing Conda for your shell..."
    conda init bash
    echo "# Sourcing conda hook for current session (post-install)..."
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
else
    echo "# Conda found. Skipping installation."
    echo "# Sourcing conda hook for current session..."
    eval "$($HOME/miniconda/bin/conda shell.bash hook)" || eval "$($(conda info --base)/bin/conda shell.bash hook)"
fi
conda activate base || { echo "ERROR: Failed to activate base conda env."; exit 1; }

# --- 2. Create/Recreate and Activate Python Environment ---
echo "# Checking/Creating Conda environment $CONDA_ENV_NAME..."
conda env remove -n $CONDA_ENV_NAME -y # Remove unconditionally for clean slate
conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y || { echo "ERROR: Failed to create conda env $CONDA_ENV_NAME."; exit 1; }
# Use source activate for better script compatibility
source activate $CONDA_ENV_NAME || conda activate $CONDA_ENV_NAME || { echo "ERROR: Failed to activate $CONDA_ENV_NAME."; exit 1; }

# Define paths within the env
PYTHON_EXEC=$HOME/miniconda/envs/$CONDA_ENV_NAME/bin/python3
PIP_EXEC=$HOME/miniconda/envs/$CONDA_ENV_NAME/bin/pip
HUGGINGFACE_CLI_EXEC=$HOME/miniconda/envs/$CONDA_ENV_NAME/bin/huggingface-cli
echo "# Python version: $($PYTHON_EXEC --version 2>&1)"

# --- 3. Clone Repositories ---
echo "# Cloning R1-V and R1-Omni repositories into $BASE_DIR..."
mkdir -p $BASE_DIR
cd $BASE_DIR || { echo "ERROR: Failed to cd to $BASE_DIR."; exit 1; }
git clone https://github.com/Deep-Agent/R1-V.git || { echo "ERROR: Failed to clone R1-V."; exit 1; }
git clone https://github.com/HumanMLLM/R1-Omni.git || { echo "ERROR: Failed to clone R1-Omni."; exit 1; }

# --- [REFINED v9] Patch Files ---
echo "# Patching library files for local paths..."

# Patch inference.py for local BERT tokenizer path
INFERENCE_PY_PATH="$BASE_DIR/R1-Omni/inference.py"
if [ -f "$INFERENCE_PY_PATH" ]; then
    if ! grep -q "BertTokenizer.from_pretrained(bert_model_path)" "$INFERENCE_PY_PATH"; then # Check if specific change exists
        echo "# Backing up $INFERENCE_PY_PATH to ${INFERENCE_PY_PATH}.bak"
        cp "$INFERENCE_PY_PATH" "${INFERENCE_PY_PATH}.bak"
        echo "# Applying inference.py patches..."
        sed -i 's|bert_model = "bert-base-uncased"|bert_model_path = "'"$HF_MODEL_DIR"'/bert-base-uncased" # Patched by setup script|' "$INFERENCE_PY_PATH"
        sed -i 's|BertTokenizer.from_pretrained(bert_model)|BertTokenizer.from_pretrained(bert_model_path) # Patched by setup script|' "$INFERENCE_PY_PATH"
        echo "# Patched $INFERENCE_PY_PATH for tokenizer."
    else
        echo "# $INFERENCE_PY_PATH already patched."
    fi
else
    echo "ERROR: $INFERENCE_PY_PATH not found. Cannot patch."
    exit 1
fi

# Patch humanomni_arch.py for local BERT model and tokenizer paths
ARCH_PY_PATH="$BASE_DIR/R1-Omni/humanomni/model/humanomni_arch.py"
if [ -f "$ARCH_PY_PATH" ]; then
     if ! grep -q "from_pretrained(config.text_model_path)" "$ARCH_PY_PATH"; then # Check if specific change exists
        echo "# Backing up $ARCH_PY_PATH to ${ARCH_PY_PATH}.bak"
        cp "$ARCH_PY_PATH" "${ARCH_PY_PATH}.bak"
        echo "# Applying humanomni_arch.py patches..."
        # Target BertModel line, replace argument within parentheses more precisely
        sed -i 's|\(BertModel.from_pretrained\)(.*)|\1(config.text_model_path)|g' "$ARCH_PY_PATH"
        # Target BertTokenizer line, replace argument within parentheses more precisely
        sed -i 's|\(BertTokenizer.from_pretrained\)(.*)|\1(config.text_model_path)|g' "$ARCH_PY_PATH"
        # Comment out the original hardcoded path assignment if it exists
        sed -i 's|^\(\s*bert_model\s*=\s*"/\?mnt/data/jiaxing.zjx.*\)|# \1 # Commented out by setup script|' "$ARCH_PY_PATH"
        # Verify patches worked
        if ! grep -q "BertModel.from_pretrained(config.text_model_path)" "$ARCH_PY_PATH" || \
           ! grep -q "BertTokenizer.from_pretrained(config.text_model_path)" "$ARCH_PY_PATH"; then
           echo "WARNING: Patching $ARCH_PY_PATH might have failed. Check backup ${ARCH_PY_PATH}.bak and apply manual edits if needed."
        else
           echo "# Patched $ARCH_PY_PATH for BERT model and tokenizer loading."
        fi
    else
        echo "# $ARCH_PY_PATH already patched."
    fi
else
    echo "ERROR: $ARCH_PY_PATH not found. Cannot patch."
    exit 1
fi


# --- 4. Install R1-V Dependency ---
echo "# Installing R1-V in editable mode (CUDA_HOME=$CUDA_HOME)..."
R1V_SRC_DIR=$BASE_DIR/R1-V/src/r1-v
if [ -d "$R1V_SRC_DIR" ]; then
    cd "$R1V_SRC_DIR" || { echo "ERROR: Failed to cd to $R1V_SRC_DIR."; exit 1; }
    echo "# Running R1-V install (pip install -e .[dev])..."
    $PIP_EXEC install -e ".[dev]" || { echo "ERROR: R1-V installation failed."; exit 1; }
    cd $BASE_DIR
else
    echo "ERROR: Directory $R1V_SRC_DIR not found."
    exit 1
fi


# --- 5. Install R1-Omni Dependencies ---
echo "# Installing R1-Omni core dependencies..."
$PIP_EXEC install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124 || { echo "ERROR: Failed to install PyTorch."; exit 1; }
$PIP_EXEC install transformers==4.49.0 || { echo "ERROR: Failed to install transformers."; exit 1; }
echo "# Installing flash-attn..."
$PIP_EXEC install flash-attn --no-build-isolation || { echo "ERROR: flash-attn installation failed."; exit 1; }
echo "# Installing other R1-Omni dependencies..."
$PIP_EXEC install timm moviepy==1.0.3 decord einops accelerate scipy huggingface_hub opencv-python-headless packaging sentencepiece ipdb h5py || { echo "ERROR: Failed to install other dependencies."; exit 1; }


# --- 6. Download Required Models ---
echo "# Downloading HF models to $HF_MODEL_DIR..."
mkdir -p $HF_MODEL_DIR
if [ -f "$HUGGINGFACE_CLI_EXEC" ]; then
    CLI_CMD="$HUGGINGFACE_CLI_EXEC"
else
    echo "# WARNING: huggingface-cli not found at $HUGGINGFACE_CLI_EXEC."
    CLI_CMD="huggingface-cli"
fi
$CLI_CMD download --resume-download openai/whisper-large-v3 --local-dir $HF_MODEL_DIR/whisper-large-v3 --local-dir-use-symlinks False || { echo "ERROR: Failed to download Whisper."; exit 1; }
$CLI_CMD download --resume-download google/siglip-base-patch16-224 --local-dir $HF_MODEL_DIR/siglip-base-patch16-224 --local-dir-use-symlinks False || { echo "ERROR: Failed to download SigLip."; exit 1; }
$CLI_CMD download --resume-download bert-base-uncased --local-dir $HF_MODEL_DIR/bert-base-uncased --local-dir-use-symlinks False || { echo "ERROR: Failed to download BERT."; exit 1; }
echo "# Downloading R1-Omni weights to $R1_OMNI_WEIGHTS_DIR..."
mkdir -p $R1_OMNI_WEIGHTS_DIR
$CLI_CMD download --resume-download StarJiaxing/R1-Omni-0.5B --local-dir $R1_OMNI_WEIGHTS_DIR --local-dir-use-symlinks False || { echo "ERROR: Failed to download R1-Omni weights."; exit 1; }


# --- 7. Configure Model Paths in R1-Omni Config ---
CONFIG_JSON_PATH="$R1_OMNI_WEIGHTS_DIR/config.json"
echo "# Updating $CONFIG_JSON_PATH..."
if [ ! -f "$CONFIG_JSON_PATH" ]; then
    echo "ERROR: $CONFIG_JSON_PATH not found."
    exit 1
fi
cp "$CONFIG_JSON_PATH" "${CONFIG_JSON_PATH}.bak"
cat << EOF > "$CONFIG_JSON_PATH"
{
  "_name_or_path": "StarJiaxing/R1-Omni-0.5B",
  "architectures": [
    "HumanOmniQwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "audio_hidden_size": 1280,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "freeze_mm_mlp_adapter": false,
  "hidden_act": "silu",
  "hidden_size": 896,
  "hyper_layers": [
    15,
    22
  ],
  "image_aspect_ratio": "pad",
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 32768,
  "max_window_layers": 21,
  "mm_audio_projector_type": "mlp2x_gelu",
  "mm_audio_tower": "$HF_MODEL_DIR/whisper-large-v3",
  "mm_hidden_size": 768,
  "mm_projector_lr": null,
  "mm_projector_type": "all_in_one_small",
  "mm_tunable_parts": "mm_mlp_adapter,audio_projector,mm_language_model",
  "mm_use_x_start_end": true,
  "mm_vision_select_feature": "patch",
  "mm_vision_select_layer": -2,
  "mm_vision_tower": "$HF_MODEL_DIR/siglip-base-patch16-224",
  "model_type": "HumanOmni_qwen2",
  "num_attention_heads": 14,
  "num_frames": 8,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "text_model_path": "$HF_MODEL_DIR/bert-base-uncased",
  "tie_word_embeddings": true,
  "tokenizer_model_max_length": 2048,
  "tokenizer_padding_side": "right",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.49.0",
  "tune_mm_mlp_adapter": false,
  "use_cache": true,
  "use_local_models": true,
  "use_mm_proj": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
EOF
echo "# config.json updated."


# --- 8. Prepare Video File ---
echo "# Ensuring video directory $VIDEO_DIR exists..."
mkdir -p $VIDEO_DIR
if [ ! -f "$VIDEO_DIR/$VIDEO_FILE" ]; then
    echo "# WARNING: Video file $VIDEO_DIR/$VIDEO_FILE not found. Please upload it."
    # Consider exiting if the video is essential for the script's purpose beyond inference test
    # exit 1
fi


# --- 9. Set Environment Variables ---
echo "# Setting environment variables..."
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_VERBOSITY=error # Suppress HF loading warnings


# --- 10. Run Inference ---
echo "# Running inference on $VIDEO_FILE..."
cd $BASE_DIR/R1-Omni || { echo "ERROR: Failed to cd to $BASE_DIR/R1-Omni."; exit 1; }
if [ ! -f "$VIDEO_DIR/$VIDEO_FILE" ]; then
    echo "ERROR: Video file $VIDEO_DIR/$VIDEO_FILE not found. Cannot run inference."
    exit 1
fi

$PYTHON_EXEC inference.py --modal video_audio \
  --model_path $R1_OMNI_WEIGHTS_DIR \
  --video_path $VIDEO_DIR/$VIDEO_FILE \
  --instruct "As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you? Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags." \
  || { echo "ERROR: Inference script failed."; exit 1; }

echo "# Inference script finished successfully."
echo "# Setup complete!"

# --- End of Script ---