#!/bin/bash

# R1-Omni Inference Automation Script (v1.2 - Cleaned based on user feedback)
# -----------------------------------------------------------------
# Assumes R1-Omni environment is already set up via a separate script.
# Processes video segments from S3, runs inference, and stores results on S3.
# NOTE: Uses original S3 folder structure (no dates in folder names).

# --- Configuration ---
# S3 Paths
S3_BUCKET="fomcthesiss3"
S3_ANALYSIS_BASE="s3://${S3_BUCKET}/fomc_analysis"
S3_SEGMENTED_CLIPS_BASE_DIR="${S3_ANALYSIS_BASE}/06_segmented_clips"      # Input MP4s (e.g., .../clips/VIDEO_ID/)
S3_METADATA_BASE_DIR="${S3_ANALYSIS_BASE}/07_metadata"                    # Input JSONs (e.g., .../metadata/VIDEO_ID/)
S3_INFERENCE_RESULTS_BASE_DIR="${S3_ANALYSIS_BASE}/08_inference_results"  # Output JSONs (e.g., .../results/VIDEO_ID/)

# Local Paths & R1-Omni Setup (MUST MATCH your R1-Omni setup script)
# Use tilde expansion explicitly or use $HOME
R1_OMNI_SETUP_BASE_DIR="$HOME/models"                                # Base dir used in setup script
R1_OMNI_REPO_DIR="${R1_OMNI_SETUP_BASE_DIR}/R1-Omni"                 # Path to the cloned R1-Omni repository
R1_OMNI_WEIGHTS_DIR="${R1_OMNI_SETUP_BASE_DIR}/R1-Omni/R1-Omni-0.5B" # Path to R1-Omni weights
CONDA_ENV_NAME="r1omni_env"
CONDA_BASE_DIR="$HOME/miniconda"                                     # Or $(conda info --base)

# Scratch space - Using ${RANDOM} for better uniqueness as suggested
BASE_TEMP_DIR="/tmp/r1_omni_inference_${RANDOM}_$$"

# Inference Parameters - Using single quotes for INFERENCE_PROMPT
INFERENCE_PROMPT='As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you? Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags.'
SKIP_EXISTING_INFERENCE=true
INFERENCE_TIMEOUT_SECONDS=600 # Timeout for python inference helper script (10 minutes)

# --- Helper Python Script Filename (will be created by this Bash script) ---
HELPER_PYTHON_SCRIPT="${BASE_TEMP_DIR}/helper_run_and_parse_inference.py"

# --- Script Start ---
echo "### R1-Omni Inference Automation Script (v1.2) ###"
SECONDS=0
# Create base temp dir, exit if fails
mkdir -p "$BASE_TEMP_DIR" || { echo "ERROR: Failed to create base temporary directory '$BASE_TEMP_DIR'. Check permissions or path."; exit 1; }

# Setup trap for cleanup
trap 'echo ""; echo ">>> Cleaning up temporary directory: $BASE_TEMP_DIR"; rm -rf "$BASE_TEMP_DIR"; echo ">>> Script finished after $SECONDS seconds."; exit' EXIT SIGINT SIGTERM

# 1. Create Helper Python Script
# (Using a heredoc to write the python script content. Single quotes around 'EOF_PYTHON' prevent shell expansion within the heredoc)
cat << 'EOF_PYTHON' > "${HELPER_PYTHON_SCRIPT}"
import subprocess
import json
import argparse
import os
import sys
import re # For more robust tag parsing

def parse_r1_output(raw_output_str):
    think_text = "NOT_FOUND"
    answer_text = "NOT_FOUND"
    # Use regex for case-insensitive and flexible tag searching
    try:
        # DOTALL allows . to match newlines, IGNORECASE makes tags case-insensitive
        think_match = re.search(r"<think>(.*?)</think>", raw_output_str, re.IGNORECASE | re.DOTALL)
        if think_match:
            think_text = think_match.group(1).strip()

        answer_match = re.search(r"<answer>(.*?)</answer>", raw_output_str, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
    except Exception as e:
        # Keep default "NOT_FOUND" but log the parsing error
        print(f"Warning: Error parsing inference tags: {e}", file=sys.stderr)
    return think_text, answer_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helper to run R1-Omni inference and structure output.")
    parser.add_argument("--python_exec_for_inference", required=True, help="Path to Python interpreter in the correct conda env.")
    parser.add_argument("--inference_script_path", required=True, help="Path to R1-Omni/inference.py.")
    parser.add_argument("--model_weights_path", required=True, help="Path to R1-Omni model weights directory.")
    parser.add_argument("--local_video_segment_path", required=True, help="Local path to the downloaded video segment.")
    parser.add_argument("--local_segment_metadata_path", required=True, help="Local path to the downloaded segment metadata JSON.")
    parser.add_argument("--instruct_prompt", required=True, help="Instruction prompt for the model.")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds for the inference subprocess.")
    args = parser.parse_args()

    segment_metadata = {}
    inference_successful = False
    raw_inference_output = "Error: Inference did not run." # Default message

    # 1. Load segment metadata JSON
    try:
        with open(args.local_segment_metadata_path, 'r') as f:
            segment_metadata = json.load(f)
        print(f"Helper: Loaded metadata for {os.path.basename(args.local_segment_metadata_path)}", file=sys.stderr)
    except Exception as e:
        print(f"Helper ERROR: Failed to load metadata from {args.local_segment_metadata_path}: {e}", file=sys.stderr)
        # Output error JSON and exit to signal failure to Bash script
        # Ensure the output is valid JSON even on error
        print(json.dumps({
            "input_segment_info": {"metadata_path": args.local_segment_metadata_path},
            "inference_details": {"inference_successful": False, "raw_output": f"Metadata load failed: {e}"}
        }))
        sys.exit(1)

    # 2. Construct and run R1-Omni's inference.py command
    try:
        inference_cmd = [
            args.python_exec_for_inference,
            args.inference_script_path,
            "--modal", "video_audio",
            "--model_path", args.model_weights_path,
            "--video_path", args.local_video_segment_path,
            "--instruct", args.instruct_prompt
        ]
        print(f"Helper: Running command: {' '.join(inference_cmd)}", file=sys.stderr)

        process_env = os.environ.copy() # Inherit environment for subprocess

        process = subprocess.run(inference_cmd,
                                 capture_output=True,
                                 text=True,
                                 check=True,
                                 timeout=args.timeout,
                                 env=process_env)
        raw_inference_output = process.stdout
        print(f"Helper: Inference completed successfully for {os.path.basename(args.local_video_segment_path)}.", file=sys.stderr)
        inference_successful = True

    except subprocess.CalledProcessError as e:
        raw_inference_output = f"Inference CPERR. Exit Code: {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}"
        print(f"Helper ERROR: Inference command failed for {os.path.basename(args.local_video_segment_path)}.", file=sys.stderr)
        print(raw_inference_output, file=sys.stderr)
    except subprocess.TimeoutExpired:
        raw_inference_output = f"Inference TIMEOUT after {args.timeout}s."
        print(f"Helper ERROR: Inference timed out for {os.path.basename(args.local_video_segment_path)}.", file=sys.stderr)
    except Exception as e:
        raw_inference_output = f"Inference EXCEPTION: {str(e)}"
        print(f"Helper ERROR: An unexpected exception occurred during inference for {os.path.basename(args.local_video_segment_path)}: {e}", file=sys.stderr)

    # 3. Parse raw_output for <think> and <answer>
    think_text, answer_text = parse_r1_output(raw_inference_output)

    # 4. Construct final JSON result object
    final_result = {
        "input_segment_info": {
            "segment_s3_path": segment_metadata.get("s3_clip_path", "UNKNOWN"),
            "metadata_s3_path": segment_metadata.get("s3_metadata_path", "UNKNOWN"),
            "segment_filename": os.path.basename(args.local_video_segment_path),
            "source_video_id": segment_metadata.get("source_video_id", "UNKNOWN"),
            "video_date": segment_metadata.get("video_date_yyyymmdd", "UNKNOWN"),
            "segment_start_s": segment_metadata.get("segment_start_time_s"),
            "segment_end_s": segment_metadata.get("segment_end_time_s"),
            "segment_duration_s": segment_metadata.get("segment_duration_s"),
            "original_turn_start_s": segment_metadata.get("original_turn_start_s"),
            "original_turn_end_s": segment_metadata.get("original_turn_end_s"),
        },
        "inference_details": {
            "model_weights_path": args.model_weights_path,
            "prompt": args.instruct_prompt,
            "inference_successful": inference_successful,
            "raw_output": raw_inference_output,
            "parsed_think": think_text,
            "parsed_answer": answer_text
        }
    }
    # 5. Print final structured JSON to stdout
    print(json.dumps(final_result, indent=2))

    # 6. Exit with appropriate status code
    sys.exit(0 if inference_successful else 1)

EOF_PYTHON

# Check if helper script was created and has content
if [ ! -s "${HELPER_PYTHON_SCRIPT}" ]; then
  echo "ERROR: Failed to create helper python script at ${HELPER_PYTHON_SCRIPT} or it is empty. Exiting."
  exit 1
fi
chmod +x "${HELPER_PYTHON_SCRIPT}" # Not strictly necessary if calling via python exec, but good practice

# 2. Environment Setup
echo "Activating conda environment: $CONDA_ENV_NAME..."
# Try sourcing directly, then check if conda command exists if sourcing fails
if ! source "${CONDA_BASE_DIR}/etc/profile.d/conda.sh" 2>/dev/null; then
  if ! command -v conda &> /dev/null; then
    echo "ERROR: Could not source conda.sh and 'conda' command not found. Is Miniconda/Anaconda installed and initialized?"
    exit 1
  else
    echo "INFO: Failed to source conda.sh directly. Will rely on 'conda activate' finding the environment."
  fi
fi
# Attempt activation
conda activate "$CONDA_ENV_NAME" || { echo "ERROR: Failed to activate conda env '$CONDA_ENV_NAME'. Did setup_r1_omni.sh run correctly?"; exit 1; }
PYTHON_EXEC=$(which python3) # Get python exec path *after* activating env
if [ -z "$PYTHON_EXEC" ] || ! command -v "$PYTHON_EXEC" &>/dev/null; then
  echo "ERROR: Could not find python3 executable after activating conda environment '$CONDA_ENV_NAME'."
  exit 1
fi
echo "Using Python: $PYTHON_EXEC ($($PYTHON_EXEC --version))"

# Resolve potential ~ in paths now that shell is set up
R1_OMNI_REPO_DIR=$(eval echo "$R1_OMNI_REPO_DIR")
R1_OMNI_WEIGHTS_DIR=$(eval echo "$R1_OMNI_WEIGHTS_DIR")

# Verify paths again after resolving tilde
if [ ! -d "$R1_OMNI_REPO_DIR" ]; then echo "ERROR: R1-Omni repo not found at '$R1_OMNI_REPO_DIR'."; exit 1; fi
INFERENCE_SCRIPT_FULL_PATH="${R1_OMNI_REPO_DIR}/inference.py"
if [ ! -f "$INFERENCE_SCRIPT_FULL_PATH" ]; then echo "ERROR: Cannot find inference script at '$INFERENCE_SCRIPT_FULL_PATH'."; exit 1; fi
if [ ! -d "$R1_OMNI_WEIGHTS_DIR" ]; then echo "ERROR: R1-Omni weights not found at '$R1_OMNI_WEIGHTS_DIR'."; exit 1; fi
echo "R1-Omni paths verified."

# Set environment variables for inference subprocesses
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} # Ensure this is set as per your setup
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
echo "Environment variables set (TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)."

# 3. Discover Processed Videos on S3
echo "Discovering video folders in S3 bucket: ${S3_SEGMENTED_CLIPS_BASE_DIR}/ ..."
# List directories (PRE prefixes ending in /) at the specified level
# The trailing / is important to only list folders
s3_video_folders_raw=$(aws s3 ls "${S3_SEGMENTED_CLIPS_BASE_DIR}/" --region us-east-1 | awk '{print $2}' | grep '/$')
if [ -z "$s3_video_folders_raw" ]; then
  echo "WARNING: No video folders (e.g., VIDEO_ID/) found under '${S3_SEGMENTED_CLIPS_BASE_DIR}/'. Processing will stop."
  exit 0
fi
# Extract just the folder names (video IDs) - remove trailing slash
mapfile -t video_folders < <(echo "$s3_video_folders_raw" | sed 's|/$||')
echo "Found ${#video_folders[@]} video folders to process."
if [ ${#video_folders[@]} -eq 0 ]; then echo "INFO: No video folders parsed from S3 listing."; exit 0; fi

# --- Statistics ---
total_videos_processed=0
total_segments_found=0
total_segments_processed_successfully=0
total_segments_skipped=0
total_segments_failed_inference=0

# 4. Process Each Video's Segments
echo "--- Starting Segment Processing ---"
for video_id in "${video_folders[@]}"; do
  # Basic check if video_id looks valid (alphanumeric, hyphen, underscore) - prevents processing weird folder names
  if [[ ! "$video_id" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo " WARN: Skipping invalid folder name found in S3 listing: '$video_id'"
    continue
  fi

  echo "Processing video: ${video_id}"
  total_videos_processed=$((total_videos_processed + 1))
  current_video_segments_found=0
  current_video_segments_processed=0
  current_video_segments_skipped=0
  current_video_segments_failed=0

  # Define S3 paths for this specific video (using original non-dated structure)
  s3_clip_dir="${S3_SEGMENTED_CLIPS_BASE_DIR}/${video_id}"
  s3_meta_dir="${S3_METADATA_BASE_DIR}/${video_id}"
  s3_output_dir="${S3_INFERENCE_RESULTS_BASE_DIR}/${video_id}" # Output goes into a subfolder per video ID

  # Create local temp directory for this video's files
  local_video_temp_dir="${BASE_TEMP_DIR}/${video_id}"
  mkdir -p "$local_video_temp_dir" || { echo " WARN: Failed to create temp dir '$local_video_temp_dir', skipping video ${video_id}."; continue; }

  # List segments for this video
  s3_segments_raw=$(aws s3 ls "${s3_clip_dir}/" --region us-east-1 | grep '\.mp4$')
  if [ -z "$s3_segments_raw" ]; then
    echo " No .mp4 segments found in ${s3_clip_dir}/. Skipping video."
    rm -rf "$local_video_temp_dir" # Clean up empty temp dir
    continue
  fi
  mapfile -t segments < <(echo "$s3_segments_raw" | awk '{print $NF}')
  total_segments_found=$((total_segments_found + ${#segments[@]}))
  current_video_segments_found=${#segments[@]}
  echo " Found ${current_video_segments_found} segments for ${video_id}."

  for segment_filename in "${segments[@]}"; do
    # Construct paths
    s3_segment_path="${s3_clip_dir}/${segment_filename}"
    base_segment_name=$(basename "$segment_filename" .mp4)
    metadata_filename="${base_segment_name}.json"
    s3_metadata_path="${s3_meta_dir}/${metadata_filename}"
    output_filename="${base_segment_name}_inference.json"
    s3_output_path="${s3_output_dir}/${output_filename}"
    local_segment_path="${local_video_temp_dir}/${segment_filename}"
    local_metadata_path="${local_video_temp_dir}/${metadata_filename}"
    local_output_path="${local_video_temp_dir}/${output_filename}" # Local path for the JSON output

    echo "  Processing segment: ${segment_filename}"

    # Check if output already exists on S3 (Skip Logic)
    if [ "$SKIP_EXISTING_INFERENCE" = true ]; then
      if aws s3 ls "$s3_output_path" --region us-east-1 &>/dev/null; then
        echo "    INFO: Output already exists on S3: ${s3_output_path}. Skipping."
        total_segments_skipped=$((total_segments_skipped + 1))
        current_video_segments_skipped=$((current_video_segments_skipped + 1))
        continue # Skip to next segment
      fi
    fi

    # Download segment MP4
    echo "    Downloading segment: ${s3_segment_path}"
    if ! aws s3 cp "$s3_segment_path" "$local_segment_path" --only-show-errors --region us-east-1; then
      echo "    ERROR: Failed to download segment ${s3_segment_path}. Skipping segment."
      total_segments_failed_inference=$((total_segments_failed_inference + 1))
      current_video_segments_failed=$((current_video_segments_failed + 1))
      continue
    fi

    # Download metadata JSON
    echo "    Downloading metadata: ${s3_metadata_path}"
    if ! aws s3 cp "$s3_metadata_path" "$local_metadata_path" --only-show-errors --region us-east-1; then
      echo "    ERROR: Failed to download metadata ${s3_metadata_path}. Skipping segment."
      rm -f "$local_segment_path" # Clean up downloaded segment
      total_segments_failed_inference=$((total_segments_failed_inference + 1))
      current_video_segments_failed=$((current_video_segments_failed + 1))
      continue
    fi

    # Run Inference via Helper Python Script
    echo "    Running inference helper..."
    inference_json_output=$("$PYTHON_EXEC" "${HELPER_PYTHON_SCRIPT}" \
      --python_exec_for_inference "$PYTHON_EXEC" \
      --inference_script_path "$INFERENCE_SCRIPT_FULL_PATH" \
      --model_weights_path "$R1_OMNI_WEIGHTS_DIR" \
      --local_video_segment_path "$local_segment_path" \
      --local_segment_metadata_path "$local_metadata_path" \
      --instruct_prompt "$INFERENCE_PROMPT" \
      --timeout "$INFERENCE_TIMEOUT_SECONDS")
    helper_exit_code=$?

    # Check helper script execution and output validity
    if [ $helper_exit_code -ne 0 ] || [ -z "$inference_json_output" ] || ! echo "$inference_json_output" | python3 -m json.tool > /dev/null 2>&1 ; then
      # Use python itself to validate JSON, it's usually available
      echo "    ERROR: Helper script failed (Exit Code: $helper_exit_code) or produced invalid/empty JSON output for segment ${segment_filename}."
      echo "    --- Helper Stdout/Stderr Start ---"
      echo "$inference_json_output" # Print potential error JSON or empty string
      # Note: Stderr from the helper is printed by the helper script itself if it encounters issues
      echo "    --- Helper Stdout/Stderr End ---"
      total_segments_failed_inference=$((total_segments_failed_inference + 1))
      current_video_segments_failed=$((current_video_segments_failed + 1))
    else
      # Save valid JSON locally before upload
      echo "$inference_json_output" > "$local_output_path"

      # Upload result to S3
      echo "    Uploading result: ${s3_output_path}"
      if ! aws s3 cp "$local_output_path" "$s3_output_path" --only-show-errors --region us-east-1; then
        echo "    ERROR: Failed to upload inference result ${s3_output_path}. Result saved locally: ${local_output_path}"
        total_segments_failed_inference=$((total_segments_failed_inference + 1))
        current_video_segments_failed=$((current_video_segments_failed + 1))
        # Keep local file in case of upload failure for debugging
      else
        echo "    Segment processing successful."
        total_segments_processed_successfully=$((total_segments_processed_successfully + 1))
        current_video_segments_processed=$((current_video_segments_processed + 1))
        rm -f "$local_output_path" # Clean up local output file ONLY if upload succeeded
      fi
    fi

    # Clean up downloaded input files for this segment in all cases (success or failure of this segment)
    rm -f "$local_segment_path" "$local_metadata_path"
  done # End segment loop

  echo "  Finished video ${video_id}. Segments: ${current_video_segments_found} | Processed OK: ${current_video_segments_processed} | Skipped: ${current_video_segments_skipped} | Failed: ${current_video_segments_failed}"
  # Clean up the video-specific temp directory now that all its segments are done or attempted
  rm -rf "$local_video_temp_dir"
done < <(printf '%s\n' "${video_folders[@]}") # Safely loop through video_folders array

# 6. Final Summary
echo ""
echo "# --- Overall Summary ---"
echo "# Total video folders found in S3: ${#video_folders[@]}"
echo "# Total video folders for which processing was attempted: ${total_videos_processed}"
echo "#"
echo "# Total segments found across processed folders: ${total_segments_found}"
echo "# Total segments processed successfully (inference ran & uploaded): ${total_segments_processed_successfully}"
echo "# Total segments skipped (output already existed on S3): ${total_segments_skipped}"
echo "# Total segments failed (download, metadata, inference error, or upload error): ${total_segments_failed_inference}"
echo "#"
echo "# Inference results stored under: ${S3_INFERENCE_RESULTS_BASE_DIR}/<VIDEO_ID>/<SEGMENT_NAME>_inference.json"
echo "-------------------------------------"

exit 0