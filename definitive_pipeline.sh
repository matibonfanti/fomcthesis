#!/bin/bash

SECONDS=0 # Start timer

# --- Configuration ---
S3_BUCKET="fomcthesiss3"
YT_COOKIE_FILE_PATH="$HOME/youtube_cookies.txt"
FASTER_WHISPER_MODEL="large-v3"
MAX_PARALLEL_JOBS=4
YTDL_FORMAT='bv*[vcodec*=avc1][height<=720]+ba[acodec*=mp4a]/best[height<=720][vcodec*=avc1][ext=mp4]'
WHISPER_MODEL_NAME="large-v3"

# --- Parameters for Segmentation ---
TARGET_SEGMENT_DURATION_S=20
MIN_SEGMENT_DURATION_S=10
MAX_SEGMENT_DURATION_S=35
MAX_MERGE_GAP_S=1.5

# --- Paths ---
S3_BASE_PATH="s3://${S3_BUCKET}/fomc_analysis"
S3_RAW_VIDEO_DIR="${S3_BASE_PATH}/01_raw_videos"; S3_RAW_AUDIO_DIR="${S3_BASE_PATH}/02_raw_audio"; S3_TRANSCRIPT_DIR="${S3_BASE_PATH}/03_transcripts"; S3_DIARIZATION_DIR="${S3_BASE_PATH}/04_diarization";
S3_SEGMENTED_CLIPS_DIR="${S3_BASE_PATH}/06_segmented_clips"; S3_METADATA_DIR="${S3_BASE_PATH}/07_metadata"
BASE_LOCAL_TEMP_DIR="/tmp/fomc_processing_parallel_v7.13_$$"
WHISPER_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/whisper"

# --- Environment ---
CONDA_ENV_NAME="r1omni_env"
_CONDA_BASE_GUESS=$(conda info --base 2>/dev/null || echo "$HOME/miniconda")
PYTHON_EXEC="$_CONDA_BASE_GUESS/envs/$CONDA_ENV_NAME/bin/python3"

# --- Files ---
LOCAL_VIDEO_LIST="$HOME/video_list.txt"

# Export variables for subshells
export S3_BUCKET S3_BASE_PATH S3_RAW_VIDEO_DIR S3_RAW_AUDIO_DIR S3_TRANSCRIPT_DIR S3_DIARIZATION_DIR S3_SEGMENTED_CLIPS_DIR S3_METADATA_DIR
export BASE_LOCAL_TEMP_DIR PYTHON_EXEC FASTER_WHISPER_MODEL WHISPER_CACHE_DIR WHISPER_MODEL_NAME
export TARGET_SEGMENT_DURATION_S MIN_SEGMENT_DURATION_S MAX_SEGMENT_DURATION_S MAX_MERGE_GAP_S
export YT_COOKIE_FILE_PATH YTDL_FORMAT MAX_PARALLEL_JOBS


# --- Function to Parse YouTube Video ID ---
parse_video_id() {
    local url="$1"; local id=""
    if [[ "$url" =~ youtube\.com/watch\?v=([^&/]+) ]]; then id="${BASH_REMATCH[1]}"
    elif [[ "$url" =~ youtu\.be/([^&/?]+) ]]; then id="${BASH_REMATCH[1]}"
    elif [[ "$url" =~ ^[a-zA-Z0-9_-]{11}$ ]]; then id="$url"; fi
    id=$(echo "$id" | head -c 11)
    if [[ "${#id}" -eq 11 ]]; then echo "$id"; else echo ""; fi
}
export -f parse_video_id

###############################################################################
# process_video — end-to-end pipeline for a single YouTube target
###############################################################################
process_video() {
    local video_target="$1"; local job_num="$2"; local total_jobs="$3";
    local video_id; local video_date="NODATE"; local local_video_path=""
    local stage_failed=0; local COOKIE_ARG=""
    local job_start_seconds=$SECONDS

    echo "[Job $job_num/$total_jobs] START Target: $video_target (Subshell PID: $BASHPID)" # Use BASHPID

    if [ -f "$YT_COOKIE_FILE_PATH" ]; then COOKIE_ARG="--cookies $YT_COOKIE_FILE_PATH"; fi

    video_id=$(parse_video_id "$video_target")
    if [ -z "$video_id" ]; then
        echo "[Job $job_num/$total_jobs - $video_target - ERROR] Could not parse valid Video ID. Skipping."
        return 1
    fi

    ############################################################################
    # NEW BLOCK A — EARLY EXIT IF SEGMENTS ALREADY EXIST
    # -------------------------------------------------------------------------
    # Set SKIP_VIDEO_IF_SEGMENTS_EXIST=false to disable this optimisation.
    ############################################################################
    if [ "${SKIP_VIDEO_IF_SEGMENTS_EXIST:-true}" = "true" ]; then
        local s3_clip_subfolder_check="${S3_SEGMENTED_CLIPS_DIR}/${video_id}"
        if aws s3 ls "${s3_clip_subfolder_check}/" --region us-east-1 | grep -q '\.mp4$'; then
            echo "[Job $job_num/$total_jobs - $video_id] Segmented clips already present in ${s3_clip_subfolder_check}/ — skipping ENTIRE video."
            return 0
        fi
    fi
    # -------------------------------------------------------------------------

    local job_temp_dir="${BASE_LOCAL_TEMP_DIR}/job_${video_id}";
    if ! mkdir -p "$job_temp_dir"/{raw_videos,raw_audio,transcripts,diarization,segmented_clips,metadata}; then
        echo "[Job $job_num/$total_jobs - $video_id - ERROR] Failed create job temp dir '$job_temp_dir'."
        return 1
    fi

    local local_video_dir="${job_temp_dir}/raw_videos"
    local local_audio_dir="${job_temp_dir}/raw_audio"
    local local_transcript_dir="${job_temp_dir}/transcripts"
    local local_diarization_dir="${job_temp_dir}/diarization"
    local local_segmented_clips_dir="${job_temp_dir}/segmented_clips"
    local local_metadata_dir="${job_temp_dir}/metadata"
    local python_log_file="${job_temp_dir}/segmentation_raw_output.log"

    echo "[Job $job_num/$total_jobs - $video_id] Retrieving video upload date..."
    video_date_output=$(yt-dlp --print "%(upload_date)s" $COOKIE_ARG "$video_target" 2>/dev/null); exit_code_date=$?
    if [ $exit_code_date -eq 0 ] && [ -n "$video_date_output" ] && [[ "$video_date_output" =~ ^[0-9]{8}$ ]]; then
        video_date="$video_date_output"
        echo "[Job $job_num/$total_jobs - $video_id] Video Date: $video_date"
    else
        video_date="NODATE"
    fi

    if [ "$video_date" == "NODATE" ]; then
        echo "[Job $job_num/$total_jobs - $video_id - ERROR] Upload date NOT FOUND for video '$video_target'."
        echo "[Job $job_num/$total_jobs - $video_id - ACTION_REQUIRED] Possible YouTube cookie issue (e.g., expired/invalid)."
        echo "[Job $job_num/$total_jobs - $video_id - SUGGESTION] Check/update cookie file: '$YT_COOKIE_FILE_PATH' and retry video."
        if [ -d "$job_temp_dir" ]; then
            rm -rf "$job_temp_dir"
            echo "[Job $job_num/$total_jobs - $video_id] Cleaned temp dir: $job_temp_dir"
        fi
        return 2 # Specific exit code for NODATE failure
    fi

    local_video_path="${local_video_dir}/${video_id}.mp4"
    s3_video_path="${S3_RAW_VIDEO_DIR}/${video_id}.mp4"
    if [ -f "$local_video_path" ] && [ -s "$local_video_path" ]; then
        echo "[Job $job_num/$total_jobs - $video_id] Video found locally."
    elif aws s3 ls "$s3_video_path" --region us-east-1 > /dev/null; then
        echo "[Job $job_num/$total_jobs - $video_id] Video in S3. Downloading..."
        aws s3 cp "$s3_video_path" "$local_video_path" --only-show-errors --region us-east-1 || stage_failed=1
    else
        echo "[Job $job_num/$total_jobs - $video_id] Downloading video (720p)..."
        if [ -z "$COOKIE_ARG" ]; then
            echo "[Job $job_num/$total_jobs - $video_id] WARNING: No cookie file for yt-dlp."
        fi
        yt-dlp -P "$local_video_dir" -o "${video_id}.%(ext)s" $COOKIE_ARG --quiet --no-progress --no-warnings -f "$YTDL_FORMAT" --merge-output-format mp4 "$video_target"
        ytdlp_exit_code=$?
        if [ $ytdlp_exit_code -ne 0 ]; then
            echo "[Job $job_num/$total_jobs - $video_id] ERROR: yt-dlp failed (Code: $ytdlp_exit_code)."
            stage_failed=1
        fi
        if [ $stage_failed -eq 0 ] && [ -s "$local_video_path" ]; then
            echo "[Job $job_num/$total_jobs - $video_id] Download OK. Uploading to S3..."
            aws s3 cp "$local_video_path" "$s3_video_path" --only-show-errors --region us-east-1 || \
                echo "[Job $job_num/$total_jobs - $video_id] WARNING: S3 video upload failed."
        else
            echo "[Job $job_num/$total_jobs - $video_id] ERROR: Video download failed/empty."
            stage_failed=1
            if [ -f "$local_video_path" ]; then rm "$local_video_path"; fi
        fi
    fi
    if [ $stage_failed -eq 1 ] || [ ! -s "$local_video_path" ]; then
        echo "[Job $job_num/$total_jobs - $video_id] ERROR: Video handling failed."
        rm -rf "$job_temp_dir"
        return 1
    fi
    
    local s3_audio_path="${S3_RAW_AUDIO_DIR}/${video_id}.wav"
    local local_audio_path_expected="${local_audio_dir}/${video_id}.wav"
    local extracted_audio_path=""
    if [ -s "$local_audio_path_expected" ]; then
        extracted_audio_path="$local_audio_path_expected"
    elif aws s3 ls "$s3_audio_path" --region us-east-1 > /dev/null; then
        echo "[Job $job_num/$total_jobs - $video_id] Audio in S3. Downloading..."
        aws s3 cp "$s3_audio_path" "$local_audio_path_expected" --only-show-errors --region us-east-1 || stage_failed=1
        extracted_audio_path="$local_audio_path_expected"
    else
        echo "[Job $job_num/$total_jobs - $video_id] Extracting audio..."
        audio_output_full=$($PYTHON_EXEC "${BASE_LOCAL_TEMP_DIR}/extract_audio.py" --video_path "$local_video_path" --audio_dir "$local_audio_dir" 2>&1)
        exit_code=$?
        extracted_audio_path=$(echo "$audio_output_full" | grep "OUTPUT_PATH:" | tail -n 1 | cut -d: -f2)
        if [ $exit_code -ne 0 ] || [ -z "$extracted_audio_path" ] || [ ! -s "$extracted_audio_path" ]; then
            echo "[Job $job_num/$total_jobs - $video_id] ERROR: Audio extraction failed."
            echo "$audio_output_full"
            stage_failed=1
        else
            aws s3 cp "$extracted_audio_path" "$s3_audio_path" --only-show-errors --region us-east-1 || \
                echo "[Job $job_num/$total_jobs - $video_id] WARNING: S3 audio upload failed."
        fi
    fi
    if [ $stage_failed -eq 1 ] || [ ! -s "$extracted_audio_path" ]; then
        echo "[Job $job_num/$total_jobs - $video_id] ERROR: Audio processing invalid."
        rm -rf "$job_temp_dir"
        return 1
    fi

    local s3_transcript_path="${S3_TRANSCRIPT_DIR}/${video_id}_transcript.json"
    local local_transcript_path_expected="${local_transcript_dir}/${video_id}_transcript.json"
    local transcript_path=""
    if [ -s "$local_transcript_path_expected" ]; then
        transcript_path="$local_transcript_path_expected"
    elif aws s3 ls "$s3_transcript_path" --region us-east-1 > /dev/null; then
        echo "[Job $job_num/$total_jobs - $video_id] Transcript in S3. Downloading..."
        aws s3 cp "$s3_transcript_path" "$local_transcript_path_expected" --only-show-errors --region us-east-1 || stage_failed=1
        transcript_path="$local_transcript_path_expected"
    else
        echo "[Job $job_num/$total_jobs - $video_id] Transcribing ($WHISPER_MODEL_NAME)..."
        transcript_output_full=$($PYTHON_EXEC "${BASE_LOCAL_TEMP_DIR}/transcribe_faster_whisper.py" --audio_path "$extracted_audio_path" --transcript_dir "$local_transcript_dir" 2>&1)
        exit_code=$?
        transcript_path=$(echo "$transcript_output_full" | grep "OUTPUT_PATH:" | tail -n 1 | cut -d: -f2)
        if [ $exit_code -ne 0 ] || [ -z "$transcript_path" ] || [ ! -f "$transcript_path" ]; then
            echo "[Job $job_num/$total_jobs - $video_id] ERROR: Transcription failed."
            echo "$transcript_output_full"
            stage_failed=1
        else
            aws s3 cp "$transcript_path" "$s3_transcript_path" --only-show-errors --region us-east-1 || \
                echo "[Job $job_num/$total_jobs - $video_id] WARNING: S3 transcript upload failed."
        fi
    fi
    if [ $stage_failed -eq 1 ] || [ ! -f "$transcript_path" ]; then
        echo "[Job $job_num/$total_jobs - $video_id] ERROR: Transcript processing invalid."
        rm -rf "$job_temp_dir"
        return 1
    fi

    local s3_rttm_path="${S3_DIARIZATION_DIR}/${video_id}.rttm"
    local local_rttm_path_expected="${local_diarization_dir}/${video_id}.rttm"
    local diarization_path=""
    if [ -s "$local_rttm_path_expected" ]; then
        diarization_path="$local_rttm_path_expected"
    elif aws s3 ls "$s3_rttm_path" --region us-east-1 > /dev/null; then
        echo "[Job $job_num/$total_jobs - $video_id] Diarization in S3. Downloading..."
        aws s3 cp "$s3_rttm_path" "$local_rttm_path_expected" --only-show-errors --region us-east-1 || stage_failed=1
        diarization_path="$local_rttm_path_expected"
    else
        echo "[Job $job_num/$total_jobs - $video_id] Diarizing..."
        diarization_output_full=$($PYTHON_EXEC "${BASE_LOCAL_TEMP_DIR}/diarize_audio.py" --audio_path "$extracted_audio_path" --diarization_dir "$local_diarization_dir" 2>&1)
        exit_code=$?
        diarization_path=$(echo "$diarization_output_full" | grep "OUTPUT_PATH:" | tail -n 1 | cut -d: -f2)
        if [ $exit_code -ne 0 ] || [ -z "$diarization_path" ] || [ ! -f "$diarization_path" ]; then
            echo "[Job $job_num/$total_jobs - $video_id] ERROR: Diarization failed."
            echo "$diarization_output_full"
            stage_failed=1
        else
            aws s3 cp "$diarization_path" "$s3_rttm_path" --only-show-errors --region us-east-1 || \
                echo "[Job $job_num/$total_jobs - $video_id] WARNING: S3 RTTM upload failed."
        fi
    fi
    if [ $stage_failed -eq 1 ] || [ ! -f "$diarization_path" ]; then
        echo "[Job $job_num/$total_jobs - $video_id] ERROR: Diarization processing invalid."
        rm -rf "$job_temp_dir"
        return 1
    fi

    ############################################################################
    # NEW BLOCK B — SKIP SEGMENTATION STAGE IF CLIPS ALREADY IN S3
    ############################################################################
    num_segments=0
    if aws s3 ls "${S3_SEGMENTED_CLIPS_DIR}/${video_id}/" --region us-east-1 | grep -q '\.mp4$'; then
        echo "[Job $job_num/$total_jobs - $video_id] Segments already exist in S3 — skipping segmentation stage."
    else
        echo "[Job $job_num/$total_jobs - $video_id] Segmenting (video validation active)..."
        tmp_exit_code_file="${job_temp_dir}/py_exit_code.$$"
        {
            $PYTHON_EXEC "${BASE_LOCAL_TEMP_DIR}/segment_video_v7_9.py" \
                --video_id "$video_id" --video_path "$local_video_path" \
                --transcript_path "$transcript_path" --rttm_path "$diarization_path" \
                --video_date "$video_date" \
                --output_clip_dir "$local_segmented_clips_dir" --output_meta_dir "$local_metadata_dir" \
                2>&1
            echo $? > "$tmp_exit_code_file"
        } | tee "$python_log_file"

        seg_exit_code=$(cat "$tmp_exit_code_file")
        rm "$tmp_exit_code_file"

        if [ "$seg_exit_code" -ne 0 ]; then
            echo "[Job $job_num/$total_jobs - $video_id] ERROR: Segmentation script failed (Code $seg_exit_code). See $python_log_file"
            stage_failed=1
        else
            clip_files=(); meta_files=()
            while IFS=':' read -r _ file_type local_file_path; do
                if [[ $file_type == "CLIP" ]]; then clip_files+=("$local_file_path")
                elif [[ $file_type == "META" ]]; then meta_files+=("$local_file_path"); fi
            done < <(grep '^OUTPUT_PATH:' "$python_log_file")
            num_segments=${#clip_files[@]}
            if [ $num_segments -eq 0 ] && ! grep -q "created 0 valid video segment(s)" "$python_log_file" && ! grep -q "INFO: No Chair turns for" "$python_log_file"; then
                echo "[Job $job_num/$total_jobs - $video_id] INFO: Segmentation script (exit 0) seems to have produced no segments, or Bash failed to parse. Check $python_log_file"
            fi
        fi

        segment_upload_success=1; num_segments_uploaded=0
        if [ $stage_failed -eq 0 ] && [ $num_segments -gt 0 ]; then
            s3_clip_subfolder="${S3_SEGMENTED_CLIPS_DIR}/${video_id}"
            s3_meta_subfolder="${S3_METADATA_DIR}/${video_id}"
            echo "[Job $job_num/$total_jobs - $video_id] Uploading $num_segments validated video segment(s) to S3..."
            for local_meta_file in "${meta_files[@]}"; do
                sed -i "s|__S3_CLIP_DIR__|${s3_clip_subfolder}|g" "$local_meta_file"
                sed -i "s|__S3_META_DIR__|${s3_meta_subfolder}|g" "$local_meta_file"
            done
            for local_file in "${clip_files[@]}"; do
                aws s3 cp "$local_file" "${s3_clip_subfolder}/$(basename "$local_file")" --only-show-errors --region us-east-1 || { segment_upload_success=0; break; }
            done
            if [ $segment_upload_success -eq 1 ]; then
                for local_file in "${meta_files[@]}"; do
                    aws s3 cp "$local_file" "${s3_meta_subfolder}/$(basename "$local_file")" --only-show-errors --region us-east-1 || { segment_upload_success=0; break; }
                done
            fi
            if [ $segment_upload_success -eq 0 ]; then
                echo "[Job $job_num/$total_jobs - $video_id] ERROR: S3 upload failed for some segments."
                stage_failed=1
            else
                num_segments_uploaded=$num_segments
            fi
        elif [ $stage_failed -eq 0 ] && [ $num_segments -eq 0 ]; then
            echo "[Job $job_num/$total_jobs - $video_id] No valid video segments created to upload."
        fi
    fi
    # -------------------------------------------------------------------------

    local job_duration=$((SECONDS - job_start_seconds))
    if [ $stage_failed -eq 1 ]; then
        echo "[Job $job_num/$total_jobs - $video_id] FAIL ($job_duration s). Temp files: $job_temp_dir"
        return 1
    else
        echo "[Job $job_num/$total_jobs - $video_id] OK ($num_segments_uploaded video segments, $job_duration s)."
        rm -rf "$job_temp_dir"
        return 0
    fi
}
export -f process_video


# ==============================================================================
# --- Main Script Logic (Setup & Parallel Execution) ---
# ==============================================================================


# --- Main Script Logic ---
echo "Preprocessing Pipeline (v8.2 - Clean Shutdown)"
echo "------------------------------------------------------------"

mkdir -p "$BASE_LOCAL_TEMP_DIR" "$WHISPER_CACHE_DIR" || { echo "ERROR: Failed to create base temp dir on RAM disk '$BASE_LOCAL_TEMP_DIR'."; exit 1; }
trap 'echo ""; echo ">>> Cleaning up base temp dir: $BASE_LOCAL_TEMP_DIR"; rm -rf "$BASE_LOCAL_TEMP_DIR"; echo ">>> Script finished after $SECONDS seconds."; exit' EXIT SIGINT SIGTERM

if ! command -v "$PYTHON_EXEC" &> /dev/null; then echo "ERROR: Python executable not found at '$PYTHON_EXEC'."; exit 1; fi
echo "Using Python: $($PYTHON_EXEC --version)"

# ==============================================================================
# --- PY HELPER SCRIPTS) --- (using new scripts)
# ==============================================================================


# EXTRACT AUDIO
cat << 'EOF' > "${BASE_LOCAL_TEMP_DIR}/extract_audio.py"
import os, subprocess, argparse, sys, json

def run_subprocess(cmd):
    try:
        # Using shell=False is generally safer, ensure cmd is a list of strings
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        return result
    except subprocess.CalledProcessError as e:
        print(f"  ERROR in subprocess {' '.join(cmd)}: Command failed with exit code {e.returncode}", flush=True)
        print(f"  Stdout: {e.stdout.strip()}", flush=True)
        print(f"  Stderr: {e.stderr.strip()}", flush=True)
        return None
    except subprocess.TimeoutExpired as e:
        print(f"  ERROR in subprocess {' '.join(cmd)}: Command timed out.", flush=True)
        print(f"  Stdout: {e.stdout.strip() if e.stdout else ''}", flush=True)
        print(f"  Stderr: {e.stderr.strip() if e.stderr else ''}", flush=True)
        return None
    except Exception as e:
        print(f"  ERROR in subprocess {' '.join(cmd)}: {e}", flush=True)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts audio to 16kHz mono PCM WAV and validates it.")
    parser.add_argument("--video_path", required=True, help="Path to the input video file.")
    parser.add_argument("--audio_dir", required=True, help="Directory to save the extracted audio file.")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"  ERROR: Video file not found: {args.video_path}", flush=True)
        sys.exit(1)

    os.makedirs(args.audio_dir, exist_ok=True)
    base_name = os.path.basename(args.video_path)
    video_id = os.path.splitext(base_name)[0]
    output_audio_filename = f"{video_id}.wav"
    output_audio_path = os.path.join(args.audio_dir, output_audio_filename)

    # ffmpeg command to re-encode to 16kHz mono PCM s16le WAV
    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-threads", "1", "-i", args.video_path,
        "-vn",                 # No video
        "-ac", "1",            # Mono audio channel
        "-ar", "16000",        # 16 kHz sample rate
        "-c:a", "pcm_s16le",   # 16-bit PCM Little Endian
        "-sample_fmt", "s16",  # Explicitly s16 sample format (though pcm_s16le implies it)
        "-y",                  # Overwrite output file if it exists
        output_audio_path
    ]

    print(f"  Executing ffmpeg: {' '.join(ffmpeg_cmd)}", flush=True)
    ffmpeg_result = run_subprocess(ffmpeg_cmd)

    if not ffmpeg_result:
        print(f"  ERROR: ffmpeg command failed during audio extraction for {args.video_path}.", flush=True)
        if os.path.exists(output_audio_path): # Clean up potentially corrupted file
            try: os.remove(output_audio_path)
            except OSError: pass
        sys.exit(1)

    if not os.path.exists(output_audio_path) or os.path.getsize(output_audio_path) == 0:
        print(f"  ERROR: Extracted audio file not found or is empty: {output_audio_path}", flush=True)
        sys.exit(1)

    # Sanity-check with ffprobe
    ffprobe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,channels,sample_rate,sample_fmt",
        "-show_entries", "format=duration",
        "-of", "json", output_audio_path
    ]
    print(f"  Executing ffprobe: {' '.join(ffprobe_cmd)}", flush=True)
    probe_result = run_subprocess(ffprobe_cmd)

    if not probe_result:
        print(f"  ERROR: ffprobe command failed for {output_audio_path}.", flush=True)
        sys.exit(1)

    try:
        meta = json.loads(probe_result.stdout or "{}")
        if not meta or "streams" not in meta or not meta["streams"] or "format" not in meta:
            print(f"  ERROR: ffprobe output is incomplete for {output_audio_path}. Output: {probe_result.stdout}", flush=True)
            sys.exit(1)

        stream_info = meta["streams"][0]
        codec = stream_info.get("codec_name")
        channels = stream_info.get("channels")
        sample_rate = stream_info.get("sample_rate")
        # sample_fmt = stream_info.get("sample_fmt") # Useful for debugging, pcm_s16le should be s16
        duration_str = meta["format"].get("duration", "0")
        duration = float(duration_str)

        valid_codec = codec == "pcm_s16le"
        valid_channels = channels == 1
        valid_sample_rate = sample_rate == "16000" # ffprobe returns string for sample_rate
        valid_duration = duration >= 1.0

        if valid_codec and valid_channels and valid_sample_rate and valid_duration:
            print(f"  Audio extraction successful and validated: {output_audio_path} (Codec: {codec}, Channels: {channels}, Rate: {sample_rate}Hz, Duration: {duration:.2f}s)", flush=True)
            print(f"OUTPUT_PATH:{output_audio_path}", flush=True)
            sys.exit(0)
        else:
            error_messages = []
            if not valid_codec: error_messages.append(f"Invalid codec (Expected pcm_s16le, Got {codec})")
            if not valid_channels: error_messages.append(f"Invalid channels (Expected 1, Got {channels})")
            if not valid_sample_rate: error_messages.append(f"Invalid sample rate (Expected 16000, Got {sample_rate})")
            if not valid_duration: error_messages.append(f"Duration too short (Expected >= 1.0s, Got {duration:.2f}s)")
            print(f"  ERROR: Extracted audio validation failed for {output_audio_path}: {'; '.join(error_messages)}", flush=True)
            if os.path.exists(output_audio_path): # Clean up invalid file
                try: os.remove(output_audio_path)
                except OSError: pass
            sys.exit(1)

    except json.JSONDecodeError:
        print(f"  ERROR: Failed to parse ffprobe JSON output for {output_audio_path}. Output: {probe_result.stdout}", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: Unexpected error during ffprobe validation for {output_audio_path}: {e}", flush=True)
        sys.exit(1)
EOF

#TRANSCRIBE FASTER WHISPER
cat << 'EOF' > "${BASE_LOCAL_TEMP_DIR}/transcribe_faster_whisper.py"
import os, argparse, json, sys, time
from faster_whisper import WhisperModel

# Get model name and cache dir from environment variables
WHISPER_MODEL_NAME_PY = os.environ.get("FASTER_WHISPER_MODEL", "large-v3")
WHISPER_CACHE_DIR_PY = os.environ.get("WHISPER_CACHE_DIR", "/tmp/whisper_cache")

def format_timestamp(seconds):
    """Converts seconds to HH:MM:SS,sss format."""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    seconds = milliseconds // 1_000
    milliseconds %= 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribes audio using Faster-Whisper.")
    parser.add_argument("--audio_path", required=True, help="Path to the input audio file.")
    parser.add_argument("--transcript_dir", required=True, help="Directory to save the transcript JSON.")
    parser.add_argument("--device_id", default="0", help="GPU device ID to use for transcription.")
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"ERROR: Audio not found: {args.audio_path}", file=sys.stderr, flush=True)
        sys.exit(1)
    os.makedirs(args.transcript_dir, exist_ok=True)

    base = os.path.basename(args.audio_path)
    audio_id = os.path.splitext(base)[0]
    out_name = f"{audio_id}_transcript.json"
    out_path = os.path.join(args.transcript_dir, out_name)

    device = "cuda"
    compute_type = "float16" # Use float16 for modern GPUs
    
    try:
        print(f"  Loading faster-whisper model: '{WHISPER_MODEL_NAME_PY}' on device 'cuda:{args.device_id}' (compute: {compute_type})", flush=True)
        print(f"  Model cache directory: {WHISPER_CACHE_DIR_PY}", flush=True)
        model = WhisperModel(WHISPER_MODEL_NAME_PY, device=device, device_index=int(args.device_id), compute_type=compute_type, download_root=WHISPER_CACHE_DIR_PY)
    except Exception as e:
        print(f"  ERROR loading faster-whisper model: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    
    try:
        start_time = time.time()
        print(f"  Starting transcription for {args.audio_path}...", flush=True)
        # Faster-whisper returns an iterator
        segments_iterator, info = model.transcribe(args.audio_path, language="en", word_timestamps=True, condition_on_previous_text=False)

        # We must consume the iterator to build the JSON object expected by the downstream script
        all_text_parts = []
        segments_list = []
        
        for i, segment in enumerate(segments_iterator):
            all_text_parts.append(segment.text)
            segment_dict = {
                "id": i,
                "seek": segment.seek,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "tokens": segment.tokens,
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
                "words": []
            }
            if segment.words:
                for word in segment.words:
                    segment_dict["words"].append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability
                    })
            segments_list.append(segment_dict)

        result_json = {
            "text": "".join(all_text_parts).strip(),
            "segments": segments_list,
            "language": info.language
        }

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)
        
        duration = time.time() - start_time
        print(f"  Transcription successful: {out_path} (took {duration:.2f}s)", flush=True)
        print(f"OUTPUT_PATH:{out_path}", flush=True)
        sys.exit(0)

    except Exception as e:
        print(f"  ERROR during transcription for {args.audio_path}: {e}", file=sys.stderr, flush=True)
        if os.path.exists(out_path):
            try: os.remove(out_path)
            except OSError: pass
        sys.exit(1)
EOF

#DIARIZATION NEW ONE
cat << 'EOF' > "${BASE_LOCAL_TEMP_DIR}/diarize_audio.py"
#DIARIZATION (GPU-serial, cache-friendly)
import os
import sys
import argparse
import json
import torch
import fcntl
from pyannote.audio import Pipeline

###############################################################################
# Configuration via env vars (all optional)
###############################################################################
MODEL_NAME = os.environ.get("DIAR_MODEL", "pyannote/speaker-diarization-3.1")
LOCK_FILE  = os.environ.get("DIAR_LOCK_FILE", "/tmp/diar_gpu.lock")
DEVICE_ENV = os.environ.get("DIAR_DEVICE", "auto")     # "auto" | "cuda" | "cpu"

###############################################################################
def acquire_lock(path: str):
    """Exclusive lock so only one diarization job touches the GPU at a time."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd

def release_lock(fd: int):
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)

###############################################################################
def load_pipeline():
    """Load the PyAnnote pipeline on the requested device."""
    want_cuda = (
        (DEVICE_ENV == "cuda") or
        (DEVICE_ENV == "auto" and torch.cuda.is_available())
    )

    device = torch.device("cuda") if want_cuda else torch.device("cpu")
    pipeline = Pipeline.from_pretrained(
        MODEL_NAME,
        use_auth_token=os.environ.get("HUGGING_FACE_HUB_TOKEN", None)
    )
    pipeline.to(device)
    return pipeline, device.type

###############################################################################
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simple wrapper around pyannote diarization with GPU lock.")
    ap.add_argument("--audio_path", required=True)
    ap.add_argument("--diarization_dir", required=True)
    ap.add_argument("--num_speakers", type=int, default=2)
    ap.add_argument("--rttm_name", default=None, help="Override output file name")
    args = ap.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"ERROR: audio not found: {args.audio_path}", file=sys.stderr, flush=True)
        sys.exit(1)

    os.makedirs(args.diarization_dir, exist_ok=True)
    audio_id  = os.path.splitext(os.path.basename(args.audio_path))[0]
    out_name  = args.rttm_name or f"{audio_id}.rttm"
    out_path  = os.path.join(args.diarization_dir, out_name)

    # --------------------------------------------------------------------- #
    # 1.  If RTTM already exists, short-circuit immediately
    # --------------------------------------------------------------------- #
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        print(f"OUTPUT_PATH:{out_path}", flush=True)
        sys.exit(0)

    # --------------------------------------------------------------------- #
    # 2.  Acquire exclusive GPU lock (CPU jobs skip this section)
    # --------------------------------------------------------------------- #
    lock_fd = None
    if DEVICE_ENV != "cpu":
        try:
            lock_fd = acquire_lock(LOCK_FILE)
        except OSError as e:
            print(f"ERROR: could not obtain GPU lock: {e}", file=sys.stderr, flush=True)
            sys.exit(1)

    try:
        # ---------------------------------------------------------------- #
        # 3.  Load pipeline and diarise
        # ---------------------------------------------------------------- #
        pipeline, dev = load_pipeline()
        diarization   = pipeline(args.audio_path, num_speakers=args.num_speakers)

        with open(out_path, "w") as f:
            diarization.write_rttm(f)

        print(f"OUTPUT_PATH:{out_path}", flush=True)
        sys.exit(0)

    except Exception as e:
        print(f"ERROR during diarization: {e}", file=sys.stderr, flush=True)
        if os.path.exists(out_path):
            try: os.remove(out_path)
            except OSError:
                pass
        sys.exit(1)

    finally:
        if lock_fd is not None:
            release_lock(lock_fd)
EOF

# Create segment_video_v7_9.py (OLD ONE)
cat << 'EOF' > "${BASE_LOCAL_TEMP_DIR}/segment_video_v7_9.py"
import os
import subprocess
import argparse
import sys
import json
import math
from collections import defaultdict

MIN_SEGMENT_DURATION_S = float(os.environ.get('MIN_SEGMENT_DURATION_S', 10.0))
MAX_SEGMENT_DURATION_S = float(os.environ.get('MAX_SEGMENT_DURATION_S', 35.0))
TARGET_SEGMENT_DURATION_S = float(os.environ.get('TARGET_SEGMENT_DURATION_S', 20.0))
MAX_MERGE_GAP_S = float(os.environ.get('MAX_MERGE_GAP_S', 1.5))

def run_subprocess_ffmpeg(cmd):
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        return result
    except subprocess.CalledProcessError as e:
        print(f"      ERROR: FFmpeg command failed (code {e.returncode}) for: {' '.join(e.cmd)}", flush=True)
        print(f"      FFmpeg Stderr:\n{e.stderr.strip()}", flush=True)
        return None
    except subprocess.TimeoutExpired:
        print(f"      ERROR: FFmpeg command timed out: {' '.join(cmd)}", flush=True)
        return None
    except Exception as e:
        print(f"      ERROR running FFmpeg command {' '.join(cmd)}: {e}", flush=True)
        return None

def has_video_stream(file_path):
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        if result.returncode == 0 and result.stdout.strip(): return True
        else: return False
    except subprocess.TimeoutExpired:
        print(f"        WARNING: ffprobe timeout for {os.path.basename(file_path)}.", flush=True); return False
    except Exception as e:
        print(f"        WARNING: ffprobe exception for {os.path.basename(file_path)}: {e}", flush=True); return False

def find_word_boundary(words, target_time, look_after=True, tolerance=0.5):
    best_time = None; min_diff = float('inf')
    if not words: return target_time
    for word_info in words:
        word_start = word_info.get('start'); word_end = word_info.get('end')
        if word_start is None or word_end is None: continue
        boundary_time = word_start if look_after else word_end; diff = abs(boundary_time - target_time)
        if diff <= tolerance and diff < min_diff: min_diff = diff; best_time = boundary_time
    return best_time if best_time is not None else target_time

def merge_short_turns(turns, min_duration, max_gap):
    if not turns: return []
    sorted_turns = sorted(turns, key=lambda x: x['start']); merged = []
    if not sorted_turns: return merged
    current_turn = sorted_turns[0].copy()
    for i in range(1, len(sorted_turns)):
        next_turn = sorted_turns[i]; gap = next_turn['start'] - current_turn['end']
        current_duration = current_turn['end'] - current_turn['start']
        if current_duration < min_duration and 0 <= gap <= max_gap: current_turn['end'] = next_turn['end']
        else:
            if current_duration >= min_duration: merged.append(current_turn)
            current_turn = next_turn.copy()
    last_duration = current_turn['end'] - current_turn['start']
    if last_duration >= min_duration: merged.append(current_turn)
    return merged

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment video with video stream validation.")
    parser.add_argument("--video_date", required=True); parser.add_argument("--video_id", required=True)
    parser.add_argument("--video_path", required=True); parser.add_argument("--transcript_path", required=True)
    parser.add_argument("--rttm_path", required=True); parser.add_argument("--output_clip_dir", required=True)
    parser.add_argument("--output_meta_dir", required=True); args = parser.parse_args()

    os.makedirs(args.output_clip_dir, exist_ok=True); os.makedirs(args.output_meta_dir, exist_ok=True)
    all_words = []
    try:
        with open(args.transcript_path, 'r') as f: transcript_data = json.load(f)
        if 'segments' in transcript_data:
            for segment in transcript_data['segments']:
                if 'words' in segment and isinstance(segment['words'], list):
                    all_words.extend(w for w in segment['words'] if isinstance(w, dict) and 'start' in w and 'end' in w)
        if not all_words: print(f"    WARNING: No word timestamps in transcript for {args.video_id}.", flush=True)
    except Exception as e: print(f"  ERROR: Load transcript {args.transcript_path} ({args.video_id}) failed: {e}", flush=True); sys.exit(1)

    all_turns_raw = []; first_speaker_id = None; min_start_time = float('inf')
    try:
        with open(args.rttm_path, 'r') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0]=="SPEAKER":
                    try:
                        start = float(parts[3]); duration = float(parts[4]); speaker_id = parts[7]
                        if duration > 0.01:
                            if start < min_start_time: min_start_time = start; first_speaker_id = speaker_id
                            all_turns_raw.append({'start': start, 'end': start + duration, 'speaker': speaker_id})
                    except (ValueError, IndexError): print(f"    WARNING: Malformed RTTM line {line_num+1} in {args.rttm_path}.", flush=True)
        if not all_turns_raw: print(f"  ERROR: No turns in RTTM {args.rttm_path} ({args.video_id}).", flush=True); sys.exit(1)
        if first_speaker_id is None: print(f"  ERROR: No first speaker in RTTM {args.rttm_path} ({args.video_id}).", flush=True); sys.exit(1)
    except FileNotFoundError: print(f"  ERROR: RTTM not found: {args.rttm_path}", flush=True); sys.exit(1)
    except Exception as e: print(f"  ERROR: Load RTTM {args.rttm_path} ({args.video_id}) failed: {e}", flush=True); sys.exit(1)

    target_speaker_id = first_speaker_id; chair_name = "Powell"
    chair_turns_raw = [turn for turn in all_turns_raw if turn['speaker'] == target_speaker_id]
    if not chair_turns_raw: print(f"    ERROR: No turns for Chair '{target_speaker_id}' ({args.video_id}).", flush=True); sys.exit(1)
    chair_turns_to_process = merge_short_turns(chair_turns_raw, MIN_SEGMENT_DURATION_S, MAX_MERGE_GAP_S)
    if not chair_turns_to_process: print(f"    INFO: No Chair turns for '{target_speaker_id}' ({args.video_id}) after merge.", flush=True); sys.exit(0)

    segment_count = 0; output_files = []
    for turn_idx, turn in enumerate(chair_turns_to_process):
        turn_start = turn['start']; turn_end = turn['end']; turn_duration = turn_end - turn_start
        if turn_duration < MIN_SEGMENT_DURATION_S / 2.0 : continue

        if turn_duration <= MAX_SEGMENT_DURATION_S: num_sub_segments = 1
        else: num_sub_segments = math.ceil(turn_duration / TARGET_SEGMENT_DURATION_S); effective_sub_segment_dur = turn_duration / num_sub_segments
        
        for i in range(num_sub_segments):
            if num_sub_segments > 1:
                 seg_start_ideal = turn_start + i * effective_sub_segment_dur; seg_end_ideal = seg_start_ideal + effective_sub_segment_dur
                 if i == num_sub_segments - 1: seg_end_ideal = turn_end
            else: seg_start_ideal = turn_start; seg_end_ideal = turn_end

            seg_start_adj = find_word_boundary(all_words, seg_start_ideal, look_after=True)
            seg_end_adj = find_word_boundary(all_words, seg_end_ideal, look_after=False)

            if seg_start_adj >= seg_end_adj:
                 seg_start_adj = seg_start_ideal; seg_end_adj = seg_end_ideal
                 if seg_start_adj >= seg_end_adj: print(f"        WARNING: Invalid segment times ({args.video_id}). Skipping.", flush=True); continue

            seg_duration = seg_end_adj - seg_start_adj
            if seg_duration < MIN_SEGMENT_DURATION_S / 2.0: continue

            current_segment_num_display = segment_count + 1 
            date_str = args.video_date
            base_filename_part = f"FOMC_{date_str}_{args.video_id}_{chair_name}_seg{current_segment_num_display:03d}"
            clip_filename = f"{base_filename_part}.mp4"; output_clip_path = os.path.join(args.output_clip_dir, clip_filename)
            meta_filename = f"{base_filename_part}.json"; output_meta_path = os.path.join(args.output_meta_dir, meta_filename)
            clip_created_successfully = False
            
            ffmpeg_command_copy = ["ffmpeg","-hide_banner","-loglevel","error","-i",args.video_path,"-ss",str(seg_start_adj),"-to",str(seg_end_adj),"-c","copy","-map","0","-avoid_negative_ts","make_zero","-movflags","+faststart","-y",output_clip_path]
            result_copy = run_subprocess_ffmpeg(ffmpeg_command_copy)
            if result_copy and os.path.exists(output_clip_path) and os.path.getsize(output_clip_path) > 0:
                if has_video_stream(output_clip_path): clip_created_successfully = True
                else:
                    print(f"        WARNING: Stream copy seg {current_segment_num_display} ({args.video_id}) lacks video. Re-encoding.", flush=True)
                    if os.path.exists(output_clip_path): # CORRECTED SYNTAX BLOCK
                        try: os.remove(output_clip_path)
                        except OSError as e: print(f"        WARNING: Failed remove invalid stream copy file {output_clip_path}: {e}", flush=True)
            
            if not clip_created_successfully:
                if os.path.exists(output_clip_path): # CORRECTED SYNTAX BLOCK
                    try: os.remove(output_clip_path)
                    except OSError as e: print(f"        WARNING: Failed remove previous attempt file {output_clip_path}: {e}", flush=True)
                ffmpeg_command_gpu = ["ffmpeg","-hide_banner","-loglevel","error","-i", args.video_path,"-ss", str(seg_start_adj),"-to", str(seg_end_adj),"-c:v", "h264_nvenc","-preset", "p5","-cq", "23","-c:a", "aac","-b:a", "128k","-movflags", "+faststart","-y", output_clip_path]
                result_gpu = run_subprocess_ffmpeg(ffmpeg_command_gpu)
                if result_gpu and os.path.exists(output_clip_path) and os.path.getsize(output_clip_path) > 0:
                    if has_video_stream(output_clip_path): clip_created_successfully = True
                    else: print(f"        WARNING: GPU encode seg {current_segment_num_display} ({args.video_id}) lacks video. CPU next.", flush=True)
            
            if not clip_created_successfully:
                if os.path.exists(output_clip_path): # CORRECTED SYNTAX BLOCK
                    try: os.remove(output_clip_path)
                    except OSError as e: print(f"        WARNING: Failed remove previous attempt file {output_clip_path}: {e}", flush=True)
                ffmpeg_command_cpu = ["ffmpeg","-hide_banner","-loglevel","error","-i",args.video_path,"-ss",str(seg_start_adj),"-to",str(seg_end_adj),"-c:v","libx264","-preset","superfast","-crf","23","-c:a","aac","-b:a","128k","-movflags","+faststart","-y",output_clip_path]
                result_cpu = run_subprocess_ffmpeg(ffmpeg_command_cpu)
                if result_cpu and os.path.exists(output_clip_path) and os.path.getsize(output_clip_path) > 0:
                    if has_video_stream(output_clip_path): clip_created_successfully = True
                    else: print(f"        WARNING: CPU encode seg {current_segment_num_display} ({args.video_id}) lacks video.", flush=True)
            
            if clip_created_successfully:
                segment_count += 1 
                metadata={"segment_id": base_filename_part,"source_video_id": args.video_id,"video_date_yyyymmdd": args.video_date,"chair_name": chair_name,"diarization_speaker_id": target_speaker_id,"segment_start_time_s": round(seg_start_adj, 3),"segment_end_time_s": round(seg_end_adj, 3),"segment_duration_s": round(seg_duration, 3),"original_turn_start_s": round(turn_start, 3),"original_turn_end_s": round(turn_end, 3),"clip_filename": clip_filename,"s3_clip_path": f"__S3_CLIP_DIR__/{clip_filename}","s3_metadata_path": f"__S3_META_DIR__/{meta_filename}"}
                try:
                    with open(output_meta_path, 'w') as f: json.dump(metadata, f, indent=2)
                    output_files.append(f"CLIP:{output_clip_path}"); output_files.append(f"META:{output_meta_path}")
                except Exception as e:
                    print(f"      ERROR write metadata {output_meta_path} ({args.video_id} seg {current_segment_num_display}): {e}", flush=True)
                    if os.path.exists(output_clip_path): 
                         try: os.remove(output_clip_path)
                         except OSError as oe: print(f"      Error remove clip {output_clip_path}: {oe}", flush=True)
                    segment_count -=1 
            else:
                print(f"      ERROR: All ffmpeg attempts for seg {current_segment_num_display} ({args.video_id}) failed. Skipping.", flush=True)
                if os.path.exists(output_clip_path): # CORRECTED SYNTAX BLOCK
                    try: os.remove(output_clip_path)
                    except OSError as oe: print(f"        WARNING: Failed remove final failed output {output_clip_path}: {oe}", flush=True)

    final_video_segment_count = len([f for f in output_files if f.startswith("CLIP:")])
    print(f"  Segmentation for {args.video_id} created {final_video_segment_count} valid video segment(s).", flush=True)
    if final_video_segment_count > 0:
        for fpath in output_files: print(f"OUTPUT_PATH:{fpath}", flush=True)
    sys.exit(0)
EOF


mapfile -t VIDEO_TARGETS < <(grep -vE '^\s*(#|$)' "$LOCAL_VIDEO_LIST"); TOTAL_VIDEOS=${#VIDEO_TARGETS[@]};
echo "Found $TOTAL_VIDEOS video targets in '$LOCAL_VIDEO_LIST'."
if [ $TOTAL_VIDEOS -eq 0 ]; then echo "ERROR: No valid video targets found. Exiting."; exit 1; fi
echo "-------------------------------------"

echo "Starting main processing loop ($MAX_PARALLEL_JOBS jobs in parallel)..."
active_jobs=0; job_num=0; pids=(); results_dir="${BASE_LOCAL_TEMP_DIR}/job_results"; mkdir -p "$results_dir"

for video_target_line in "${VIDEO_TARGETS[@]}"; do
    [[ -z "$video_target_line" ]] && continue; job_num=$((job_num + 1))
    ( process_video "$video_target_line" "$job_num" "$TOTAL_VIDEOS"; echo $? > "${results_dir}/job_status_num${job_num}.txt" ) &
    pids+=($!); active_jobs=$((active_jobs + 1))
    if [ "$active_jobs" -ge "$MAX_PARALLEL_JOBS" ]; then
        wait -n # Wait for any job
        for pid_idx in "${!pids[@]}"; do if ! kill -0 "${pids[$pid_idx]}" 2>/dev/null; then wait "${pids[$pid_idx]}"; unset pids[$pid_idx]; active_jobs=$((active_jobs - 1)); fi; done
        pids=("${pids[@]}") # Re-index array
    fi
done
echo "All jobs dispatched. Waiting for remaining $active_jobs jobs..."
while [ "$active_jobs" -gt 0 ]; do
    wait -n
    for pid_idx in "${!pids[@]}"; do if ! kill -0 "${pids[$pid_idx]}" 2>/dev/null; then wait "${pids[$pid_idx]}"; unset pids[$pid_idx]; active_jobs=$((active_jobs - 1)); fi; done
    pids=("${pids[@]}")
done
echo "All background jobs completed."; echo "-------------------------------------"

SUCCESS_COUNT=0; FAIL_COUNT=0; NODATE_FAIL_COUNT=0
for i in $(seq 1 $job_num); do # Iterate up to actual job_num dispatched
    status_file="${results_dir}/job_status_num${i}.txt"
    if [ -f "$status_file" ]; then
        status_val=$(cat "$status_file")
        if [ "$status_val" -eq 0 ]; then SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        elif [ "$status_val" -eq 2 ]; then NODATE_FAIL_COUNT=$((NODATE_FAIL_COUNT + 1)); FAIL_COUNT=$((FAIL_COUNT + 1))
        else FAIL_COUNT=$((FAIL_COUNT + 1)); fi
    else echo "WARNING: Status file for job num ${i} not found. Assuming failure."; FAIL_COUNT=$((FAIL_COUNT + 1)); fi
done

echo ""; echo "# --- Overall Summary ---";
echo "# Total video targets processed: $job_num (out of $TOTAL_VIDEOS listed)";
echo "# Successfully completed jobs (valid date & video segments): $SUCCESS_COUNT";
echo "# Failed jobs (total): $FAIL_COUNT"
if [ "$NODATE_FAIL_COUNT" -gt 0 ]; then echo "#   - Jobs failed due to missing upload date (NODATE/cookie issue): $NODATE_FAIL_COUNT"; echo "#     (Check logs for 'ACTION_REQUIRED' messages for these.)"; fi
if [ $((FAIL_COUNT - NODATE_FAIL_COUNT)) -gt 0 ]; then echo "#   - Jobs failed due to other processing errors: $((FAIL_COUNT - NODATE_FAIL_COUNT))"; fi
echo "#"; echo "# S3 Base: $S3_BASE_PATH"; echo "# Segmented clips: $S3_SEGMENTED_CLIPS_DIR/<VIDEO_ID>/"; echo "# Metadata: $S3_METADATA_DIR/<VIDEO_ID>/"
echo "#"; echo "# Detailed segmentation logs: $BASE_LOCAL_TEMP_DIR/job_<video_id>/segmentation_raw_output.log"
echo "# Main script errors/progress were printed above."; echo "# Temp dirs may remain on failure: $BASE_LOCAL_TEMP_DIR/job_<video_id>"
echo "-------------------------------------"; exit 0
