############## Evaluation Parameters ################
export MODEL_PATH=$1     # Search agent checkpoint (served on :6001)
export DATASET=$2        # gaia | browsecomp_zh | browsecomp_en
export OUTPUT_PATH=$3    # Output path for prediction results
export PARADIGM=${4:-resum}   # react | resum | fixed_interval | recent_history | selfcompact
export DCK_MODE=${5:-dck}     # off | dck | single | random

# DCK artifacts (required when DCK_MODE is dck/single)
export DCK_LOCAL_MODEL=${DCK_LOCAL_MODEL:-$MODEL_PATH}  # local HF copy of the same checkpoint
export DCK_HEAD=${DCK_HEAD:-/path/to/closure_head_7b}
export DCK_SINGLE_HEAD=${DCK_SINGLE_HEAD:-/path/to/single_head_7b}
export DCK_SCALER=${DCK_SCALER:-/path/to/closure_head_7b/scaler.json}
export DCK_STOP_TABLES=${DCK_STOP_TABLES:-/path/to/stop_tables_7b.json}
export DCK_EVENT_LOG=${DCK_EVENT_LOG:-}
export DCK_CONTEXT_LOG=${DCK_CONTEXT_LOG:-}

######################################
##### 0. System Configuration   #####
######################################
# Search Tool
export GOOGLE_SEARCH_KEY="Your Google Search API Key"

# Visit Tool
export JINA_API_KEYS="Your Jina API Key"
export SUMMARY_API_KEY="EMPTY"
export SUMMARY_API_BASE="http://localhost:8001/v1"
export SUMMARY_MODEL_NAME="/path/to/your/Qwen3-30B-A3B-Instruct-2507"

# Summarizer (frozen across all arms of one host)
export RESUM_TOOL_NAME="/path/to/your/ReSum-Tool-30B-A3B"
export RESUM_TOOL_URL="http://localhost:6002/v1/chat/completions"

######################################
### 1. Start server (background)   ###
######################################
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6001 --tensor-parallel-size 4 &
CUDA_VISIBLE_DEVICES=4,5 vllm serve "/path/to/your/Qwen3-30B-A3B-Instruct-2507" --host 0.0.0.0 --port 8001 --tensor-parallel-size 2 &
CUDA_VISIBLE_DEVICES=6,7 vllm serve "/path/to/your/ReSum-Tool-30B-A3B" --host 0.0.0.0 --port 6002 --tensor-parallel-size 2 &

#####################################
#### 2. Wait for server ready     ###
#####################################
timeout=2400
sleep_interval=30

check_port() {
    local port=$1
    local model_name=$2
    local start_time=$(date +%s)

    echo "Wait for $port ($model_name) to start..."
    while true; do
        if netstat -tuln 2>/dev/null | grep -q ":$port " || ss -tuln 2>/dev/null | grep -q ":$port "; then
            if curl -s --connect-timeout 10 --max-time 5 http://localhost:$port/v1/chat/completions > /dev/null 2>&1; then
                echo "Port $port ($model_name) is ready."
                return 0
            fi
        fi

        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        if [ $elapsed -gt $timeout ]; then
            echo "Error: start port $port ($model_name) timeout ($timeout seconds)!"
            return 1
        fi
        echo "Port $port ($model_name) is not started, waiting $sleep_interval seconds to retry... (elapsed ${elapsed} seconds)"
        sleep $sleep_interval
    done
}

check_port 6001 "Infer Model" || exit 1
check_port 8001 "Summary Model" || exit 1
check_port 6002 "Summarizer" || exit 1

echo "All vLLM services are ready!"

#####################################
#### 3. Start inference           ####
#####################################
echo "==== Starting inference ($PARADIGM / $DCK_MODE)... ===="

# Todo: Activate inference conda environment

python3 -u main.py \
        --dataset $DATASET \
        --output $OUTPUT_PATH \
        --max_workers 40 \
        --model $MODEL_PATH \
        --paradigm $PARADIGM \
        --dck_mode $DCK_MODE \
        --dck_local_model $DCK_LOCAL_MODEL \
        --dck_head $DCK_HEAD \
        ${DCK_SINGLE_HEAD:+--dck_single_head $DCK_SINGLE_HEAD} \
        --dck_scaler $DCK_SCALER \
        --dck_stop_tables $DCK_STOP_TABLES \
        ${DCK_EVENT_LOG:+--dck_event_log $DCK_EVENT_LOG} \
        ${DCK_CONTEXT_LOG:+--dck_context_log $DCK_CONTEXT_LOG}

echo "==== Inference completed! ===="
exit 0
