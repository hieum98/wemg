#!/bin/bash
#SBATCH --job-name=wemg-eval-all
#SBATCH --output=results/wemg-eval-all_%A_%a.log
#SBATCH --error=results/wemg-eval-all_%A_%a.log
#SBATCH --account=uonlp
#SBATCH --partition=gpulong,memorylong,cisds
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=256G
#SBATCH --array=0-8

set -euo pipefail
set +u
source ~/.bashrc
set -u
conda activate wemg

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs

datasets=(
    # "cwq"
    # "webqsp"
    "webqsp_sub"
    "cwq_sub"
    "2wiki"
    "hotpotqa"
    "musique"
    "bamboogle"
    # "frames"
)

dataset_name_or_path="${datasets[$SLURM_ARRAY_TASK_ID]}"
model_name="${MODEL_NAME:-Qwen3-8B}"
search_strategy="${SEARCH_STRATEGY:-cot}"
max_concurrent="${MAX_CONCURRENT:-8}"
max_examples="${MAX_EXAMPLES:-200}"
output_path="results/wemg-v5/${model_name}/cot/${dataset_name_or_path}"

kb_source="${KB_SOURCE:-wikidata}"
freebase_subgraph_cache="${FREEBASE_SUBGRAPH_CACHE:-}"
case "${dataset_name_or_path}" in
    webqsp_sub)
        kb_source="freebase_subgraph"
        freebase_subgraph_cache="${FREEBASE_SUBGRAPH_CACHE:-cache/webqsp_sub_graph_cache}"
        ;;
    cwq_sub)
        kb_source="freebase_subgraph"
        freebase_subgraph_cache="${FREEBASE_SUBGRAPH_CACHE:-cache/cwq_sub_graph_cache_split}"
        ;;
esac

python -m litellm_proxy redis start >"results/logs/litellm_proxy_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
redis_launcher_pid=$!
trap 'kill "${redis_launcher_pid}" >/dev/null 2>&1 || true' EXIT

python -m wemg.evaluation.evaluate \
    dataset_name_or_path="${dataset_name_or_path}" \
    output_path="${output_path}" \
    llm.model_name="${model_name}" \
    search.strategy="${search_strategy}" \
    node_generation.kb_source="${kb_source}" \
    memory.interaction_memory.scope=dataset \
    max_concurrent="${max_concurrent}" \
    max_examples="${max_examples}" \
    resume=True \
    ${freebase_subgraph_cache:+node_generation.freebase_subgraph_cache="${freebase_subgraph_cache}"}

echo "Job completed successfully for ${dataset_name_or_path}."
