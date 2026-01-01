#!/bin/bash
#SBATCH --job-name=wemg-evaluation
#SBATCH --output=logs/wemg-evaluation_%j.log
#SBATCH --error=logs/wemg-evaluation_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --partition=gpulong,preempt,cisds,memorylong
#SBATCH --array=0-10

source ~/.bashrc
conda activate wemg

# Stagger job starts by 60 seconds per array task to avoid ChromaDB race conditions
sleep $((SLURM_ARRAY_TASK_ID * 60))

dataset=(
    "cwq"
    "webqsp"
    "qald_10"
    "hotpotqa_adv"
    "grail_qa"
    "2wiki"
    "hotpotqa"
    "musique"
    "bamboogle" 
    "frames"
)

data_name_or_path=${dataset[$SLURM_ARRAY_TASK_ID]}
output_path=results/Qwen3-8B/Qwen3-4B-Emb/${data_name_or_path}
model_name=Qwen3-8B
search_strategy=mcts

python -m wemg.evaluation.evaluate \
    +dataset_name_or_path=${data_name_or_path} \
    +output_path=${output_path} \
    llm.model_name=${model_name} \
    search.strategy=${search_strategy}


echo "Job completed successfully."
