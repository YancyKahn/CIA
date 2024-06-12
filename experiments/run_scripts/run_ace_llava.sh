#!/bin/bash

# Set the path to the llava model
model_name="llava"
python_script_path="../experiment_evaluate.py"
epsilon='16/255'
device=0

# Run the experiment_evaluate.py script
python $python_script_path --model_name $model_name \
                           --epsilon $epsilon \
                           --lr 0.05 \
                           --max_iter 2000 \
                           --alpha 0.6 \
                           --beta 0.6 \
                           --instruction_length 8 \
                           --max_new_tokens 32 \
                           --eval_dataset "CLS" "CAP" "VQA" \
                           --padding_token '@' \
                           --seed 42 \
                           --image_bench_path "../../data/visualQA/val2014" \
                           --text_bench_path "../../data/text-bench/target.jsonl" \
                           --device $device \
                           --benchmark "vllm-attack" \
                           --embel_setting "mixed" \
                           --output_dir "../../output/final-llava"