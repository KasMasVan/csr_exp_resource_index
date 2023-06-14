#!/bin/bash
seeds=(0 1 2 3 4)
model_family="Pythia"  # "OPT-IML" "FLAN-T5" "Pythia" 
checkpoints=("EleutherAI/pythia-2.8b" "EleutherAI/pythia-410m") # "EleutherAI/pythia-2.8b" "facebook/opt-iml-1.3b" "facebook/opt-iml-max-1.3b" "google/flan-t5-large" "google/flan-t5-xl"
loading_precision="FP16" # FP32 FP16 BF16(for 7b models) INT8
amateur_checkpoints=("EleutherAI/pythia-70m" "EleutherAI/pythia-160m") # 160m
datasets="anli code_line_description conceptual_combinations copa cqa disambiguation_qa emoji_movie obqa piqa ruin_names temporal_sequences winogrande"
# datasets="anli emoji_movie temporal_sequences"
batch_size=16
sample=100
n_shot=0

multiple_choice_prompt=""
# multiple_choice_prompt="Select the most suitable option to answer the question."
calibration_prompt=" the answer is:"
process_of_elimination_prompt="Select the most suitable option to answer the question. Ignore [MASK] options."
num_random_search=0
weighting_parameter=-1
expert_methods=("channel") # "language_modeling" "calibration" "channel" "multiple_choice_prompt"
amateur_methods=("language_modeling") # best one for all expert methods


for seed in "${seeds[@]}"; do
    for checkpoint in "${checkpoints[@]}"; do
        for amateur_checkpoint in "${amateur_checkpoints[@]}"; do
            for expert_method in "${expert_methods[@]}"; do
                for amateur_method in "${amateur_methods[@]}"; do

                # contrastive decoding
                python contrastive_decoding.py \
                    --seed ${seed} \
                    --model_family ${model_family} \
                    --checkpoint ${checkpoint} \
                    --amateur_checkpoint ${amateur_checkpoint} \
                    --datasets "$datasets" \
                    --batch_size  ${batch_size} \
                    --loading_precision ${loading_precision} \
                    --expert_method ${expert_method} \
                    --amateur_method ${amateur_method} \
                    --weighting_parameter ${weighting_parameter} \
                    --num_random_search ${num_random_search} \
                    --n_shot ${n_shot} \
                    --sample ${sample} \
                    # --weighting_parameters "$weighting_parameters" \
                    # --push_data_to_hub \
                done
            done
        done
    done
done