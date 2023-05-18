#!/bin/bash

seeds=(0 1 2 3 4)
model_family="FLAN-T5"
checkpoints=("google/flan-t5-large" "google/flan-t5-xl")
loading_precision="FP16"
# checkpoints=("google/flan-t5-small" "google/flan-t5-base" "google/flan-t5-large")
# amateur_checkpoint="google/flan-t5-small"
# expert_checkpoint="google/flan-t5-base"
datasets="anli cqa qasc conceptual_combinations emoji_movie ruin_names strange_stories temporal_sequences"
# datasets="cqa copa obqa piqa qasc siqa winogrande"
batch_size=16
sample=100

multiple_choice_prompt="Question:"

for seed in "${seeds[@]}"; do
    for checkpoint in "${checkpoints[@]}"; do
    # language modeling and average language modeling
    python language_modeling.py \
        --seed ${seed} \
        --model_family ${model_family} \
        --checkpoint ${checkpoint} \
        --datasets "$datasets" \
        --batch_size  ${batch_size} \
        --loading_precision ${loading_precision} \
        --sample ${sample} \
        # --push_data_to_hub \
        

    # contrastive decoding
    # python contrastive_decoding.py \
    #     --model_family ${model_family} \
    #     --amateur_checkpoint ${amateur_checkpoint} \
    #     --expert_checkpoint ${expert_checkpoint} \
    #     --datasets "$datasets" \
    #     --batch_size  ${batch_size} \

    # multiple choice prompt, using the same script as language modeling
    python language_modeling.py \
        --seed ${seed} \
        --model_family ${model_family} \
        --checkpoint ${checkpoint} \
        --datasets "$datasets" \
        --batch_size  ${batch_size} \
        --loading_precision ${loading_precision} \
        --multiple_choice_prompt ${multiple_choice_prompt} \
        --sample ${sample} \
        # --push_data_to_hub \
        

    # process of elimination
    python process_of_elimination.py \
        --seed ${seed} \
        --model_family ${model_family} \
        --checkpoint ${checkpoint} \
        --loading_precision ${loading_precision} \
        --datasets "$datasets" \
        --batch_size  ${batch_size} \
        --multiple_choice_prompt ${multiple_choice_prompt} \
        --sample ${sample} 
        # --push_data_to_hub 
    done
done