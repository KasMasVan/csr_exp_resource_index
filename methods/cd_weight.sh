#!/bin/bash
seeds=(0 1 2 3 4)
model_family="Pythia"  # "OPT-IML" "FLAN-T5" "Pythia" 
checkpoints=("EleutherAI/pythia-2.8b") # "EleutherAI/pythia-2.8b" "facebook/opt-iml-1.3b" "facebook/opt-iml-max-1.3b" "google/flan-t5-large" "google/flan-t5-xl"
loading_precision="FP16" # FP32 FP16 BF16(for 7b models) INT8
amateur_checkpoint="EleutherAI/pythia-70m" # 160m
datasets="anli copa cqa obqa piqa winogrande code_line_description conceptual_combinations disambiguation_qa emoji_movie ruin_names temporal_sequences"
# datasets="anli cqa siqa logical_deduction_five_objects disambiguation_qa conceptual_combinations strange_stories symbol_interpretation"
# datasets="disambiguation_qa conceptual_combinations date_understanding emoji_movie ruin_names temporal_sequences code_line_description penguins_in_a_table strange_stories symbol_interpretation tracking_shuffled_objects logical_deduction_three_objects logical_deduction_five_objects logical_deduction_seven_objects" 
# datasets="copa cqa obqa piqa qasc siqa winogrande anli disambiguation_qa conceptual_combinations date_understanding emoji_movie ruin_names temporal_sequences code_line_description penguins_in_a_table strange_stories symbol_interpretation tracking_shuffled_objects logical_deduction_three_objects logical_deduction_five_objects logical_deduction_seven_objects"
batch_size=16
# sample=100 # inference on whole evaluation set.
n_shot=0

multiple_choice_prompt=""
# multiple_choice_prompt="Select the most suitable option to answer the question."
calibration_prompt=" the answer is:"
process_of_elimination_prompt="Select the most suitable option to answer the question. Ignore [MASK] options."
num_random_search=5
weighting_parameter=-1
expert_methods=("channel")
amateur_methods=("language_modeling") # best one for all expert methods


for seed in "${seeds[@]}"; do
    for checkpoint in "${checkpoints[@]}"; do
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
                # --sample ${sample} \
                # --push_data_to_hub \
            done
        done
    done
done