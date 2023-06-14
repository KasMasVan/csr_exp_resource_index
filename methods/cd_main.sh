#!/bin/bash
seeds=(0 1 2 3 4)
model_family="Pythia"  # "OPT-IML" "FLAN-T5" "Pythia" 
checkpoints=("EleutherAI/pythia-2.8b") # "EleutherAI/pythia-2.8b" "facebook/opt-iml-1.3b" "facebook/opt-iml-max-1.3b" "google/flan-t5-large" "google/flan-t5-xl"
loading_precision="FP16" # FP32 FP16 BF16(for 7b models) INT8
amateur_checkpoint="EleutherAI/pythia-70m" # 160m
datasets="anli code_line_description conceptual_combinations copa cqa disambiguation_qa emoji_movie obqa piqa ruin_names temporal_sequences winogrande"
# datasets="anli cqa siqa logical_deduction_five_objects disambiguation_qa conceptual_combinations strange_stories symbol_interpretation"
# datasets="disambiguation_qa conceptual_combinations date_understanding emoji_movie ruin_names temporal_sequences code_line_description penguins_in_a_table strange_stories symbol_interpretation tracking_shuffled_objects logical_deduction_three_objects logical_deduction_five_objects logical_deduction_seven_objects" 
# datasets="copa cqa obqa piqa qasc siqa winogrande anli disambiguation_qa conceptual_combinations date_understanding emoji_movie ruin_names temporal_sequences code_line_description penguins_in_a_table strange_stories symbol_interpretation tracking_shuffled_objects logical_deduction_three_objects logical_deduction_five_objects logical_deduction_seven_objects"
batch_size=16
sample=100
n_shot=0

multiple_choice_prompt=""
# multiple_choice_prompt="Select the most suitable option to answer the question."
calibration_prompt=" the answer is:"
process_of_elimination_prompt="Select the most suitable option to answer the question. Ignore [MASK] options."
num_random_search=0
# weighting_parameter=-1
weighting_parameters="-0.6045423508054584 -0.0480091550541323 -0.1125038429707516 -0.0474510904475165 -0.0331986456493744 -0.4651900658197523 -0.0925155461102666 -0.0866940645715281 -0.0427633155344719 -1.9612660842594056 -1.8932749097658392 -0.1272327002791391"
expert_methods=("channel") # "language_modeling" "calibration" "channel" "multiple_choice_prompt"
amateur_methods=("language_modeling") # best one for all expert methods


for seed in "${seeds[@]}"; do
    for checkpoint in "${checkpoints[@]}"; do
        for expert_method in "${expert_methods[@]}"; do
            for amateur_method in "${amateur_methods[@]}"; do
            # language modeling and average language modeling
            # python language_modeling.py \
            #     --seed ${seed} \
            #     --model_family ${model_family} \
            #     --checkpoint ${checkpoint} \
            #     --datasets "$datasets" \
            #     --batch_size  ${batch_size} \
            #     --loading_precision ${loading_precision} \
            #     --n_shot ${n_shot} \
            #     --sample ${sample} \
            #     --push_data_to_hub \
                

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
                --weighting_parameters "$weighting_parameters" \
                --num_random_search ${num_random_search} \
                --n_shot ${n_shot} \
                --sample ${sample} \
                # --weighting_parameter ${weighting_parameter} \
                # --push_data_to_hub \
                

            # channel
            # python language_modeling.py \
            #     --seed ${seed} \
            #     --model_family ${model_family} \
            #     --checkpoint ${checkpoint} \
            #     --datasets "$datasets" \
            #     --batch_size  ${batch_size} \
            #     --loading_precision ${loading_precision} \
            #     --do_channel \
            #     --n_shot ${n_shot} \
            #     --sample ${sample} \
                # --push_data_to_hub \

            # multiple choice prompt, using the same script as language modeling
            # python language_modeling.py \
            #     --seed ${seed} \
            #     --model_family ${model_family} \
            #     --checkpoint ${checkpoint} \
            #     --datasets "$datasets" \
            #     --batch_size  ${batch_size} \
            #     --loading_precision ${loading_precision} \
            #     --multiple_choice_prompt "$multiple_choice_prompt" \
            #     --n_shot ${n_shot} \
            #     --sample ${sample} \
            #     --push_data_to_hub \
            
            # calibration, i.e., PMI and PMI_DC.
            # python language_modeling.py \
            #     --seed ${seed} \
            #     --model_family ${model_family} \
            #     --checkpoint ${checkpoint} \
            #     --datasets "$datasets" \
            #     --batch_size  ${batch_size} \
            #     --loading_precision ${loading_precision} \
            #     --calibration_prompt "${calibration_prompt}" \
            #     --n_shot ${n_shot} \
            #     --sample ${sample} \
                # --push_data_to_hub \
            done
        done
    done
done