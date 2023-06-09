#!/bin/bash
seeds=(0 1 2 3 4)
model_family="FLAN-T5"  # "OPT-IML" "FLAN-T5" "Pythia" 
checkpoints=("google/flan-t5-xl") # "EleutherAI/pythia-2.8b" "facebook/opt-iml-1.3b" "facebook/opt-iml-max-1.3b" "google/flan-t5-large" "google/flan-t5-xl"
loading_precision="FP16" # FP32 FP16 BF16(for 7b models) INT8
# amateur_checkpoint="EleutherAI/pythia-70m" # 160m
datasets="anli cqa siqa logical_deduction_five_objects disambiguation_qa conceptual_combinations strange_stories symbol_interpretation"
# csr_datasets="copa cqa obqa piqa qasc siqa winogrande" 
# datasets="disambiguation_qa conceptual_combinations date_understanding emoji_movie ruin_names temporal_sequences code_line_description penguins_in_a_table strange_stories symbol_interpretation tracking_shuffled_objects logical_deduction_three_objects logical_deduction_five_objects logical_deduction_seven_objects" 
# datasets="copa cqa obqa piqa qasc siqa winogrande anli" 
# datasets="logical_deduction_three_objects disambiguation_qa"
# datasets="anli_r1 anli_r2 anli_r3"
# datasets="copa cqa obqa piqa qasc siqa winogrande anli disambiguation_qa conceptual_combinations date_understanding emoji_movie ruin_names temporal_sequences code_line_description penguins_in_a_table strange_stories symbol_interpretation tracking_shuffled_objects logical_deduction_three_objects logical_deduction_five_objects logical_deduction_seven_objects"
batch_size=16
sample=100
n_shot=0

multiple_choice_prompt=""
# multiple_choice_prompt="Select the most suitable option to answer the question."
calibration_prompt=" the answer is:"
process_of_elimination_prompt="Select the most suitable option to answer the question. Ignore [MASK] options."
# number_of_synonyms=5
# generate_synonyms_prompt="Generate a synonym to '{option}':"
expert_method="channel" # "language_modeling" "calibration" "channel" "multiple_choice_prompt"
amateur_method="language_modeling" # "language_modeling" "calibration" "channel" "multiple_choice_prompt"

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
        --n_shot ${n_shot} \
        --sample ${sample} \
    #     --push_data_to_hub \
        

    # contrastive decoding
    # python contrastive_decoding.py \
    #     --seed ${seed} \
    #     --model_family ${model_family} \
    #     --checkpoint ${checkpoint} \
    #     --amateur_checkpoint ${amateur_checkpoint} \
    #     --datasets "$datasets" \
    #     --batch_size  ${batch_size} \
    #     --loading_precision ${loading_precision} \
    #     --expert_method ${expert_method} \
    #     --amateur_method ${amateur_method} \
    #     --n_shot ${n_shot} \
    #     --sample ${sample} \
        # --push_data_to_hub \
        

    # channel
    python language_modeling.py \
        --seed ${seed} \
        --model_family ${model_family} \
        --checkpoint ${checkpoint} \
        --datasets "$datasets" \
        --batch_size  ${batch_size} \
        --loading_precision ${loading_precision} \
        --do_channel \
        --n_shot ${n_shot} \
        --sample ${sample} \
        # --push_data_to_hub \

    # multiple choice prompt, using the same script as language modeling
    python language_modeling.py \
        --seed ${seed} \
        --model_family ${model_family} \
        --checkpoint ${checkpoint} \
        --datasets "$datasets" \
        --batch_size  ${batch_size} \
        --loading_precision ${loading_precision} \
        --multiple_choice_prompt "$multiple_choice_prompt" \
        --n_shot ${n_shot} \
        --sample ${sample} \
    #     --push_data_to_hub \
    
    # calibration, i.e., PMI and PMI_DC.
    python language_modeling.py \
        --seed ${seed} \
        --model_family ${model_family} \
        --checkpoint ${checkpoint} \
        --datasets "$datasets" \
        --batch_size  ${batch_size} \
        --loading_precision ${loading_precision} \
        --calibration_prompt "${calibration_prompt}" \
        --n_shot ${n_shot} \
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
        --multiple_choice_prompt "$multiple_choice_prompt" \
        --process_of_elimination_prompt "${process_of_elimination_prompt}" \
        --scoring_method_for_process_of_elimination "multiple_choice_prompt" \
        --mask_strategy_for_process_of_elimination "below_average" \ # "lowest"
        --n_shot ${n_shot} \
        --sample ${sample} \
        # --push_data_to_hub 

    # generate synonyms
    # python language_modeling.py \
    #     --seed ${seed} \
    #     --model_family ${model_family} \
    #     --checkpoint ${checkpoint} \
    #     --datasets "$datasets" \
    #     --batch_size  ${batch_size} \
    #     --loading_precision ${loading_precision} \
    #     --do_synonym \
    #     --number_of_synonyms ${number_of_synonyms} \
    #     --generate_synonyms_prompt "${generate_synonyms_prompt}" \
    # --n_shot ${n_shot} \
    #     --sample ${sample} \
        # --push_data_to_hub \
    done
done