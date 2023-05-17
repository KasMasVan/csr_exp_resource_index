seed=0
model_family="FLAN-T5"
checkpoints=("google/flan-t5-small")
loading_precision="FP16"
# checkpoints=("google/flan-t5-small" "google/flan-t5-base" "google/flan-t5-large")
# amateur_checkpoint="google/flan-t5-small"
# expert_checkpoint="google/flan-t5-base"
# datasets="cqa copa obqa piqa qasc siqa winogrande"
datasets="emoji_movies"
batch_size=16
sample=100

multiple_choice_prompt="Question:"

for checkpoint in "${checkpoints[@]}"
do
    # language modeling and average language modeling
    python language_modeling.py \
        --model_family ${model_family} \
        --checkpoint ${checkpoint} \
        --datasets "$datasets" \
        --batch_size  ${batch_size} \
        --loading_precision ${loading_precision} \
        # --push_data_to_hub \
        # --sample ${sample} \

    # contrastive decoding
    # python contrastive_decoding.py \
    #     --model_family ${model_family} \
    #     --amateur_checkpoint ${amateur_checkpoint} \
    #     --expert_checkpoint ${expert_checkpoint} \
    #     --datasets "$datasets" \
    #     --batch_size  ${batch_size} \

    # multiple choice prompt, using the same script as language modeling
    python language_modeling.py \
        --model_family ${model_family} \
        --checkpoint ${checkpoint} \
        --datasets "$datasets" \
        --batch_size  ${batch_size} \
        --loading_precision ${loading_precision} \
        --multiple_choice_prompt ${multiple_choice_prompt} \
        # --push_data_to_hub \
        # --sample ${sample} \

    # process of elimination
    # python process_of_elimination.py \
    #     --model_family ${model_family} \
    #     --checkpoint ${checkpoint} \
    #     --loading_precision ${loading_precision} \
    #     --datasets "$datasets" \
    #     --batch_size  ${batch_size} \
    #     --multiple_choice_prompt ${multiple_choice_prompt} \
        # --push_data_to_hub 
        # --sample ${sample} 
done