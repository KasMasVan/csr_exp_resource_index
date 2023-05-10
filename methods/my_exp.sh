seed=0
model_family="FLAN-T5"
# checkpoints=("EleutherAI/pythia-70m-deduped" "EleutherAI/pythia-160m-deduped" "EleutherAI/pythia-410m-deduped")
checkpoints=("google/flan-t5-small" "google/flan-t5-base" "google/flan-t5-large")
# amateur_checkpoint="google/flan-t5-small"
# expert_checkpoint="google/flan-t5-base"
datasets="cqa copa obqa piqa qasc siqa winogrande"
batch_size=16

multiple_choice_prompt="Question:"

for checkpoint in "${checkpoints[@]}"
do
    # language modeling and average language modeling
    python language_modeling.py \
        --model_family ${model_family} \
        --checkpoint ${checkpoint} \
        --datasets "$datasets" \
        --batch_size  ${batch_size} \

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
        --multiple_choice_prompt ${multiple_choice_prompt}

    # process of elimination
    python process_of_elimination.py \
        --model_family ${model_family} \
        --checkpoint ${checkpoint} \
        --datasets "$datasets" \
        --batch_size  ${batch_size} \
        --multiple_choice_prompt ${multiple_choice_prompt}
done