seed=0
model_family="Pythia"
# checkpoint="EleutherAI/pythia-70m-deduped"
checkpoints=("EleutherAI/pythia-160m-deduped" "EleutherAI/pythia-410m-deduped" "EleutherAI/pythia-1b-deduped")
# amateur_checkpoint="google/flan-t5-small"
# expert_checkpoint="google/flan-t5-base"
datasets="cqa copa obqa piqa siqa winogrande"
batch_size=32 

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