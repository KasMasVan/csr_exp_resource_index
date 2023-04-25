seed=0
model_family="GPT2"
checkpoint="gpt2"
amateur_checkpoint="google/flan-t5-small"
expert_checkpoint="google/flan-t5-base"
datasets="cqa copa"
batch_size=32 
# method="language_modeling"

multiple_choice_prompt="Question:"

# for method in "language_modeling"  "contrastive_decoding" 
# do
#     python inference.py \
#     --model_family ${model_family} \
#     --checkpoint ${checkpoint} \
#     --datasets ${datasets} \
#     --batch_size  ${batch_size} \
#     --method ${method} 
# done

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

# multiple choice prompt 
# python multiple_choice_prompt.py \
#     --model_family ${model_family} \
#     --checkpoint ${checkpoint} \
#     --datasets "$datasets" \
#     --batch_size  ${batch_size} \
#     --multiple_choice_prompt ${multiple_choice_prompt}

# process of elimination
# python process_of_elimination.py \
#     --model_family ${model_family} \
#     --checkpoint ${checkpoint} \
#     --datasets "$datasets" \
#     --batch_size  ${batch_size} \