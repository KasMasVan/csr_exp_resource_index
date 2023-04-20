model_family="FLAN-T5"
checkpoint="google/flan-t5-small"
amateur_checkpoint="google/flan-t5-small"
expert_checkpoint="google/flan-t5-base"
data="copa"
batch_size=32 
# method="language_modeling"

# for method in "language_modeling"  "contrastive_decoding" 
# do
#     python inference.py \
#     --model_family ${model_family} \
#     --checkpoint ${checkpoint} \
#     --data ${data} \
#     --batch_size  ${batch_size} \
#     --method ${method} 
# done

# language modeling
python language_modeling.py \
    --model_family ${model_family} \
    --checkpoint ${checkpoint} \
    --data ${data} \
    --batch_size  ${batch_size} \

# contrastive decoding
python contrastive_decoding.py \
    --model_family ${model_family} \
    --amateur_checkpoint ${amateur_checkpoint} \
    --expert_checkpoint ${expert_checkpoint} \
    --data ${data} \
    --batch_size  ${batch_size} \