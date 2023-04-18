model_family="FLAN-T5"
checkpoint="google/flan-t5-base"
data="cqa"
batch_size=32 
# method="language_modeling"

for method in "language_modeling" # "contrastive_decoding" 
do
    python inference.py \
    --model_family ${model_family} \
    --checkpoint ${checkpoint} \
    --data ${data} \
    --batch_size  ${batch_size} \
    --method ${method} 
done

