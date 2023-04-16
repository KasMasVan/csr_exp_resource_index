model_family="FLAN-T5"
checkpoint="google/flan-t5-small"
data="copa"
batch_size=32

python language_modeling.py \
    --model_family ${model_family} \
    --checkpoint ${checkpoint} \
    --data ${data} \
    --batch_size  ${batch_size}\