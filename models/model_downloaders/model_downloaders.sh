# you may specify model families and checkpoints here
model_family="FLAN-T5"
checkpoint="google/flan-t5-small"


python models/model_downloaders/model_downloaders.py \
    --model_family ${model_family} \
    --checkpoint ${checkpoint} \
    --download_all_checkpoints \
