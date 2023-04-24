# you may specify model families and checkpoints here
# model_family="FLAN-T5"
model_families=("GPT2" "T5")
# checkpoint="google/flan-t5-small"
checkpoints=("google/flan-t5-small" "google/flan-t5-base" "google/flan-t5-large" "google/flan-t5-xl")

# for checkpoint in "${checkpoints[@]}"
# do
#     python models/model_downloaders/model_downloaders.py \
#         --model_family ${model_family} \
#         --checkpoint ${checkpoint} \
#         # --download_all_checkpoints 
# done

for model_family in "${model_families[@]}"
do
    python models/model_downloaders/model_downloaders.py \
        --model_family ${model_family} \
        --download_all_checkpoints
done