# you may specify model families and checkpoints here
# model_family="FLAN-T5"
model_families=("Pythia")
# checkpoint="google/flan-t5-small"
checkpoints=("EleutherAI/pythia-70m" "EleutherAI/pythia-160m")

for model_family in "${model_families[@]}"
do
    for checkpoint in "${checkpoints[@]}"
    do
        python models/model_downloaders/model_downloaders.py \
            --model_family ${model_family} \
            --checkpoint ${checkpoint} \
            # --download_all_checkpoints 
    done
done

# for model_family in "${model_families[@]}"
# do
#     python models/model_downloaders/model_downloaders.py \
#         --model_family ${model_family} \
#         --download_all_checkpoints
# done