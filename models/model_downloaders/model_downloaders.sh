# you may specify model families and checkpoints here
model_families=("Pythia")
checkpoints=("EleutherAI/pythia-160m" "EleutherAI/pythia-410m")

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