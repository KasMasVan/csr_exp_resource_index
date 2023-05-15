# you may specify model families and checkpoints here
model_families=("OPT-IML")
checkpoints=("facebook/opt-iml-1.3b" "facebook/opt-iml-max-1.3b")

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