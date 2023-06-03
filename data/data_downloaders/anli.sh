mkdir data/anli/
cd data/anli/
wget https://dl.fbaipublicfiles.com/anli/anli_v1.0.zip
unzip anli_v1.0.zip

cp anli_v1.0/R1/dev.jsonl R1_dev.jsonl
cp anli_v1.0/R2/dev.jsonl R2_dev.jsonl
cp anli_v1.0/R3/dev.jsonl R3_dev.jsonl

cp anli_v1.0/R1/train.jsonl R1_train.jsonl
cp anli_v1.0/R2/train.jsonl R2_train.jsonl
cp anli_v1.0/R3/train.jsonl R3_train.jsonl

