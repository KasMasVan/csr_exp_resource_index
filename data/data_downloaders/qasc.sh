mkdir data/qasc/
cd data/qasc/
wget https://s3-us-west-2.amazonaws.com/data.allenai.org/downloads/qasc/qasc_dataset.tar.gz
tar -xvf qasc_dataset.tar.gz
cp QASC_Dataset/{dev,test}.jsonl .
