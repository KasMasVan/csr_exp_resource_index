mkdir data/cqa/
cd data/cqa/
wget https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl -O dev.jsonl
wget https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl -O test_no_answers.jsonl  
wget https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl -O train.jsonl