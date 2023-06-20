# Process of Elimination for Multiple Choice Reasoning

The official implementation of our paper *Process of Elimination for Multiple Choice Reasoning*.


## Prerequisites

Install dependencies:
```
pip install -r requirements.txt
```
Download data (replace `task_name` with names in `data_downloaders/` like `anli`):
```
bash data/data_downloaders/task_name.sh
```
Download models:
```
bash models/model_downloaders/model_downloaders.sh
```

## Reproduce Results
Run scripts in `methods/` to reproduce results in the paper:
```
cd methods
bash main_exp.sh
bash llm.sh
bash few_shot.sh
bash mask.sh
bash num_option.sh
```
The results will be saved in `results/`.