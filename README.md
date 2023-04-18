# Resource Index for Zero-Shot and Few-Shot Commonsense Reasoning (CSR) Experiments

I find that several of my projects use the same data and models. This repository is to provide a unified resource index for these projects. 
With this repo, I do not need to download specific models and data for each project. Instead, I can simply clone this repo and use the data and models in this repo.

This repositoy contains code to prepare data, models and baseline methods for my experiments on CSR.


## To Do
- [ ] Report experiment result for reproducibility?
- [ ] More language model. Specifically, add support for causal language models.
- [ ] More data. 
- [ ] **More methods**. 
- [ ] Use relative path in your python scripts (for models and data, etc.), so that running them from different location would be okay.
- [ ] TBD.

## Data 

You can check [this website](https://cs.nyu.edu/~davise/Benchmarks/) for a comprehensive collection of commonsense reasoning benchmarks, which are mentioned in the [accompanying paper](https://arxiv.org/pdf/2302.04752.pdf).

I have tried several benchmarks. Their names and downloaders (scripts) are listed below.

Usage (at the root directory):

 ```
 bash data/data_downloaders/copa.sh
 ```

| Index | Data      | Data Downloader |
| --- | ----------- | ----------- |
| 1 | [COPA](https://people.ict.usc.edu/~gordon/copa.html)      | `./data/data_downloaders/copa.sh`       |
| 2 | [CommonsenseQA](https://aclanthology.org/N19-1421/)   | `./data/data_downloaders/cqa.sh`        |
| 3 | [OpenBookQA](https://allenai.org/data/open-book-qa)   | `./data/data_downloaders/obqa.sh`        |
| 4 | [Winogrande](https://leaderboard.allenai.org/winogrande/submissions/get-started)   | `./data/data_downloaders/winogrande.sh`        |
| 5 | [PIQA](https://yonatanbisk.com/piqa/)   | `./data/data_downloaders/piqa.sh`        |
| 6 | [Social IQA](https://leaderboard.allenai.org/socialiqa/submissions/get-started)   | `./data/data_downloaders/siqa.sh`        |

## Models
The language models are listed below. For small language models, I provide the downloaders for the model with the most common head, i.e., language modeling head for GPT2. If you want to download a model with a different head, you may need to tweak the code of the corresponding downloader.

For large language modles, **TBD**.

### Small Languae Models
These models can be stored locally, and are avialable on either Hugging Face or Github. **Autoencoding models** (BERT-like models) are excluded, because they are not suitable for generation. 

Usage (at the root directory): 
```
bash models/model_downloaders/model_downloaders.sh
```

#### Autoregressive Models
|Index| Model Family| Checkpoints| Note |
| --- | ----------- | ----------- | --- |
|1 | [GPT2](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2#openai-gpt2) |`gpt2`<br> `gpt2-medium` <br>`gpt2-large`<br> `gpt2-xl` | |
|2 | [Dolly](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neox#gptneox) |`databricks/dolly-v1-6b` <br> `databricks/dolly-v2-3b`<br> `databricks/dolly-v2-7b`<br> `databricks/dolly-v2-12b` | Code Not Yet |


#### Encoder-Decoder Models
|Index| Model Family| Checkpoints| 
| --- | ----------- | ----------- | 
|1 | [T5](https://huggingface.co/docs/transformers/main/en/model_doc/t5#t5) |`t5-small` <br> `t5-base`<br> `t5-large`<br> `t5-3b`<br> `t5-11b` |
|2 | [FLAN-T5](https://huggingface.co/docs/transformers/main/en/model_doc/t5#t5) |`google/flan-t5-small` <br> `google/flan-t5-base`<br> `google/flan-t5-large`<br> `google/flan-t5-xl`<br> `google/flan-t5-xxl` |


#### Other Models

- [Open Assistant models](https://huggingface.co/OpenAssistant)

### Large Language Models
These models cannot be stored locally, and are accessed through API. For now, the only option is ChatGPT, because it is very powerful, and much cheaper than other GPT-3.5 models.

Usage: 
```
TBD
```

| Index | Model Family      | Checkpoints |
| --- | ----------- | ----------- |
|1| ChatGPT | `gpt-3.5-turbo` <br> `gpt-3.5-turbo-0301` |

## Methods

Usage: 
```
TBD
```

### Supported Methods

1. [Language Modeling]
2. [Contrastive Decoding](https://arxiv.org/pdf/2210.15097.pdf)

### Useful Links
- [HF: Causal Language Modeling](https://huggingface.co/docs/transformers/tasks/language_modeling)
- [HF: Multiple Choice](https://huggingface.co/docs/transformers/tasks/multiple_choice)
- [HF: Scripts](https://huggingface.co/docs/transformers/run_scripts)
- [HF: Fine-tuning](https://huggingface.co/docs/transformers/training#train-in-native-pytorch)
- [Paper: Surface Form Competition](https://aclanthology.org/2021.emnlp-main.564.pdf): Remember to compare your results with the results in this paper.
- [My Onenotes on multiple choice]
- [HF: Using Datasets with Pytorch](https://huggingface.co/docs/datasets/use_with_pytorch#use-with-pytorch): Also check the [Datasets Documentation](https://huggingface.co/docs/datasets/index)
- [Pytorch: Data and Dataloader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Pytorch: Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
- [Pytorch: CUDA](https://pytorch.org/docs/stable/notes/cuda.html#cuda-semantics)
- [Pytorch: no_grad() and eval()](https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615): I learnt the hard way that `torch.no_grad()` should be used together with `model.eval()` when testing models.
- [Deep Learning: Tuning Playbook](https://github.com/google-research/tuning_playbook)