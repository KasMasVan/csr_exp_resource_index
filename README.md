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
- [ ] **Consider**: Do I need to rearrange the inference pipeline by starting with the method? Different method may require modifications to data and models. Alternatively, I could create a different pipeline for each method.
- [ ] Some experimends require going over multiple datasets, so maybe I can make this a feature. Without this faeture, a lot of time would be wasted on loading (large) models. Maybe I need to know how to "free" the memory used by a previous dataset when using the next one.
- [ ] Generate requirements.txt automatically?
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
|3 | [StableLM](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neox#gptneox) |`stabilityai/stablelm-base-alpha-3b`<br> `stabilityai/stablelm-tuned-alpha-3b` <br>`stabilityai/stablelm-base-alpha-7b`<br> `stabilityai/stablelm-tuned-alpha-7b` | Check [repo](https://github.com/Stability-AI/StableLM) for newer models. |


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

The following methods are designed for multiple choice tasks. Formally, the task requires a model to select an answer $\hat{y}$ from a set of options $Y = \{y_1, ..., y_n\}$ given a question $x$ using a scoring function $f$: $$ \hat{y}=argmax_i f(x, y_i) $$ 

The following methods focus on different specifications of the scoring method $f$.

Usage: 
```
TBD
```

### Supported Methods

| Index | Method      | Scoring Formula     | Script |
| ---   | ----------- | ----------- | ----------- |
| 1 | Language Modeling |   $P_{LM}(y_i \| x)$   | `./methods/language_modeling.py` |
| 2 | [Contrastive Decoding](https://arxiv.org/pdf/2210.15097.pdf) |   $\frac{P_{ExpertLM}(y_i \| x)}{P_{AmateurLM}(y_i \| x)}$  | `./methods/contrastive_decoding.py` |
| 3 | Multiple Choice Prompt |      |  |
| 4 | Process of Elimination |      |  |



### Useful Links
- [Ideas for zero-shot and few-shot commonsense reasoning](https://docs.google.com/document/d/1J8CmrKwgmApjZlp-HPDqHALPSvDYI30zihY-vXdExYY/edit?usp=sharing)
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
