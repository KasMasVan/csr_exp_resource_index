# Zero-Shot and Few-Shot Prompting for Commonsense Reasoning (CSR)

I find that several of my projects use the same data and models. This repository is to provide a unified resource index for these projects. 
With this repo, I do not need to download specific models and data for each project. Instead, I can simply clone this repo and use the data and models in this repo.

This repositoy contains code to prepare data, models and baseline methods for my experiments on CSR.


## To Do
- [x] Reproducibility to PMI_DC and my other projects.
- [ ] **Important**: Consider the form of prompts, especially some nuances like the whitespace between a semicolon and an answer. You may use existing templates instead. 
- [ ] More language model. For larger models, you may load with different precisions like FP16 and INT8 (Check [this](https://huggingface.co/docs/transformers/main/en/performance)).
- [ ] More data. 
- [ ] **More [methods](#methods)**. 
- [ ] Use relative path in your python scripts (for models and data, etc.), so that running them from different location would be okay.
- [x] **Important**: Aggregate datasets in an evaluation pipeline to avoid loading the same model multiple times. Consider aggregating similar methods as well.
- [ ] Generate requirements.txt automatically?
- [ ] TBD.

## Data 

You can check [this website](https://cs.nyu.edu/~davise/Benchmarks/) for a comprehensive collection of commonsense reasoning benchmarks, which are mentioned in the [accompanying paper](https://arxiv.org/pdf/2302.04752.pdf).

I have tried several benchmarks. Their names and downloaders (scripts) are listed below.

Usage (at the root directory):

 ```
 bash data/data_downloaders/copa.sh
 ```

### Multiple-Choice Commonsense Reasoning Benchmarks

| Index | Data      | Number of Options | Data Downloader |
| --- | ----------- | --- |----------- |
| 1 | [COPA](https://people.ict.usc.edu/~gordon/copa.html)      | 2 | `./data/data_downloaders/copa.sh`       |
| 2 | [CommonsenseQA](https://aclanthology.org/N19-1421/)   | 5|`./data/data_downloaders/cqa.sh`        |
| 3 | [OpenBookQA](https://allenai.org/data/open-book-qa)   | 4|`./data/data_downloaders/obqa.sh`        |
| 4 | [Winogrande](https://leaderboard.allenai.org/winogrande/submissions/get-started)  | 2 | `./data/data_downloaders/winogrande.sh`        |
| 5 | [PIQA](https://yonatanbisk.com/piqa/)  |2 | `./data/data_downloaders/piqa.sh`        |
| 6 | [Social IQA](https://leaderboard.allenai.org/socialiqa/submissions/get-started)  |3 | `./data/data_downloaders/siqa.sh`        |
| 7 | [ARC](https://allenai.org/data/arc)   | 4 |    |
| 8 | [QASC](https://leaderboard.allenai.org/qasc/submissions/get-started)   | 8 |  `bash data/data_downloaders/qasc.sh`  |
| 9 | [HellaSWAG]()   | 4 |    |
| 10 | [StrategyQA]()   |  2   |    |
| 11 | [MC-TACO]()   |   2  |    |


### Other Multiple-Choice Benchmarks

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
|Index| Model Family| Checkpoints| Training Data|  Note    |
| --- | ----------- | -----------| --- |--------  |
|1.0 | [GPT2](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2#openai-gpt2) |`gpt2`<br> `gpt2-medium` <br>`gpt2-large`<br> `gpt2-xl` |WebText (Private) | |
|2.0 | [GPT-Neo](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neo#gpt-neo) |`EleutherAI/gpt-neo-125m`<br> `EleutherAI/gpt-neo-1.3B` <br>`EleutherAI/gpt-neo-2.7B`<br> `EleutherAI/gpt-j-6b` <br> `EleutherAI/gpt-neox-20b` | [Pile](https://pile.eleuther.ai/) |Similar structure to GPT-2 and GPT-3. |
|2.1 | [Pythia](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neox#gptneox) |`EleutherAI/pythia-size`<br> `EleutherAI/pythia-size-deduped` | [Pile](https://pile.eleuther.ai/) and its deduplicated version | Available `size`: `[70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b]` Check [repo](https://github.com/EleutherAI/pythia).  *deliberately designed to promote scientific research on large language models, especially interpretability research* |
|3.0 | [OPT](https://huggingface.co/docs/transformers/main/en/model_doc/opt#opt) |`facebook/opt-size`|BookCorpus, CC-Stories, The Pile, Pushshift.io Reddit dataset, CCNewsV2  | Available `size`: `[125m, 350m, 1.3b, 2.7b, 6.7b, 13b, 30b, 66b]` |
|3.1 | [OPT-IML](https://huggingface.co/docs/transformers/main/en/model_doc/opt#opt) |`facebook/opt-iml-1.3b` <br> `facebook/opt-iml-30b`<br> `facebook/opt-iml-max-1.3b` <br> `facebook/opt-iml-max-30b` <br> | [OPT-IML Bench](https://arxiv.org/abs/2212.12017)  | *Instruction-tuned versions of OPT on a collection of ~2000 NLP tasks gathered from 8 NLP benchmarks.* |
|2 | [Dolly](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neox#gptneox) |`databricks/dolly-v1-6b` <br> `databricks/dolly-v2-3b`<br> `databricks/dolly-v2-7b`<br> `databricks/dolly-v2-12b` | Downloader Code Not Yet |
|3 | [StableLM](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neox#gptneox) |`stabilityai/stablelm-base-alpha-3b`<br> `stabilityai/stablelm-tuned-alpha-3b` <br>`stabilityai/stablelm-base-alpha-7b`<br> `stabilityai/stablelm-tuned-alpha-7b` | Check [repo](https://github.com/Stability-AI/StableLM) for newer models. |
|4 | [Phoenix](https://huggingface.co/docs/transformers/main/en/model_doc/bloom#bloom) |`FreedomIntelligence/phoenix-chat-7b`<br> `FreedomIntelligence/phoenix-inst-chat-7b` | Check [repo](https://github.com/FreedomIntelligence/LLMZoo) for newer models. |
|5 | MOSS |`fnlp/moss-moon-003-base`<br> `fnlp/moss-moon-003-sft` <br> `fnlp/moss-moon-003-sft-plugin` | Bilingual model: Check [repo](https://github.com/OpenLMLab/MOSS) for newer models. |
|7 | [MPT](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neox#gptneox) |`mosaicml/mpt-7b`<br> `mosaicml/mpt-7b-chat` <br> `mosaicml/mpt-7b-instruct` <br> `mosaicml/mpt-7b-storywriter` |  Check [repo](https://github.com/mosaicml/llm-foundry). |
|8 | [RedPajama-INCITE](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neox#gptneox) |`togethercomputer/RedPajama-INCITE-Base-3B-v1`<br> `togethercomputer/RedPajama-INCITE-Base-7B-v0.1` |  Check [blog](https://www.together.xyz/blog/redpajama-models-v1) for more models and other details. |
|9 | [StarCoder](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neox#gptneox) |`bigcode/starcoderbase`<br> `bigcode/starcoder` |  Check [blog](https://huggingface.co/blog/starcoder) for more models and other details. |

#### Encoder-Decoder Models
|Index| Model Family| Checkpoints| Notes |
| --- | ----------- | ----------- | ----- |
|1 | [T5](https://huggingface.co/docs/transformers/main/en/model_doc/t5#t5) |`t5-small` <br> `t5-base`<br> `t5-large`<br> `t5-3b`<br> `t5-11b` |
|2 | [FLAN-T5](https://huggingface.co/docs/transformers/main/en/model_doc/t5#t5) |`google/flan-t5-small` <br> `google/flan-t5-base`<br> `google/flan-t5-large`<br> `google/flan-t5-xl`<br> `google/flan-t5-xxl` | Chcek page 47 of [paper](https://arxiv.org/pdf/2210.11416.pdf) for instruction tuning data |


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

The following methods are designed for multiple choice tasks. Formally, the task requires a model to select an answer $\hat{y}$ from a set of options $Y = \{y_1, ..., y_n\}$ given a question $x$: $$\hat{y} = select(x, Y )$$ 

Usage: 
```
cd methods
bash my_exp.sh
```

### Zero-Shot Scoring Methods (Single-Step)

Scoring methods calculate a score for each option, and then select the option with the highest score: $$ \hat{y}=argmax_i score(y_i) $$

Supported methods are organized by hierarchy, which means some methods are more fundamental than others. For example, language modeling is the most fundamental method, and it is used by other methods.

| Index | Method      | Scoring Function     | Script |
| ---   | ----------- | ----------- | ----------- |
| 0.0 | Language Modeling |   $P_{LM}(y_i \| x)$   | `./methods/language_modeling.py` |
| 0.1 | Average Language Modeling |   $P_{LM}(y_i \| x)^{1/len(y_i)}$   | `./methods/language_modeling.py` |
| 1.x | [Multiple Choice Prompt](https://openreview.net/forum?id=yKbprarjc5B) |   $P_{LM}(symbol(y_i) \| T_{mcp}(x))$   | `./methods/language_modeling.py` |
| 2.x | [Contrastive Decoding](https://arxiv.org/pdf/2210.15097.pdf) |   $\frac{P_{ExpertLM}(y_i \| x)}{P_{AmateurLM}(y_i \| x)}$  | `./methods/contrastive_decoding.py` |


### Other Methods (Multi-Step)

These methods choose the answer in multiple steps, which may involve generation and (multi-step) prompting.

| Index | Method      | Prodecure     | Script |
| ---   | ----------- | ----------- | ----------- |
| 1 | [SEQA](https://aclanthology.org/2021.acl-long.237) |  1. $\{s_i\} = genenerate(x)$  <br> 2. $\hat{y} = argmax_jsim(y_j, \{s_i\})$ | [Official Implementation](https://github.com/heyLinsir/Semantic-based-QA) |
| 2 | [Process of Elimination](https://docs.google.com/document/d/14aRC2C6-fb64hDW5lFTQSjOuOrfK1ItoNJw17VTaEes/edit?usp=sharing) |  1. $score_i= score(y_i)$  <br> 2. $Y^\prime = \{y_i \| score_i > threshold\}$ <br> 3. $\hat{y} = prompting([demonstrations,] x, Y^\prime)$| `./methods/process_of_elimination.py` |
| 3 | Generate Similar Text | 1. Generate similar text for an option. <br> 2. Compute probabilities for option as well as similar text. <br> 3. Aggregate, e.g., averaging, to get final score for a option, and choose the best option. | n/a |

### Useful Links
- [Google Doc: Ideas for zero-shot and few-shot commonsense reasoning](https://docs.google.com/document/d/1J8CmrKwgmApjZlp-HPDqHALPSvDYI30zihY-vXdExYY/edit?usp=sharing)
- [Repo: Natural Instruction](https://github.com/allenai/natural-instructions): A benchmark that annotate instructions for many tasks.
- [Repo: Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide/tree/main): A comprehensive collection.
- [Repo: PromptSouce](https://github.com/bigscience-workshop/promptsource): A repo for prompts.
- [HF: Causal Language Modeling](https://huggingface.co/docs/transformers/tasks/language_modeling)
- [HF: Multiple Choice](https://huggingface.co/docs/transformers/tasks/multiple_choice)
- [HF: Scripts](https://huggingface.co/docs/transformers/run_scripts)
- [HF: Fine-tuning](https://huggingface.co/docs/transformers/training#train-in-native-pytorch)
- [HF: Performance and Scalability](https://huggingface.co/docs/transformers/main/en/performance)
- [HF: Text Generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [HF: Text Generation Startegies](https://huggingface.co/docs/transformers/generation_strategies)
- [Paper: Surface Form Competition](https://aclanthology.org/2021.emnlp-main.564.pdf): Remember to compare your results with the results in this paper.
- [My Onenotes on multiple choice]
- [HF: Using Datasets with Pytorch](https://huggingface.co/docs/datasets/use_with_pytorch#use-with-pytorch): Also check the [Datasets Documentation](https://huggingface.co/docs/datasets/index)
- [Pytorch: Data and Dataloader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Pytorch: Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
- [Pytorch: CUDA](https://pytorch.org/docs/stable/notes/cuda.html#cuda-semantics)
- [Pytorch: no_grad() and eval()](https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615): I learnt the hard way that `torch.no_grad()` should be used together with `model.eval()` when testing models.
- [Deep Learning: Tuning Playbook](https://github.com/google-research/tuning_playbook)
- [Blog: Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
