# Resource Index for Zero-Shot and Few-Shot Commonsense Reasoning (CSR) Experiments

I find that several of my projects use the same data and models. This repository is to provide a unified resource index for these projects. 
With this repo, I do not need to download specific models and data for each project. Instead, I can simply clone this repo and use the data and models in this repo.

This repositoy contains code to prepare data, models and baseline methods for my experiments on CSR.

## Data 

The datasets and corresponding downloaders (scripts) are listed below. 

Usage (COPA example):

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
TBD

## Methods
Some scoring methods.
