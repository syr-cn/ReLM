# ReLM: Leveraging Language Models for Enhanced Chemical Reaction Prediction

EMNLP 2023 \[[paper](https://arxiv.org/abs/2310.13590v1)\]

Authors: Yaorui Shi, An Zhang, Enzhi Zhang, Zhiyuan Liu, Xiang Wang

This repository contains the official code impementation for the paper **ReLM: Leveraging Language Models for Enhanced Chemical Reaction Prediction**.

## Installation

```bash
conda create -n ReLLM python=3.8
conda activate ReLLM
pip install -r requirements.txt
```

## Reproducing results

To reproduce the results, you may need to run the bash scripts in the scripts folder:
```bash
sh scripts/run_Imidazo.sh
sh scripts/run_NiCOlit.sh
sh scripts/run_Rexgen30k.sh
sh scripts/run_Rexgen40k.sh
```

## Citation
```latex
@inproceedings{
    ReLM,
    title={ReLM: Leveraging Language Models for Enhanced Chemical Reaction Prediction},
    author={Yaorui Shi, An Zhang, Enzhi Zhang, Zhiyuan Liu, Xiang Wang},
    booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
    year={2023}
}
```
