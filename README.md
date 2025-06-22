# Tradutor
This repository contains the code and resources for training a model to translate text from English to European Portuguese.

## Overview

The translations produced by our model, as well as baseline translations for the FRMT and NTrex benchmarks, can be found in the [cache](cache/) directory. The statistics for automatic evaluation are available in the [results](results/) folder. To print a summary table of the results in the terminal, run the following command:

```bash
python scripts/print_results.py
```

## Development Environment Setup

Follow these steps to set up your development environment:

1. Create a `.env` file with the keys defined in the `.env.example` file.

2. Install the project dependencies:

    ```bash
    pip install -e .
    ```

Your environment should now be ready for use.

## Training the Model

This repository uses `torchtune` with customized training recipes for training the LLaMA-3 models and `trl` for training the Phi-3 and Gemma-2 models.

All training scripts are located in the [train](scripts/train/) directory. For example, to launch the full fine-tuning training script for the Gemma-2 model, run the following command in the terminal:

```bash
sh scripts/train/gemma2_fft.sh
```

## Evaluating Translations

To obtain the evaluation metrics presented in the paper, run the following command:

```bash
sh scripts/eval.sh
```

This script contains all the necessary commands to evaluate the models.


## Citation

If you use this model in your work, please cite the following paper:
```
@article{Sousa2025,
  author    = {Hugo Sousa and Satya Almasian and Ricardo Campos and Alipio Jorge},
  title     = {Tradutor: Building a Variety Specific Translation Model},
  journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {39},
  number    = {24},
  pages     = {25183--25191},
  year      = {2025},
  doi       = {10.1609/aaai.v39i24.34704},
  issn      = {2374-3468},
  month     = {April}
}
```
