[![License](https://img.shields.io/badge/License-Academic%20Non--Commercial-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

# CLM-LER: Clinical Language Models for Lab and Electronic Health Records

CLM-LER is a package designed to preprocess, train, and fine-tune transformer models on huggingface for tasks involving electronic health records (EHR) and laboratory data. It supports workflows for data preprocessing, model training, and evaluation, leveraging tools like PySpark, Hugging Face Transformers, and WandB for efficient and scalable operations.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Pretraining Pipeline Description](#pretraining-pipeline-description)
   - [Data Processing for Model Pre-Training](#data-processing-for-model-pre-training)
   - [Model Pre-Training](#model-pre-training)
   - [Model Fine-Tuning](#model-fine-tuning)
4. [Testing with EHRSHOT Benchmarks](#testing-with-ehrshot-benchmarks)
5. [UMLS for Mapping](#umls-for-mapping)
6. [Acknowledgements](#acknowledgements)

---

## Overview

CLM-LER provides a modular framework for working with EHR data. It includes utilities for:
- Preprocessing raw EHR data into tokenized formats.
- Training CLM models on large-scale EHR datasets.
- Fine-tuning models for specific downstream tasks like classification.
- Handling unit conversions, percentile calculations, and UMLS-based translations.

---

## Installation

### Prerequisites
- Python 3.10
- PySpark
- Hugging Face Transformers
- WandB (Weights and Biases)

### Work in a Conda Environment or Python Virtual Environment
Using a virtual environment prevents conflicting installations of packages. You can create one with the following:
```
conda create -n train-clm python=3.10 -y
conda activate train-clm
```

### Define Torch Dependencies
Specify the version of `torch` to install and the index for downloading it. Torch 2.0.1 (compiled for CUDA 117) was found to work well for this project. If you're using another version of CUDA, adjust the torch version accordingly.

### Basic Installation
Good for development work and training models interactively.
```
<install-torch> # e.g. pip install torch==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117/
pip install -e .[dev,train,spark]
```

### Full Installation
For all dependencies:
```
<install-torch> # e.g. pip install torch==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117/
pip install -e .[full]
```

### Installation Options:
dev -- dependencies to run unit tests and develop in this package.
train -- dependencies to train a model.
spark -- what you need for any data processing, i.e. a pyspark installation.

N.B. None of the packages above depend on torch explicitly, but torch is required to be installed.

## The Use of S3 
We used Amazon S3 for handling much of our I/O. We open source this work showcasing how AWS access keys with permissions to a s3 bucket may work. 
We encourage anyone following along to perform a similar setup to reduce friction in getting started! 

## The Use of W&B
Given how many clinical language models we were training to get toward our CLM-LER architecture, we used W&B for support and logging. 

We have online and offline options, but please raise an issue if you have some trouble!

To set this up:
```
wandb login
```

## Spark Cluster Setup (Optional and Encouraged)
We used spark clusters on our side to handle the tens of millions patients in EMRs we work with.
If you have AWS EMR, this could work for you! We suggest the use of [emrflow](https://github.com/Sanofi-Public/emrflow).

We removed this from the repo to increase adoption/minimize dependencies in the case you use another distributed compute tool (e.g. Databricks clusters).

## Environment setup of following
```
export CLM_AWS_ACCESS_KEY_ID=XYZ
export CLM_AWS_SECRET_ACCESS_KEY=XYZ
export CLM_AWS_DEFAULT_REGION=<region> # e.g. eu-west-1
export WANDB_API_KEY=XYZ
export WANDB_USERNAME=XYZ
export WANDB_ENTITY=XYZ
export CLMENCODER_DEPS=train,spark
```

---

# Pretraining Pipeline Description

There are four key stages handled by this package: data processing for model training, model pre-training, model fine-tuning, and testing with EHRSHOT benchmarks. _Note: Explainability has been split into a separate repository [here](https://github.com/Sanofi-Public/Clinical-BERT-Explainability?tab=readme-ov-file)._

---

## Data Processing for Model Pre-Training

The input datasets are defined in configuration files, such as `clm_ler/config/data_files_full.yaml`.

### Step 1: Create Train/Test/Validation Splits
The first step to building up the clinical language model, is to split our clinical data into train/val/test. The following script showcases how you may trigger a similar process:
```
bash scripts/create_global_data_split.sh
```

### Step 2: Arrange Data in the Required Format
The data must be arranged in the CLM-LER data model. For an example of how to preprocess the data, refer to the script `scripts/preprocess_data.py`. This script demonstrates the steps to preprocess patient, diagnosis, prescription, procedure, and lab data into tokenized formats.

---

## Step 3: Model Pre-Training

Once the data is preprocessed, follow these steps to pre-train a CLM model:

### Generate Vocabulary
Generate a vocabulary file for the model. See `scripts/preprocess_data.py` for an example!


### Train the Model
Train the CLM model using the preprocessed data and generated vocabulary.
Training a full CLM model typically takes about a week on an NVIDIA A10G GPU (e.g., g5.xlarge EC2 instance) for an EHR dataset with over 40M U.S patients.
See `scripts/preprocess_data.py` for usage example.

---

# Step 4: Model Fine-Tuning

Fine-tune the pre-trained model for specific tasks, such as classification.

### Prepare Data like Step 3 and add Labeled Data
Ensure the dataset includes a label column for the target classification task.

### Fine-Tune the Model
See `scripts/run_asthma_with_labs.sh` or `run_all_ehrshot_training.sh` scripts for examples on the fine-tuning call.

---

## Testing with EHRSHOT Benchmarks

EHRSHOT is a benchmarking dataset for evaluating model performance on various EHR-related tasks. Learn more: [EHRSHOT](https://github.com/som-shahlab/ehrshot-benchmark).
It provides multiple tasks for which a model can be finetuned. Using this dataset involves a few steps.

Firstly, if you are using a pre-trained model, you will want to map the EHRShot dataset's tokens to those expected by your model's vocabulary. This is handled by the script src/clm_ler/data_processing/process_ehrshot_data.py. This script takes a model and the raw data. Given a config file like src/clm_ler/config/mapping_config_to_clm_ler.yaml, the data in ehrshot and the models vocabulary is normalized into the names expected by UMLS. When running the script, you will be notified of any failed sources of codes that couldn't be sourced to UMLS. For example, this could be because you did not map ICD9 -> ICD9CM (The name of this source in UMLS.).

Secondly, this is a timeseries dataset with labelled events stored separately. Once you have created the dataset above, you need to join the labels into it, creating the dataset needed for inference and training. A config example is suppllied: src/clm_ler/config/config_add_labels_translated_data.yaml.

To see an example of processing data for CLM-LER, see `scripts/run_clmler_ehrshot_preprocess.py`.

---

## UMLS for Mapping

The Unified Medical Language System (UMLS) is used for mapping medical codes to standardized concepts. This ensures consistency across datasets and models.

- **Mapping Configurations**: See `clm_ler/config/mapping_config_to_clm_ler.yaml` for examples of UMLS mappings.
- **Translation Utilities**: The `clm_ler.data_processing.data_processing_utils` module provides functions for deriving UMLS translations.

For more details, refer to the [UMLS documentation](https://www.nlm.nih.gov/research/umls/index.html).

---

## Acknowledgements

This project leverages several key resources and contributions:

1. **UMLS (Unified Medical Language System)**  
   - The UMLS Metathesaurus is used for mapping medical codes to standardized concepts.  
   - Learn more: [UMLS](https://www.nlm.nih.gov/research/umls/index.html)
   - _Ref_: UMLS Knowledge Sources [dataset on the Internet]. Release 2024AA. Bethesda (MD): National Library of Medicine (US); 2024 May 6 [cited 2024 Jul 15]. Available from: http://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html

2. **EHRSHOT**  
   - The EHRSHOT benchmark datasets are used for evaluating model performance on various EHR-related tasks.  
   - Learn more: [EHRSHOT](https://github.com/som-shahlab/ehrshot-benchmark)
   - _Ref_: Michael Wornow, Rahul Thapa, Ethan Steinberg, Jason A. Fries, and Nigam H. Shah. 2023. EHRSHOT: an EHR benchmark for few-shot evaluation of foundation models. In Proceedings of the 37th International Conference on Neural Information Processing Systems (NIPS '23). Curran Associates Inc., Red Hook, NY, USA, Article 2933, 67125–67137.

3. **Inventors**  
   This project was developed by:
   - Lukas Adamek
   - Jenny Du
   - Maksim Kriukov
   - Towsif Rahman
   - Utkarsh Vashisth
   - Brandon Rufino

   Special thanks to the inventors for their contributions to the development of CLM-LER.
