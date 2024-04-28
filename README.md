# chatbot_tutor - Dissertation Project Repository

## Overview
This repository contains all the code used in my dissertation on the optimization of large language models for educational purposes. The primary research question I explore is:

> Can prompt engineering, retrieval augmented generation (RAG), and fine-tuning be used to tailor large language models for more effective teaching, measured through the lens of the International Baccalaureate Biology syllabus?

## Research Context
This study applies RAG to OpenAI's GPT-3.5-Turbo and GPT-4, and fine-tunes Mistral-7B and Biomistral-DARE-7B, aiming to enhance their capabilities in educational capabilities within the International Baccalaureate (IB) Biology domain. To evaluate the models' performance, specialized evaluation metrics and pipelines have been designed, incorporating both objective quantitative benchmarks and qualitative measures.
## Repository Structure
This repository is organized into several key folders, each containing Jupyter Notebooks and scripts pertinent to different aspects of the research:

- **`/main/scrape_bioninja`**: Contains notebooks used for scraping data from BioNinja, relevant to the IB Biology syllabus.
- **`/main/RAG/v0`**: Includes notebooks and scripts for implementing and testing the Retrieval Augmented Generation models.
- **`/main/Finetune`**: Contains notebooks for fine-tuning the Mistral and Biomistral models on the custom datasets created as part of this study.
- **`/main/Evaluation`**: Holds all notebooks related to the evaluation of the models, including the creation of a custom multiple-choice question dataset and implementation of the Elo ranking system to compare model variants.

### Notebooks and Python scripts
- `scrape_bioninja.ipynb` - Scrapes and preprocesses content from BioNinja.
- `custom_rag.ipynb`, `rag_models.py` - Sets up the retriever and generator for RAG models.
- `Make_finetune_dataset.ipynb`, `mistral-finetune.ipynb`, `biomistral-dare-finetune.ipynb` - Notebooks for dataset creation and model fine-tuning.
- Evaluation notebooks (`generate_responses_qualitative_benchmark.ipynb`, `analysis_results.ipynb`, `eval_models_p1.ipynb`, `create_benchmark_ds.ipynb`, `chatbotarena.ipynb`) - Detailed evaluation procedures and metrics reporting.

## Setup and Installation
All code was originally implemented on Google Colab Pro+ with A100 GPUs for fine-tuning. Each notebook includes the necessary `pip install` commands for running in a Google Colab environment.

### Requirements
A general `requirements.txt` file is included in the root of this repository. It lists all libraries used across the different scripts and notebooks. To install the required libraries:

```bash
pip install -r requirements.txt
```

## Data and Models
Note: All datasets and model weights are not included in this repository due to their large sizes.

