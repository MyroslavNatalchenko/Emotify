# Project Emotify

## Table of Contents:
- [Team Members](#team-members)
- [Development Workflow](#development-workflow-on-github)
- [Data Strategy & Model Training](#data-strategy--model-training)
  - [Baseline Training](#baseline-training-deam-dataset)
  - [Advanced Annotation](#advanced-annotation--augmentation-mert)

## Team Members

* **Myroslav Natalchenko**
* **Kiryl Sankouski**
* **Michał Zach**

## Development Workflow on GitHub

To ensure code stability and minimize merge conflicts, we will strictly follow a Fork & Branch workflow.

1.  Each team member must fork the main Emotify repository to their personal GitHub account
2.  Create a specific branch in your fork for your tasks
3.  Once task is complete, open a Pull Request (PR) from your fork's branch to the upstream repository's `main` branch


## Data Strategy & Model Training

To achieve robust emotional analysis in music, **Emotify** will utilize a two-staged approach regarding data ingestion and feature extraction.

### Baseline Training (DEAM Dataset)
We will initiate the training of our core model using the **DEAM (Database for Emotional Analysis in Music)** dataset. This provides a standardized benchmark for arousal and valence in music.
* **Source:** [DEAM MediaEval Dataset on Kaggle](https://www.kaggle.com/datasets/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music/data)
* **Goal:** Establish a baseline performance for emotion recognition.

### Advanced Annotation & Augmentation (MERT)
To expand our dataset and provide granular, high-quality audio annotations, we will integrate the **MERT-v1-330M** model. MERT (Music Understanding Model with Large-Scale Self-Supervised Training) will allow us to extract deep acoustic features and generate pseudo-labels for unlabelled data.
* **Model:** [m-a-p/MERT-v1-330M on Hugging Face](https://huggingface.co/m-a-p/MERT-v1-330M)
* **Goal:** Enhance the model's ability to detect subtle musical nuances and generalize better across different genres.