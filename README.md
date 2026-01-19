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

### Baseline Training (MTG-Jamendo Dataset)
To ensure robust generalization, we utilize the **MTG-Jamendo Dataset** for baseline training. Unlike smaller datasets limited to the Arousal-Valence plane, MTG-Jamendo provides a massive collection of Creative Commons audio annotated with rich semantic tags.

**Key Resources:**
* **Official Repository:** [MTG-Jamendo-Dataset](https://github.com/MTG/mtg-jamendo-dataset/tree/master)
* **Data Statistics & Balance:**
  We specifically focus on the `mood/theme` split. To understand the dataset's composition and the number of samples available for specific emotions (e.g., *happy, dark, energetic*), refer to the official statistics:
  [> Link: Detailed Class Counts & Distribution](https://github.com/MTG/mtg-jamendo-dataset/blob/master/stats/raw_30s_cleantags_50artists/mood_theme.tsv)

**Objective:**
Leverage this large-scale data to train a model capable of predicting high-level emotional descriptors from raw audio spectrograms.
