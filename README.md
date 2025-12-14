# MLOps Project

## Overview
This project is designed to implement Machine Learning Operations (MLOps) principles for managing the lifecycle of machine learning models. It includes data preparation, model training, evaluation, and deployment workflows.

## Project Structure
The project is organized into the following directories:

### 1. `data/`
- **`raw/`**: Contains raw, unprocessed data files.
- **`interim/`**: Stores intermediate data files after preprocessing.
- **`augmented/`**: Contains augmented datasets for model training.
- **`annotated/`**: Includes annotated datasets for supervised learning tasks.

### 2. `notebooks/`
- **`data_preparation.ipynb`**: Jupyter notebook for data cleaning and preprocessing.
- **`baseline_models.ipynb`**: Notebook for training and evaluating baseline models.
- **`annotation.ipynb`**: Notebook for annotating datasets.

### 3. `results/`
- **`models/`**: Stores metadata and artifacts related to trained models.



## Key Files
- **`data/raw/podcasts_without_transc.json`**: Raw dataset of podcasts without transcriptions.
- **`data/interim/podcasts_keywords_clean.csv`**: Preprocessed dataset with cleaned keywords.
- **`data/augmented/podcasts_keywords_augmented.csv`**: Augmented dataset for training.
- **`data/annotated/podcasts_annotated.csv`**: Annotated dataset with additional 
  
## Getting Started
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MLOps_Project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebooks in the `notebooks/` directory to preprocess data and train models.


