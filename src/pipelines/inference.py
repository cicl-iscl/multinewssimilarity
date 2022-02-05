"""
Inference Pipeline for regression using multiple models.

Steps:
1. Get & Split data into training and eval data
2. pass the data to train pipeline, Wandb Log scores and hyperparams, Evaluate models scores and store models
3. pass the trained models to inference pipeline and predict overall scores on test data using various models
4. Score clipping
5. Generate CSVs
6. add_missing_entries
"""
import argparse
import wandb
from pandas import read_csv
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, cross_validate
from src.config import (CLEANED_PATH, UNCLEANED_PATH, RESULT_PATH,
                        MODEL_PATH, RAW_FILE, INFERENCE_FILE, WandbParams, DataType)

from src.logger import log


def _get_data_splits():
    train_path = CLEANED_PATH.format(data_type=DataType.train.name) + INFERENCE_FILE
    log.info(f"Using {train_path} for training linear regression.")

    train_df = read_csv(train_path, sep=',', usecols=['title', 'sentences', 'start_para', 'overall'])

    y = train_df.pop('overall')
    x = train_df
    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2, random_state=5)
    return train_x, val_x, train_y, val_y


def _train_pipeline(features, labels, models=None):
    """
    If model=None then train all the models. Log the hyperparameters.
    """
    train_x, val_x, train_y, val_y = _get_data_splits()
    wandb.init(config=hyperparameters)
    _log_hyperparameters()


def _inference_pipeline(models, test_features):
    """
    Return: Dict containing models name as key and prediction scores on test data.
    """
    test_path = CLEANED_PATH.format(data_type=DataType.test.name) + INFERENCE_FILE
    test_df = read_csv(test_path, sep=',', usecols=['title', 'sentences', 'start_para'])


def _log_hyperparameters():
    pass


def _clip_scores(scores):
    pass


def _eval_model(predicted_scores, true_scores):
    pass


def _generate_csv(pair_ids, models_predictions_dict, model_):
    pass


def _add_missing_entries(source_path, target_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train models and predict regression scores.')
