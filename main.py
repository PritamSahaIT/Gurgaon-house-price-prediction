"""
Housing Price Prediction - End-to-End Machine Learning Pipeline

This module provides a production-grade, modular pipeline for:
1. Loading and stratifying housing data.
2. Building an automated preprocessing ColumnTransformer (imputation, scaling, one-hot encoding).
3. Training a RandomForestRegressor model and evaluating performance.
4. Serializing model & pipeline artifacts.
5. Performing batch inference on new inputs and exporting predictions.
"""

import argparse
import os
import sys
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Default configuration paths
DEFAULT_DATA_FILE = "housing.csv"
DEFAULT_INPUT_FILE = "input.csv"
DEFAULT_OUTPUT_FILE = "output.csv"
DEFAULT_MODEL_FILE = "model.pkl"
DEFAULT_PIPELINE_FILE = "pipeline.pkl"


def build_pipeline(num_attribs: List[str], cat_attribs: List[str]) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer pipeline for numerical and categorical features.

    Parameters:
        num_attribs (list): List of numerical column names.
        cat_attribs (list): List of categorical column names.

    Returns:
        ColumnTransformer: Preprocessing pipeline with imputation, scaling, and one-hot encoding.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    return full_pipeline


def stratified_split(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset using stratified sampling based on median income categories.
    This prevents sampling bias compared to pure random sampling.

    Parameters:
        df (pd.DataFrame): The raw housing dataframe.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_set, test_set) without the temporary income_cat column.
    """
    data = df.copy()
    data["income_cat"] = pd.cut(
        data["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(data, data["income_cat"]):
        train_set = data.loc[train_index].drop(columns=["income_cat"])
        test_set = data.loc[test_index].drop(columns=["income_cat"])

    return train_set, test_set


def train(
    data_path: str = DEFAULT_DATA_FILE,
    model_path: str = DEFAULT_MODEL_FILE,
    pipeline_path: str = DEFAULT_PIPELINE_FILE,
    test_output_path: str = DEFAULT_INPUT_FILE,
    random_state: int = 42,
) -> None:
    """
    Train the RandomForestRegressor model, evaluate metrics, and save artifacts.

    Parameters:
        data_path (str): Path to raw housing.csv dataset.
        model_path (str): Filepath to save the trained model (.pkl).
        pipeline_path (str): Filepath to save the fitted pipeline (.pkl).
        test_output_path (str): Filepath to save the test split (used as input.csv for inference).
        random_state (int): Random seed for reproducibility.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'. Please check the file path.")

    print(f"[*] Loading training dataset from: {data_path}")
    raw_data = pd.read_csv(data_path)
    print(f"[+] Loaded {len(raw_data):,} rows with {raw_data.shape[1]} columns.")

    print("[*] Performing stratified train/test split based on income categories...")
    train_set, test_set = stratified_split(raw_data, test_size=0.2, random_state=random_state)

    # Save test set as input.csv for batch prediction testing
    test_set.to_csv(test_output_path, index=False)
    print(f"[+] Saved {len(test_set):,} test records to: {test_output_path}")

    # Separate target variable and predictors
    target_col = "median_house_value"
    y_train = train_set[target_col].copy()
    X_train = train_set.drop(columns=[target_col])

    # Identify numeric and categorical features
    cat_attribs = ["ocean_proximity"]
    num_attribs = [col for col in X_train.columns if col not in cat_attribs]

    print(f"[*] Building preprocessing pipeline (Numerical: {len(num_attribs)}, Categorical: {len(cat_attribs)})...")
    pipeline = build_pipeline(num_attribs, cat_attribs)

    print("[*] Preprocessing training features...")
    X_train_prepared = pipeline.fit_transform(X_train)

    print("[*] Training RandomForestRegressor model...")
    model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    model.fit(X_train_prepared, y_train)

    # Compute training RMSE
    train_preds = model.predict(X_train_prepared)
    train_rmse = root_mean_squared_error(y_train, train_preds)
    print(f"[+] Model Training Completed! Training RMSE: ${train_rmse:,.2f}")

    # Serialize artifacts
    joblib.dump(model, model_path)
    joblib.dump(pipeline, pipeline_path)
    print(f"[+] Saved trained model to: {model_path}")
    print(f"[+] Saved preprocessing pipeline to: {pipeline_path}")


def predict(
    input_path: str = DEFAULT_INPUT_FILE,
    model_path: str = DEFAULT_MODEL_FILE,
    pipeline_path: str = DEFAULT_PIPELINE_FILE,
    output_path: str = DEFAULT_OUTPUT_FILE,
) -> None:
    """
    Perform batch inference on an input dataset and save predictions to output CSV.

    Parameters:
        input_path (str): Filepath to input CSV with features.
        model_path (str): Filepath to saved model (.pkl).
        pipeline_path (str): Filepath to saved pipeline (.pkl).
        output_path (str): Filepath to export predictions CSV.
    """
    for required_file in [input_path, model_path, pipeline_path]:
        if not os.path.exists(required_file):
            raise FileNotFoundError(
                f"Required file not found: '{required_file}'. "
                f"Please run training first using `python main.py --mode train`."
            )

    print(f"[*] Loading model from '{model_path}' and pipeline from '{pipeline_path}'...")
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)

    print(f"[*] Loading input data for inference from: {input_path}")
    input_data = pd.read_csv(input_path)

    # Separate features (drop target if present in input)
    target_col = "median_house_value"
    features = input_data.drop(columns=[target_col], errors="ignore")

    print("[*] Preprocessing input data through pipeline...")
    transformed_input = pipeline.transform(features)

    print("[*] Generating predictions...")
    predictions = model.predict(transformed_input)

    output_df = input_data.copy()
    output_df["predicted_median_house_value"] = predictions

    output_df.to_csv(output_path, index=False)
    print(f"[+] Inference complete! Results ({len(output_df):,} rows) saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Housing Price Prediction Pipeline - Modular ML System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "train", "predict"],
        default="auto",
        help="Execution mode: 'train', 'predict', or 'auto' (trains if model missing, else predicts).",
    )
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_FILE, help="Path to raw dataset CSV for training.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_FILE, help="Path to input CSV for inference.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE, help="Path for inference output CSV.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_FILE, help="Path to saved model (.pkl).")
    parser.add_argument("--pipeline", type=str, default=DEFAULT_PIPELINE_FILE, help="Path to saved pipeline (.pkl).")

    args = parser.parse_args()

    mode = args.mode
    if mode == "auto":
        if not os.path.exists(args.model) or not os.path.exists(args.pipeline):
            print("[INFO] Model or pipeline artifact not found. Launching training mode...")
            mode = "train"
        else:
            print("[INFO] Existing model artifacts detected. Launching inference mode...")
            mode = "predict"

    if mode == "train":
        train(
            data_path=args.data,
            model_path=args.model,
            pipeline_path=args.pipeline,
            test_output_path=args.input,
        )
    elif mode == "predict":
        predict(
            input_path=args.input,
            model_path=args.model,
            pipeline_path=args.pipeline,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()