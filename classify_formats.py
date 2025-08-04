import argparse
import pickle
from pathlib import Path
from unittest import case

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

def load_records(path: Path) -> pd.DataFrame:
    """
    Load pickled records and return a DataFrame.

    Args:
        path (Path): Path to the pickled file containing a list of dicts.

    Returns:
        pd.DataFrame: DataFrame constructed from the records.
    """
    with path.open('rb') as fp:
        records = pickle.load(fp)
    return pd.DataFrame(records)


def prepare_features(df: pd.DataFrame, alpha: float) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Filter DataFrame by alpha, extract feature matrix X and encoded labels y.

    Args:
        df (pd.DataFrame): Original DataFrame with 'alpha', 'entropy', and 'format' columns.
        alpha (float): Alpha value to filter on.

    Returns:
        X (np.ndarray): 2D array of flattened entropy features.
        y (np.ndarray): Label-encoded formats.
        le (LabelEncoder): Fitted label encoder for inverse transforms.
    """
    sub = df[(df['alpha'] == alpha)].reset_index(drop=True)
    # from sub, select only formats that contain the word 'chat', 'poem', or 'piece'
    sub = sub[sub['format'].str.contains('chat|poem|piece', case=False, na=False)]
    X = np.vstack(sub['entropy'].apply(lambda v: np.array(v).flatten()).to_list())
    le = LabelEncoder()
    y = le.fit_transform(sub['format'].values)
    return X, y, le


def train_and_evaluate_cv(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    neighbors_list: list[int],
    cv_folds: int = 10,
    random_state: int = 42
) -> None:
    """
    Train KNN classifiers with different neighbor settings using cross-validation.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Encoded labels.
        le (LabelEncoder): Fitted encoder for translating labels.
        neighbors_list (list[int]): List of k values to try.
        cv_folds (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.
    """
    print(f"\nPerforming {cv_folds}-fold cross-validation:")
    print("=" * 50)
    
    for k in neighbors_list:
        model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X, y, 
            cv=cv_folds, 
            scoring='accuracy'
        )
        
        mean_acc = cv_scores.mean()
        std_acc = cv_scores.std()
        
        print(f"k = {k}:")
        print(f"  Mean accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"  Individual fold scores: {[f'{score:.3f}' for score in cv_scores]}")
        print()


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    neighbors_list: list[int],
    test_size: float = 0.5,
    random_state: int = 42
) -> None:
    """
    Train KNN classifiers with different neighbor settings and print evaluation.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Encoded labels.
        le (LabelEncoder): Fitted encoder for translating labels.
        neighbors_list (list[int]): List of k values to try.
        test_size (float): Fraction of data reserved for testing.
        random_state (int): Random seed for reproducibility.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    for k in neighbors_list:
        model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"k = {k}: Test accuracy = {acc:.3f}")

        y_pred = model.predict(X_test)
        report = classification_report(
            y_test,
            y_pred,
            target_names=le.classes_,
            digits=3
        )
        print(report)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate KNN on entropy features."
    )
    parser.add_argument(
        '--data', '-d',
        type=Path,
        default=Path('./data/gemma/formats.pkl'),
        help='Path to the pickled data file.'
    )
    parser.add_argument(
        '--alpha', '-a',
        type=float,
        nargs='+',
        default=[0.5, 1, 5],
        help='List of alpha values to filter records.'
    )
    parser.add_argument(
        '--neighbors', '-k',
        type=int,
        nargs='+',
        default=[3, 5, 7],
        help='List of k values for KNN.'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data to use for testing.'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for train/test split.'
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    df = load_records(args.data)
    print(f"Loaded DataFrame with columns: {', '.join(df.columns)}")

    # Loop through each alpha value
    for alpha in args.alpha:
        print(f"\n{'='*60}")
        print(f"EVALUATING ALPHA = {alpha}")
        print(f"{'='*60}")
        
        X, y, le = prepare_features(df, alpha=alpha)
        print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features each.")
        
        if X.shape[0] == 0:
            print(f"No samples found for alpha = {alpha}. Skipping...")
            continue

        # Perform 10-fold cross-validation
        train_and_evaluate_cv(
            X,
            y,
            le,
            neighbors_list=args.neighbors,
            cv_folds=10,
            random_state=args.random_state
        )

        # Original train/test split evaluation for comparison
        print("\nTrain/Test Split Evaluation (for comparison):")
        print("=" * 50)
        train_and_evaluate(
            X,
            y,
            le,
            neighbors_list=args.neighbors,
            test_size=args.test_size,
            random_state=args.random_state
        )


if __name__ == '__main__':
    main()
