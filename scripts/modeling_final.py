import pandas as pd
import ast
import warnings
from time import time
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

RAW_DATA_PATH = 'data/processed/cleaned_tv_shows.csv'
TRANSFORMED_DATA_PATH = 'data/processed/transformed_tv_shows.csv'
REPORT_PATH = 'data/processed/modeling_report_multilabel.txt'
MODELS_DIR = Path('data/processed/models')

RANDOM_STATE = 42


def parse_genres(x):
    try:
        return ast.literal_eval(x)
    except:
        return []


def slugify(value: str) -> str:
    return ''.join(ch if ch.isalnum() else '_' for ch in value).strip('_').lower()


def load_and_prep_dataset(csv_path: str, label: str):
    print(f"Loading dataset ({label})...")

    df = pd.read_csv(csv_path)

    # Parse genres
    genre_col = 'genres_parsed' if 'genres_parsed' in df.columns else 'genres'
    df['genres_list'] = df[genre_col].apply(parse_genres)

    # MULTI-LABEL TARGET BUILDING
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df['genres_list'])
    genre_classes = mlb.classes_

    # Combine text features
    title_bits = df['title'].fillna('')
    if 'title_normalized' in df.columns:
        title_bits = (title_bits + ' ' + df['title_normalized'].fillna('')).str.strip()

    df['text_all'] = (title_bits + ' ' + df['description'].fillna('')).str.strip()

    numeric_candidates = [
        'rating_avg_minmax_01',
        'runtime_minutes_zscore',
        'release_year_decscale',
        'premiere_month',
        'premiere_dayofweek',
        'premiere_year',
        'premiere_decade',
        'rating_avg',
        'runtime_minutes',
        'release_year',
    ]
    categorical_candidates = [
        'language',
        'type',
        'status',
        'on_netflix',
        'on_disney',
        'on_amazon',
        'on_hulu',
    ]

    numeric_features = [c for c in numeric_candidates if c in df.columns]
    categorical_features = [c for c in categorical_candidates if c in df.columns]
    text_feature = 'text_all'

    X = df[numeric_features + categorical_features + [text_feature]]

    # ColumnTransformer
    transformers = []

    if numeric_features:
        transformers.append(
            ('num', SimpleImputer(strategy='median'), numeric_features)
        )

    if categorical_features:
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('cat', cat_transformer, categorical_features))

    transformers.append(
        ('txt', Pipeline([('tfidf', TfidfVectorizer(max_features=8000, stop_words='english'))]), text_feature)
    )

    preprocessor = ColumnTransformer(transformers=transformers)

    feature_summary = {
        "numeric": numeric_features,
        "categorical": categorical_features,
        "text": [text_feature]
    }

    return X, Y, preprocessor, feature_summary, genre_classes


def train_and_evaluate(X, Y, preprocessor, genre_classes):
    print("\nTraining models (multi-label)...")

    stratify_y = Y if getattr(Y, "ndim", 1) == 1 else None
    if stratify_y is not None and getattr(stratify_y, "ndim", 1) > 1:
        stratify_y = None

    if stratify_y is None:
        print("  -> Skipping stratification: multi-label target can't be stratified safely.")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_y
    )

    base_models = {
        'Logistic Regression': LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, random_state=RANDOM_STATE)
    }

    # Wrap into MultiOutputClassifier
    models = {
        name: MultiOutputClassifier(model)
        for name, model in base_models.items()
    }

    results = []
    best_f1 = -1
    best_model = None
    best_name = None

    model_items = list(models.items())
    durations = []

    with tqdm(total=len(model_items), desc="Training models", unit="model") as pbar:
        for idx, (name, model) in enumerate(model_items):

            start_time = time()
            pbar.set_description(f"Training {name}")

            clf = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            clf.fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)

            duration = time() - start_time
            durations.append(duration)
            eta = (len(model_items) - idx - 1) * (sum(durations) / len(durations))
            pbar.set_postfix({"last_s": f"{duration:.1f}", "eta_s": f"{eta:.1f}"})
            pbar.update(1)

            # Multi-label metrics
            acc = accuracy_score(Y_test, Y_pred)
            prec = precision_score(Y_test, Y_pred, average='macro', zero_division=0)
            rec = recall_score(Y_test, Y_pred, average='macro', zero_division=0)
            f1 = f1_score(Y_test, Y_pred, average='macro', zero_division=0)

            # Per-genre metrics
            per_genre = {}
            for i, genre in enumerate(genre_classes):
                per_genre[genre] = {
                    "prec": precision_score(Y_test[:, i], Y_pred[:, i], zero_division=0),
                    "rec": recall_score(Y_test[:, i], Y_pred[:, i], zero_division=0),
                    "f1": f1_score(Y_test[:, i], Y_pred[:, i], zero_division=0)
                }

            results.append({
                "name": name,
                "model": clf,
                "macro_acc": acc,
                "macro_prec": prec,
                "macro_rec": rec,
                "macro_f1": f1,
                "per_genre": per_genre
            })

            if f1 > best_f1:
                best_f1 = f1
                best_model = clf
                best_name = name

    return best_model, best_name, best_f1, results


def main():
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("TV Show Project - Multi-Label Modeling Report\n")
        f.write("============================================\n\n")

        datasets = [
            ("Raw cleaned dataset", RAW_DATA_PATH),
            ("Transformed dataset", TRANSFORMED_DATA_PATH),
        ]

        for label, path in datasets:

            f.write(f"{'=' * 50}\n{label}\n{'=' * 50}\n")

            X, Y, preprocessor, feature_info, genre_classes = load_and_prep_dataset(path, label)

            f.write("Features used:\n")
            f.write(f"  Numeric: {feature_info['numeric']}\n")
            f.write(f"  Categorical: {feature_info['categorical']}\n")
            f.write(f"  Text: {feature_info['text']}\n\n")

            best_model, best_name, best_f1, results = train_and_evaluate(
                X, Y, preprocessor, genre_classes
            )

            for res in results:
                f.write(f"Model: {res['name']}\n")
                f.write(f"Macro Metrics:\n")
                f.write(f"  Accuracy: {res['macro_acc']:.4f}\n")
                f.write(f"  Precision: {res['macro_prec']:.4f}\n")
                f.write(f"  Recall: {res['macro_rec']:.4f}\n")
                f.write(f"  F1: {res['macro_f1']:.4f}\n\n")

                f.write("Per-Genre Metrics:\n")
                for genre, g in res['per_genre'].items():
                    f.write(f"  {genre}: F1={g['f1']:.4f}, Prec={g['prec']:.4f}, Rec={g['rec']:.4f}\n")
                f.write("\n" + "-" * 30 + "\n\n")

            f.write(f"Best Model: {best_name} (Macro F1={best_f1:.4f})\n")

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            slug = slugify(best_name)
            ds_slug = slugify(label)
            model_path = MODELS_DIR / f"{ds_slug}__multilabel__{slug}.joblib"

            joblib.dump(best_model, model_path)
            f.write(f"Saved best model to: {model_path}\n\n")

    print("Training complete.")
    print(f"Report written to {REPORT_PATH}")

if __name__ == "__main__":
    main()
