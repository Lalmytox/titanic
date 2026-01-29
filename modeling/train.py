from pathlib import Path

import pandas as pd
import joblib
from loguru import logger
import typer
from sklearn.ensemble import RandomForestClassifier

from titanic.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "X_train.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "y_train.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    """
    Entraînement du modèle et sauvegarde
    """

    logger.info("Chargement des données d'entraînement")
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze("columns")

    logger.info("Initialisation du modèle RandomForest")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=1,
    )

    logger.info("Entraînement du modèle")
    model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    logger.success(f"Modèle sauvegardé dans {model_path}")


if __name__ == "__main__":
    app()
