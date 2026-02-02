from pathlib import Path

import pandas as pd
import joblib
from loguru import logger
import typer

from titanic.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "X_test.csv",
    raw_test_path: Path = RAW_DATA_DIR / "test.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    output_path: Path = Path("submission.csv"),
):
    """
    Génération des prédictions et création du fichier de soumission
    """

    logger.info("Chargement du modèle")
    model = joblib.load(model_path)

    logger.info("Chargement des features de test")
    X_test = pd.read_csv(features_path)

    logger.info("Chargement des données brutes de test")
    test_data = pd.read_csv(raw_test_path)

    logger.info("Prédiction")
    predictions = model.predict(X_test)

    output = pd.DataFrame(
        {
            "PassengerId": test_data["PassengerId"],
            "Survived": predictions,
        }
    )

    output.to_csv(output_path, index=False)
    logger.success(f"Fichier de soumission sauvegardé : {output_path}")


if __name__ == "__main__":
    app()