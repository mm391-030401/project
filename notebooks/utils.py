# Imports

import json
from joblib import dump
from joblib import load
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Absoluter Pfad zum 'models/'-Ordner
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Gehe zwei Ebenen nach oben (vom utils-Skript)
MODELS_DIR = PROJECT_ROOT / "models"

# Sicherstellen, dass der Ordner existiert
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def save_model(model, model_name):
    """
    Speichert ein trainiertes Modell in einer .joblib-Datei im Ordner 'models/'.

    Args:
        model: Das trainierte Modell (z. B. sklearn-Regressor).
        model_name (str): Der Name des Modells, der auch für die Datei verwendet wird.
    
    Saves:
        Eine .joblib-Datei im Ordner 'models/' mit dem Namen '{model_name}.joblib'.
    """
    file_path = MODELS_DIR / f"{model_name}.joblib"
    dump(model, file_path)
    print(f"Modell gespeichert unter: {file_path}")

def save_features(features, model_name):
    """
    Speichert die verwendeten Features in einer JSON-Datei im Ordner 'models/'.

    Args:
        features (list): Eine Liste der Feature-Namen.
        model_name (str): Der Name des Modells, der auch für die Datei verwendet wird.
    
    Saves:
        Eine JSON-Datei im Ordner 'models/' mit dem Namen '{model_name}_features.json'.
    """
    file_path = MODELS_DIR / f"{model_name}_features.json"
    with open(file_path, 'w') as f:
        json.dump(features, f)
    print(f"Features gespeichert unter: {file_path}")

def save_results(y_test, y_pred, model_name):
    """
    Berechnet und speichert die Evaluierungsmesswerte eines Modells in einer JSON-Datei im Ordner 'models/'.

    Args:
        y_test (array-like): Die wahren Zielwerte aus dem Testdatensatz.
        y_pred (array-like): Die vom Modell vorhergesagten Werte.
        model_name (str): Der Name des Modells, der auch für die Datei verwendet wird.
    
    Saves:
        Eine JSON-Datei im Ordner 'models/' mit den Messwerten (R², MSE, RMSE, MAE).
    """
    results = {
        "R_squared": r2_score(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "MAE": mean_absolute_error(y_test, y_pred)
    }
    file_path = MODELS_DIR / f"{model_name}_results.json"
    with open(file_path, 'w') as f:
        json.dump(results, f)
    print(f"Ergebnisse gespeichert unter: {file_path}")

def save_full_workflow(model, features, y_test, y_pred, model_name):
    """
    Führt den gesamten Speicherprozess aus:
    1. Speichert das Modell.
    2. Speichert die verwendeten Features.
    3. Berechnet und speichert die Evaluierungsmesswerte.

    Args:
        model: Das trainierte Modell (z. B. sklearn-Regressor).
        features (list): Eine Liste der Feature-Namen.
        y_test (array-like): Die wahren Zielwerte aus dem Testdatensatz.
        y_pred (array-like): Die vom Modell vorhergesagten Werte.
        model_name (str): Der Name des Modells, der auch für die Dateien verwendet wird.
    
    Saves:
        - Eine .joblib-Datei im Ordner 'models/'.
        - Eine JSON-Datei für die Features im Ordner 'models/'.
        - Eine JSON-Datei für die Evaluierungsmesswerte im Ordner 'models/'.
    """
    save_model(model, model_name)
    save_features(features, model_name)
    save_results(y_test, y_pred, model_name)
    print(f"Gesamter Workflow für {model_name} abgeschlossen.")

def load_model(model_name):
    """
    Lädt ein gespeichertes Modell aus einer .joblib-Datei im Ordner 'models/'.

    Args:
        model_name (str): Der Name des Modells, das geladen werden soll.
    
    Returns:
        Das geladene Modell.
    """
    file_path = MODELS_DIR / f"{model_name}.joblib"
    return load(file_path)

def load_features(model_name):
    """
    Lädt die gespeicherten Features aus einer JSON-Datei im Ordner 'models/'.

    Args:
        model_name (str): Der Name des Modells, dessen Features geladen werden sollen.
    
    Returns:
        list: Die Liste der gespeicherten Feature-Namen.
    """
    file_path = MODELS_DIR / f"{model_name}_features.json"
    with open(file_path, 'r') as f:
        return json.load(f)

def load_results(model_name):
    """
    Lädt die gespeicherten Evaluierungsmesswerte aus einer JSON-Datei im Ordner 'models/'.

    Args:
        model_name (str): Der Name des Modells, dessen Ergebnisse geladen werden sollen.
    
    Returns:
        dict: Ein Dictionary mit den Evaluierungsmesswerten (R², MSE, RMSE, MAE).
    """
    file_path = MODELS_DIR / f"{model_name}_results.json"
    with open(file_path, 'r') as f:
        return json.load(f)
