# Imports

import json
from joblib import dump
from joblib import load
from pathlib import Path
import os
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from functools import partial

# Absoluter Pfad zum 'models/'-Ordner
NOTEBOOKS_DIR = Path(os.getcwd())  # Aktuelles Arbeitsverzeichnis
PROJECT_DIR = NOTEBOOKS_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"  # models-Ordner relativ zur Wurzel

# Sicherstellen, dass der Ordner existiert
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def highlight_rows(row, first_indices, last_indices, color1, color2):
    '''
    Funktion zum Einfärben von DataFrame-Zellen mit angegebenen Farben.
    Es werden die ersten Zeilen und die letzten Zeilen gleich eingefärbt.

    Args:
    row (pd.Series): Die Zeile des DataFrames, die eingefärbt werden soll.
    first_indices (list): Indizes der ersten Zeilen.
    last_indices (list): Indizes der letzten Zeilen.
    color1 (str): Hex-Code der Farbe für die ersten Zeilen.
    color2 (str): Hex-Code der Farbe für die letzten Zeilen.

    Returns:
    pd.Series: Die Zeile des DataFrames mit den angewendeten Stil-Strings.
    '''
    if row.name in first_indices:
        return ['background-color: {}'.format(color1)] * len(row)
    elif row.name in last_indices:
        return ['background-color: {}'.format(color2)] * len(row)
    else:
        return [''] * len(row)

def create_highlight_func(df, color1, color2):
    """
    Erstellt eine Funktion zum Einfärben von DataFrame-Zellen mit angegebenen Farben.
    Es werden die ersten und die letzten Zeilen gleich eingefärbt.

    Args:
    df (pd.DataFrame): Der DataFrame, dessen Zeilen eingefärbt werden sollen.
    color1 (str): Hex-Code der Farbe für die ersten Zeilen.
    color2 (str): Hex-Code der Farbe für die letzten Zeilen.

    Returns:
    function: Eine Funktion, die auf eine pd.Series angewendet werden kann.
    """
    first_indices = df.index[:2]
    last_indices = df.index[-2:]

    return partial(highlight_rows, first_indices=first_indices, last_indices=last_indices, color1=color1, color2=color2)


    
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
    
def get_models_dir():
    """
    Gibt den absoluten Pfad zum models-Ordner zurück.
    
    Returns:
        str: Der absolute Pfad des models-Ordners.
    """
    return str(MODELS_DIR)

