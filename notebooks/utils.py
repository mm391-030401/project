# Imports

import json
from joblib import dump
from joblib import load

from pathlib import Path
import os

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from functools import partial

import altair as alt
import matplotlib.pyplot as plt

import pandas as pd
import datetime
import numpy as np
import itertools

from IPython.display import display
from IPython.display import HTML
from IPython.display import Image


# Absoluter Pfad zum 'models/'-Ordner
NOTEBOOKS_DIR = Path(os.getcwd())  # Aktuelles Arbeitsverzeichnis
PROJECT_DIR = NOTEBOOKS_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"  # models-Ordner relativ zur Wurzel

# Sicherstellen, dass der Ordner existiert
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def format_long_text(val): 
    '''
    Formatierte Darstellung von langen Texten in einer DataFrame-Spalte, um Zeilenumbrüche zu ermöglichen.

    Diese Funktion wird in einem Styler-Objekt verwendet, um Text in Zellen, die länger als 30 Zeichen sind, so zu formatieren,
    dass sie mit Zeilenumbrüchen angezeigt werden. Dabei wird die CSS-Eigenschaft `white-space: pre-wrap` verwendet, um den Text
    innerhalb der Zelle umzubrechen. Zudem wird die Breite der Zelle angepasst, wenn der Text länger ist.

    Args:
        val (str): Der Textwert, der in einer DataFrame-Zelle gespeichert ist. Diese Funktion überprüft, ob der Text
                   länger als 30 Zeichen ist, um den Text entsprechend zu formatieren.

    Returns:
        str: Eine CSS-Stilbeschreibung. Wenn der Text länger als 30 Zeichen ist, wird `white-space: pre-wrap; width: 300px;` zurückgegeben,
              andernfalls ein leerer String, der keine Formatierung vornimmt.
    '''

    if isinstance(val, str) and len(val) > 30: 
        return 'white-space: pre-wrap; width: 300px;' 
    else: 
        return ''

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
    '''
    Erstellt eine Funktion zum Einfärben von DataFrame-Zellen mit angegebenen Farben.
    Es werden die ersten und die letzten Zeilen gleich eingefärbt.

    Args:
    df (pd.DataFrame): Der DataFrame, dessen Zeilen eingefärbt werden sollen.
    color1 (str): Hex-Code der Farbe für die ersten Zeilen.
    color2 (str): Hex-Code der Farbe für die letzten Zeilen.

    Returns:
    function: Eine Funktion, die auf eine pd.Series angewendet werden kann.
    '''
    first_indices = df.index[:2]
    last_indices = df.index[-2:]

    return partial(highlight_rows, first_indices=first_indices, last_indices=last_indices, color1=color1, color2=color2)

def calc_corr(df, y, x, method='spearman'):
    '''
    Berechnet die Korrelation zwischen der Zielvariablen (y) und einer angegebenen Variable
    unter Verwendung der angegebenen Methode (z.B. 'spearman', 'pearson', etc.).

    Args:
    df (pandas.DataFrame): Der DataFrame, der die Daten enthält.
    y (str): Der Name der Zielvariablen.
    vx (str): Der Name der anderen Variable, mit der die Korrelation berechnet werden soll.
    method (str, optional): Die Methode zur Berechnung der Korrelation. Standardwert ist 'spearman'.
                            Mögliche Werte sind 'spearman', 'pearson' und andere Methoden, die von pandas .corr() unterstützt werden.
    
    Returns:
    None: Gibt die Korrelation gerundet auf zwei Dezimalstellen aus.
    '''
    # Berechnung der Korrelation mit der angegebenen Methode
    korr = df[[y, x]].corr(method=method).iloc[0, 1].round(2)
    
    # Ausgabe der Korrelation
    print(f'Korrelation zwischen {y} und {x} beträgt: {korr}')


def create_boxplot_with_count(df, y, x, color1, x_type='N', x_limits=None):
    '''
    Erstellt ein Boxplot und eine Count-Textanzeige für eine angegebene Variable in einem DataFrame.
    
    Args:
    df (pandas.DataFrame): Der DataFrame, der die Daten enthält.
    y (str): Der Name der Zielvariablen (z.B. eine kontinuierliche Variable).
    x (str): Der Name der Variablen für die x-Achse (z.B. 'monate_seit_existenz_kohorte').
    color1 (str): Die Farbe für das Diagramm.
    x_type (str, optional): Der Typ der x-Achse. Standard ist 'N' (nominal). 
                            Verwende 'Q' für quantitative oder 'O' für ordinale Variablen.
    x_limits (tuple, optional): Ein Tupel mit zwei Werten, das die unteren und oberen Grenzen der x-Achse definiert.
                                 Beispiel: (0, 24). Standardmäßig keine Begrenzung.
    
    Returns:
    alt.Chart: Ein kombiniertes Chart mit Boxplot und Count-Daten.
    '''
    
    # Spezifikation der x-Achse basierend auf dem angegebenen x_type
    x_spec = f'{x}:{x_type}'  # Standard ist :N, aber du kannst auch :Q oder :O verwenden
    
    # Wenn x_limits angegeben sind, wende sie auf die x-Achse an
    if x_limits is not None:
        x_axis = alt.X(x_spec, title=x, scale=alt.Scale(domain=x_limits))
    else:
        x_axis = alt.X(x_spec, title=x)
    
    # Boxplot
    boxplot_chart = alt.Chart(df).mark_boxplot(color=color1).encode(
        x=x_axis,
        y=alt.Y(y, title=y),
    ).properties(
        width=1000,
        height=400,
        title=f'Boxplot von {y} über {x}'
    )

    # Count-Daten
    count_data = df.groupby(x).size().reset_index(name='count')

    # Count als Text anzeigen
    count_chart = alt.Chart(count_data).mark_text(dy=-10, color=color1, fontWeight='bold', fontSize=13).encode(
        x=x_axis,
        y=alt.Y(value=+300),  # Position im Diagramm manuell festgelegt
        text='count:Q',
        tooltip=[  # Tooltip als Legendeersatz
            alt.Tooltip('count:Q', title='Anzahl der Werte innerhalb des Boxplots'),
        ]
    ).properties(
        width=1000,
        height=400
    )

    # Kombinieren der Charts
    combined_chart = boxplot_chart + count_chart

    return combined_chart

def generate_model_names(start_num, num_models, regression_type):
    model_names = []
    
    # Holen des aktuellen Datums und Uhrzeit im gewünschten Format
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    for i in range(num_models):
        # Erzeuge den Modellnamen mit fortlaufender Nummer und Regressionstyp als Variable
        model_name = f"{str(start_num + i).zfill(2)}_{current_timestamp}_{regression_type}"
        model_names.append(model_name)
    
    return model_names

def preprocess_data(data, y_label, features, drop_first=True):
    '''
    Bereitet die Daten für das Modell vor, einschließlich One-Hot-Encoding,
    und vermeidet Multikollinearität durch das Weglassen einer Kategorie pro Feature.

    Args:
        data (pd.DataFrame): Der ursprüngliche Datensatz.
        y_label (str): Der Name der Zielvariable.
        features (list): Die Liste der Features.
        drop_first (bool): Gibt an, ob eine Kategorie pro One-Hot-encoded Feature ausgeschlossen werden soll.

    Returns:
        pd.DataFrame, pd.Series: Features (X) und Zielvariable (y).
    '''
    model_data = data[[y_label] + features]
    model_data = pd.get_dummies(model_data, drop_first=drop_first)
    X = model_data.drop(columns=[y_label])
    y = model_data[y_label]
    return X, y

def train_and_validate_model(X, y, cv=5):
    '''
    Trainiert ein Modell und führt eine Kreuzvalidierung durch.

    Args:
        X (pd.DataFrame): Feature-Daten.
        y (pd.Series): Zielvariable.
        cv (int): Anzahl der Folds für die Kreuzvalidierung.

    Returns:
        LinearRegression, pd.DataFrame: Das trainierte Modell und die Kreuzvalidierungsergebnisse.
    '''
    reg = LinearRegression()
    scores = cross_val_score(reg, X, y, cv=cv, scoring='neg_mean_squared_error') * -1
    df_scores = pd.DataFrame({'MSE': scores})
    df_scores.index += 1
    reg.fit(X, y)
    return reg, df_scores

def evaluate_model(reg, X_test, y_test):
    '''
    Bewertet ein Modell anhand verschiedener Metriken und erstellt Residualdaten.

    Args:
        reg (LinearRegression): Das trainierte Modell.
        X_test (pd.DataFrame): Test-Features.
        y_test (pd.Series): Wahre Zielwerte.

    Returns:
        dict, pd.DataFrame: Metriken und Residualdaten.
    '''
    y_pred = reg.predict(X_test)
    residuals = y_test - y_pred
    metrics = {
        'R_squared': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': mean_squared_error(y_test, y_pred, squared=False),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
    residuals_df = pd.DataFrame({
        'True Values': y_test,
        'Predicted Values': y_pred,
        'Residuals': residuals
    })
    return metrics, residuals_df

def generate_summary_table(reg, X):
    '''
    Erstellt eine Tabelle mit Intercept und Koeffizienten des Modells.
    
    Args:
        reg (LinearRegression): Das trainierte Modell.
        X (pd.DataFrame): Die Feature-Daten (einschließlich One-Hot-Encoding).
    
    Returns:
        pd.DataFrame: Tabelle mit Intercept und Koeffizienten.
    '''
    intercept = pd.DataFrame({
        'Name': ['Intercept'],
        'Coefficient': [reg.intercept_]
    })
    
    # Verwendet die Spaltennamen von X nach dem One-Hot-Encoding
    slope = pd.DataFrame({
        'Name': X.columns,
        'Coefficient': reg.coef_
    })
    
    return pd.concat([intercept, slope], ignore_index=True, sort=False).round(3)

def save_validation_results(df_scores, model_name):
    '''
    Speichert die Kreuzvalidierungsergebnisse in einer CSV-Datei.

    Args:
        df_scores (pd.DataFrame): Kreuzvalidierungsergebnisse.
        model_name (str): Modellname für die Datei.

    Saves:
        Eine CSV-Datei im Ordner 'models/' mit den Kreuzvalidierungsergebnissen unter dem Namen '{model_name}_validation.csv'.
    '''
    file_path = MODELS_DIR / f'{model_name}_validation.csv'
    df_scores.to_csv(file_path, index=False)
    print(f'Kreuzvalidierungsergebnisse gespeichert unter: {file_path}')

def save_residual_plot(residuals_df, model_name):
    '''
    Erstellt und speichert einen Residualplot mit Matplotlib.

    Args:
        residuals_df (pd.DataFrame): Residualdaten.
        model_name (str): Modellname für die Datei.

    Saves:
        Eine PNG-Datei im Ordner 'models/' mit dem Residualplot unter dem Namen '{model_name}_residual_plot.png'.
    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(residuals_df['Predicted Values'], residuals_df['Residuals'], color='#06507F')
    plt.title(f'Residual Plot für Modell: {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    
    # Speichern des Plots als PNG
    file_path = MODELS_DIR / f'{model_name}_residual_plot.png'
    plt.savefig(str(file_path))
    plt.close()  # Schließe den Plot, damit er nicht im Notebook angezeigt wird

    print(f'Residualplot gespeichert unter: {file_path}')

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
    Berechnet und speichert die Evaluierungsmesswerte eines Modells in einer CSV-Datei im Ordner 'models/'.

    Args:
        y_test (array-like): Die wahren Zielwerte aus dem Testdatensatz.
        y_pred (array-like): Die vom Modell vorhergesagten Werte.
        model_name (str): Der Name des Modells, der auch für die Datei verwendet wird.
    
    Saves:
        Eine CSV-Datei im Ordner 'models/' mit den Messwerten (R², MSE, RMSE, MAE).
    """
    # Berechnung der Evaluierungsmesswerte
    results = {
        "R_squared": [r2_score(y_test, y_pred)],
        "MSE": [mean_squared_error(y_test, y_pred)],
        "RMSE": [mean_squared_error(y_test, y_pred, squared=False)],
        "MAE": [mean_absolute_error(y_test, y_pred)]
    }

    # Umwandlung der Ergebnisse in einen DataFrame
    results_df = pd.DataFrame(results)

    # Speichern des DataFrames als CSV-Datei
    file_path = MODELS_DIR / f"{model_name}_results.csv"
    results_df.to_csv(file_path, index=False)  # index=False entfernt den Index aus der CSV-Datei
    print(f"Ergebnisse gespeichert unter: {file_path}")

def save_summary_table(reg, X, model_name):
    '''
    Speichert die Zusammenfassungstabelle (Intercept und Koeffizienten) des Modells in einer CSV-Datei.

    Args:
        reg (LinearRegression): Das trainierte Modell.
        X (pd.DataFrame): Die Feature-Daten (einschließlich One-Hot-Encoding).
        model_name (str): Der Name des Modells, unter dem die Datei gespeichert werden soll.

    Saves:
        Eine CSV-Datei im Ordner 'models/' mit den Modellkoeffizienten unter dem Namen '{model_name}_summary.csv'.
    '''
    # Generiere die Zusammenfassungstabelle
    summary_table = generate_summary_table(reg, X)
    
    # Speichern der Tabelle als CSV
    file_path = MODELS_DIR / f'{model_name}_summary.csv'
    summary_table.to_csv(file_path, index=False)
    print(f'Zusammenfassungstabelle gespeichert unter: {file_path}')

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
    file_path = MODELS_DIR / f"{model_name}_results.csv"
    
    if os.path.exists(file_path):
        # CSV-Datei in einen DataFrame einlesen
        return pd.read_csv(file_path)
    else:
        print(f"Fehler: Die Datei für die Kreuzvalidierungsergebnisse von '{model_name}' wurde nicht gefunden.")
        return None
    
def load_residual_plot(model_name):
    '''
    Lädt den Residualplot als PNG und zeigt ihn im Notebook an.
    
    Args:
        model_name (str): Der Name des Modells, dessen Residualplot geladen werden soll.
    
    Returns:
        None
    '''
    file_path = MODELS_DIR / f'{model_name}_residual_plot.png'
    
    if os.path.exists(file_path):
        display(Image(filename=str(file_path)))
    else:
        print(f"Fehler: Die Datei für den Residualplot von '{model_name}' wurde nicht gefunden.")

def load_validation_results(model_name):
    '''
    Lädt die Kreuzvalidierungsergebnisse aus einer CSV-Datei im Ordner 'models/'.

    Args:
        model_name (str): Der Name des Modells, dessen Kreuzvalidierungsergebnisse geladen werden sollen.

    Returns:
        pd.DataFrame: Ein DataFrame mit den Kreuzvalidierungsergebnissen.
    '''
    file_path = MODELS_DIR / f"{model_name}_validation.csv"
    
    if os.path.exists(file_path):
        # CSV-Datei in einen DataFrame einlesen
        return pd.read_csv(file_path)
    else:
        print(f"Fehler: Die Datei für die Kreuzvalidierungsergebnisse von '{model_name}' wurde nicht gefunden.")
        return None
    
def load_summary_table(model_name):
    '''
    Lädt die Zusammenfassungstabelle (Intercept und Koeffizienten) aus einer CSV-Datei im Ordner 'models/'.

    Args:
        model_name (str): Der Name des Modells, dessen Zusammenfassung geladen werden soll.

    Returns:
        pd.DataFrame: Ein DataFrame mit der Zusammenfassungstabelle (Intercept und Koeffizienten).
    '''
    file_path = MODELS_DIR / f"{model_name}_summary.csv"
    
    if os.path.exists(file_path):
        # CSV-Datei in einen DataFrame einlesen
        return pd.read_csv(file_path)
    else:
        print(f"Fehler: Die Datei für die Zusammenfassungstabelle von '{model_name}' wurde nicht gefunden.")
        return None

def full_pipeline(data, y_label, features, model_name, is_log_transformed=False):
    '''
    Führt den gesamten Prozess der Modellbildung und Bewertung durch, mit optionaler logarithmischer Rücktransformation.

    Args:
        data (pd.DataFrame): Eingabedaten.
        y_label (str): Zielvariable.
        features (list): Feature-Liste.
        model_name (str): Name des Modells.
        is_log_transformed (bool): Gibt an, ob die Zielvariable logarithmisch transformiert wurde.
    '''
    # Preprocessing
    X, y = preprocess_data(data, y_label, features)

    # Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training and Validation
    reg, df_scores = train_and_validate_model(X_train, y_train)

    # Evaluation
    metrics, residuals_df = evaluate_model(reg, X_test, y_test)

    # Vorhersagen
    y_pred_transformed = reg.predict(X_test)
    
    # Rücktransformation der Vorhersagen auf die Originalskala
    if is_log_transformed:
        y_pred_original = np.exp(y_pred_transformed)  # Rücktransformation der Vorhersagen
        y_test = np.exp(y_test)
    else:
        y_pred_original = y_pred_transformed  # Keine Rücktransformation der Vorhersagen

    # Residuen auf der Originalskala berechnen
    residuals = y_test - y_pred_original  # Residuen nach Rücktransformation (oder nicht)

    # Summary Table
    summary_table = generate_summary_table(reg, X)

    # Speichern der Ergebnisse
    save_model(reg, model_name)
    save_features(features, model_name)
    save_results(y_test, y_pred_original, model_name)  # Ergebnisse auf der Originalskala speichern
    save_validation_results(df_scores, model_name)
    save_residual_plot(residuals_df, model_name)
    save_summary_table(reg, X, model_name)


    
def run_full_pipeline_with_featurelist(data, y_label, feature_list, modeltype, start_num, is_log_transformed=False):
    """
    Führt die Pipeline für verschiedene Feature-Sets aus und generiert Modellnamen.

    Args:
        data (pd.DataFrame): Datensatz.
        y_label (str): Zielvariable.
        feature_list (list): Liste von Feature-Sets.
        is_log_transformed (bool): Gibt an, ob die Zielvariable logarithmisch transformiert wurde.
        modeltype (str): Typ des Modells (z. B. "Lineare_Regression").
        start_num (int): Startnummer für die Modellnummerierung.

    Returns:
        list: Liste der generierten Modellnamen.
    """
    # Leere Liste für Modellnamen erstellen
    model_names_list= []
    
    for features in feature_list:
        # Generiere den Modellnamen (dies gibt eine Liste zurück)
        model_names = generate_model_names(start_num, 1, modeltype)
        
        # Entpacke den ersten Modellnamen (oder den gewünschten Index)
        model_name = model_names[0]  # Hier entpackst du den ersten Namen der Liste
        
        # Füge den Modellnamen zur Liste hinzu
        model_names_list.append(model_name)
        
        # Rufe die Pipeline-Funktion auf
        full_pipeline(data, y_label, features=features, model_name=model_name, is_log_transformed=is_log_transformed)
        
        # Erhöhe die Modellnummer für den nächsten Durchgang
        start_num += 1

    return model_names_list


def generate_results_df(model_names_list):
    """
    Generiert einen kombinierten DataFrame mit den Ergebnissen und Features für eine Liste von Modellen.

    Args:
        model_names_list (list): Liste von Modellnamen.

    Returns:
        pd.DataFrame: Ein DataFrame mit den kombinierten Ergebnissen und Features.
    """

    # Leere Liste für alle DataFrames erstellen
    all_results = []

    # Iteration durch die Modellnamen und Laden der Ergebnisse
    for model_name in model_names_list:
        # Lade die Evaluierungsergebnisse des Modells
        results_df = load_results(model_name)
        
        if results_df is not None:
            # Lade die verwendeten Features
            features = load_features(model_name)
            
            # Füge Modellnamen und Features als zusätzliche Spalten hinzu
            results_df['Model'] = model_name
            results_df['Features'] = str(features)  # Features als String speichern
            
            # Füge das DataFrame der Liste hinzu
            all_results.append(results_df)

    # Kombiniere alle DataFrames in der Liste zu einem einzigen DataFrame
    final_results_df = pd.concat(all_results, ignore_index=True)

    # Setze den Modellnamen als Index
    final_results_df.set_index('Model', inplace=True)

    return final_results_df


def generate_validation_stats_df(model_names_list):
    """
    Generiert einen kombinierten DataFrame mit deskriptiven Statistiken 
    für die Kreuzvalidierungsergebnisse einer Liste von Modellen.

    Args:
        model_names_list (list): Liste von Modellnamen.

    Returns:
        pd.DataFrame: Ein DataFrame mit den kombinierten deskriptiven Statistiken,
                      wobei der Modellname als zusätzlicher Index verwendet wird.
    """

    # Liste zur Speicherung der Statistiken
    stats_list = []

    # Iteration durch alle Modellnamen
    for model_name in model_names_list:
        # Lade die Kreuzvalidierungsergebnisse für das Modell
        df_scores = load_validation_results(model_name)
        
        if df_scores is not None:
            # Berechne deskriptive Statistiken und transponiere sie
            stats = df_scores.describe().T
            
            # Füge den Modellnamen als Spalte hinzu
            stats['Model'] = model_name
            
            # Füge die Statistiken zur Liste hinzu
            stats_list.append(stats)

    # Kombiniere alle Statistiken in einem DataFrame
    combined_stats_df = pd.concat(stats_list)

    # Setze den Modellnamen als Index
    combined_stats_df = combined_stats_df.set_index('Model', append=True)

    return combined_stats_df

def generate_final_df(model_names_list, top_n=15, sort_column='R_squared', sort = False):
    """
    Generiert einen formatierten und gestylten DataFrame aus Modell- und Validierungsstatistiken.

    Args:
        model_names_list (list): Liste der Modellnamen.
        top_n (int): Anzahl der Top-Modelle, sortiert nach der angegebenen Spalte, die angezeigt werden sollen.
        sort_column (str): Die Spalte, nach der sortiert werden soll.
        sort (bool): Sortierreihenfolge, True für aufsteigend, False für absteigend. Standard ist absteigend.

    Returns:
        pd.io.formats.style.Styler: Ein gestylter DataFrame mit den Top-n Modellen.
    """

    # Generiere die Ergebnisse- und Validierungsstatistik-DataFrames
    results_df = generate_results_df(model_names_list)
    validation_stats_df = generate_validation_stats_df(model_names_list)

    # Präfix zu den Spalten von validation_stats_df hinzufügen
    validation_stats_df = validation_stats_df.add_prefix("validation_")

    # Beide DataFrames anhand des Indexes joinen
    final_df = results_df.join(validation_stats_df, how="inner")

    # Sortiere nach der angegebenen Spalte und wähle die Top-n Modelle aus
    final_df = final_df.sort_values(by=sort_column, ascending=sort).head(top_n)

    final_df = final_df.reset_index()

    final_df = final_df.drop(columns=['level_0'])

    # Anwenden des Stylers
    styled_final_df = final_df.style.applymap(format_long_text, subset=['Features'])

    # Formatierung für die numerischen Spalten
    numeric_columns = [
        'R_squared', 'MSE', 'RMSE', 'MAE', 'validation_count', 'validation_mean', 
        'validation_std', 'validation_min', 'validation_25%', 'validation_50%', 
        'validation_75%', 'validation_max'
    ]
    styled_final_df = styled_final_df.format("{:.3f}", subset=numeric_columns)

    # Linksbündige Darstellung des Texts in allen Spalten
    styled_final_df = styled_final_df.set_properties(**{'text-align': 'left'})

    return styled_final_df

def add_cyclic_features(data, columns, cycle_length=12):
    """
    Fügt zyklische Kodierungen (Sinus und Cosinus) für angegebene Spalten eines DataFrames hinzu.

    Args:
        data (pd.DataFrame): Der DataFrame, der die zu kodierenden Spalten enthält.
        columns (list of str): Liste der Spaltennamen, die zyklisch kodiert werden sollen.
        cycle_length (int, optional): Die Länge des Zyklus, z. B. 12 für Monate im Jahr. Standard ist 12.

    Returns:
        pd.DataFrame: Der DataFrame mit den hinzugefügten zyklischen Kodierungen.
    """
    for column in columns:
        data[f'sin_{column}'] = np.sin(2 * np.pi * (data[column] - 1) / cycle_length)
        data[f'cos_{column}'] = np.cos(2 * np.pi * (data[column] - 1) / cycle_length)
    return data


def save_styled_dataframe(styled_df, filepath):
    """
    Speichert ein Pandas-Styler-Objekt als HTML-Datei.

    Args:
        styled_df (pd.io.formats.style.Styler): Das zu speichernde Styler-Objekt.
        filepath (str): Der Pfad, unter dem die HTML-Datei gespeichert wird.

    Returns:
        None
    """
    # Konvertiere das Styler-Objekt in HTML
    html = styled_df.to_html()

    # Schreibe das HTML in die Datei
    with open(filepath, 'w') as f:
        f.write(html)

def load_styled_dataframe(filepath):
    """
    Lädt ein gespeichertes HTML-Styler-Objekt und gibt es zur Anzeige zurück.

    Args:
        filepath (str): Der Pfad zur gespeicherten HTML-Datei.

    Returns:
        IPython.core.display.HTML: Das geladene Styler-Objekt zur Anzeige.
    """
    # Lade das HTML aus der Datei
    with open(filepath, 'r') as f:
        html = f.read()

    # Gib das HTML zur Anzeige zurück
    return HTML(html)

def generate_combinations(strings):
    """
    Generiert alle möglichen Kombinationen von mindestens einem Element aus der gegebenen Liste von Strings.
    
    Args:
        strings (list): Eine Liste von Strings, die die Variablen repräsentieren.
        
    Returns:
        list: Eine Liste von Listen, wobei jede Liste eine Kombination von Strings enthält.
    """
    # Generiere alle möglichen Kombinationen (mindestens eine der Strings)
    combinations = []
    for r in range(1, len(strings) + 1):
        combinations.extend(itertools.combinations(strings, r))
    
    # Umwandeln in eine Liste von Listen
    return [list(comb) for comb in combinations]