
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from scipy.stats import norm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import confusion_matrix

from scipy.stats import spearmanr
import elapid
from elapid import (
    MaxentModel,
    GeographicKFold,
    annotate,
    apply_model_to_rasters,
    xy_to_geoseries,
)

warnings.filterwarnings("ignore")

# 1. CONFIGURACIÓN

# ── Modo de entrada ──────────────────────────────────────────────────────────
# "csv"    → tienes un CSV con coordenadas y variables ya extraídas
# "raster" → tienes archivos .tif y un CSV solo con coordenadas
# "ambos"  → tienes ambos; el script usará rasters para la predicción espacial
#             y CSV para el entrenamiento (más rápido)
MODO_ENTRADA = "ambos"

PRESENCIAS_CSV   = ""     
BACKGROUND_CSV   = ""    

RASTER_DIR       = ""     
OUTPUT_DIR       = Path("")


COL_LAT  = "lat"
COL_LON  = "lon"
# Si MODO_ENTRADA = "csv" o "ambos", lista las columnas de variables:
VARIABLE_COLS = []  

FEATURE_TYPES   = [] 
BETA_MULTIPLIER = 1             # RM óptimo


N_BACKGROUND    = 10_000   

N_FOLDS_GEO     = 5   
N_FOLDS_RANDOM  = 5  
RANDOM_STATE    = 42

CRS = "EPSG:4326"   # WGS84 — coordenadas geográficas lat/lon


# 2. FUNCIONES DE CARGA


def csv_a_geodataframe(csv_path: str, col_lat: str, col_lon: str,
                       crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    df  = pd.read_csv(csv_path)
    # Construimos la columna 'geometry' con objetos Point de shapely
    geo = gpd.GeoDataFrame(
        df,
        geometry = gpd.points_from_xy(df[col_lon], df[col_lat]),
        crs      = crs,
    )
    return geo


def detectar_variable_cols(gdf: gpd.GeoDataFrame,
                            coord_cols: list,
                            variable_cols: list) -> list:
    if variable_cols:
        return variable_cols
    excluir = set(coord_cols + ["geometry"])
    return [c for c in gdf.columns if c not in excluir]


def muestrear_background(raster_paths: list, n: int,
                          random_state: int = 42) -> gpd.GeoDataFrame:
    
    np.random.seed(random_state)
    bg_points = elapid.sample_raster(raster_paths[0], count=n)
    return bg_points


def extraer_variables_de_rasters(puntos_gdf: gpd.GeoDataFrame,
                                  raster_paths: list,
                                  nombres: list = None) -> gpd.GeoDataFrame:
    if nombres is None:
        nombres = [Path(p).stem for p in raster_paths]

    anotado = annotate(
        points       = puntos_gdf,
        raster_paths = raster_paths,
        labels       = nombres,
        drop_na      = True,   # elimina puntos que caigan en nodata
    )
    return anotado
                                      
# 3. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO


def entrenar_maxent(X: pd.DataFrame, y: np.ndarray,
                    feature_types: list, beta_multiplier: float,
                    random_state: int = 42) -> MaxentModel:
    model = MaxentModel(
        feature_types   = feature_types,
        beta_multiplier = beta_multiplier,
        transform       = "cloglog",   
        clamp           = True,    
        use_sklearn     = True,     
        random_state    = random_state,
    )

    model.fit(X, y)
    return model
                        
# 4. VALIDACIÓN CRUZADA


def validacion_cruzada_geografica(pres_gdf: gpd.GeoDataFrame,
                                   back_gdf: gpd.GeoDataFrame,
                                   var_cols: list,
                                   feature_types: list,
                                   beta_multiplier: float,
                                   n_folds: int,
                                   random_state: int) -> pd.DataFrame:

    print(f"\n  Validación cruzada GEOGRÁFICA ({n_folds} folds espaciales)")
    print(f"  {'─'*50}")

    gkf    = GeographicKFold(n_splits=n_folds, random_state=random_state)
    splits = list(gkf.split(pres_gdf))   # genera índices train/test

    resultados = []
    
    predicciones_geo = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):

        pres_train = pres_gdf.iloc[train_idx]
        pres_test  = pres_gdf.iloc[test_idx]
    
        multipoint = MultiPoint(pres_test.geometry.tolist())
        hull = multipoint.convex_hull 
        
        back_in_block = back_gdf[back_gdf.geometry.within(hull)]
        
        if len(back_in_block) < 30:
            print(f" Fold {fold_idx+1}: solo {len(back_in_block)} puntos de fondo en bloque; usando buffer")
            hull_buffered = hull.buffer(0.1)
            back_in_block = back_gdf[back_gdf.geometry.within(hull_buffered)]
            

        n_back_test = min(len(pres_test) * 10, len(back_in_block))
        if n_back_test < len(pres_test):
            print(f"    ⚠ Fold {fold_idx+1}: muy pocos fondos en bloque ({len(back_in_block)}), el índice Boyce puede ser inestable.")
        X_test_back = back_in_block[var_cols].sample(n_back_test, random_state=fold_idx) if n_back_test > 0 else pd.DataFrame(columns=var_cols)

        X_train_pres = pres_train[var_cols]
        X_train_back = back_gdf[var_cols]
        X_train = pd.concat([X_train_pres, X_train_back], ignore_index=True)
        y_train = np.array([1]*len(X_train_pres) + [0]*len(X_train_back))

        X_test_pres = pres_test[var_cols]
        X_test = pd.concat([X_test_pres, X_test_back], ignore_index=True)
        y_test = np.array([1]*len(X_test_pres) + [0]*len(X_test_back))


        try:
            model = entrenar_maxent(X_train, y_train, feature_types,
                                    beta_multiplier, random_state)
            preds = model.predict(X_test)
            predicciones_geo.append((y_test.copy(), preds.copy()))
            auc   = roc_auc_score(y_test, preds)
            tss   = calcular_tss(y_test, preds)
            boyce = calcular_boyce(preds[y_test == 1], preds[y_test == 0])

            print(f"    Fold {fold_idx+1}: AUC={auc:.4f}  TSS={tss:.4f}  "
                  f"Boyce={boyce:.4f}  "
                  f"(n_test={len(pres_test)} pres)")
        except Exception as e:
            print(f"    Fold {fold_idx+1}: ERROR — {e}")
            auc, tss, boyce = np.nan, np.nan, np.nan
            predicciones_geo.append((None, None))

        resultados.append({
            "fold"     : fold_idx + 1,
            "tipo"     : "geografico",
            "n_pres_train": len(pres_train),
            "n_pres_test" : len(pres_test),
            "auc"      : auc,
            "tss"      : tss,
            "boyce"    : boyce,
        })

    return pd.DataFrame(resultados), predicciones_geo


def validacion_cruzada_aleatoria(X: pd.DataFrame, y: np.ndarray,
                                  feature_types: list,
                                  beta_multiplier: float,
                                  n_folds: int,
                                  random_state: int) -> pd.DataFrame:

    print(f"\n  Validación cruzada ALEATORIA ({n_folds} folds, stratified)")
    print(f"  {'─'*50}")

    skf        = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                 random_state=random_state)
    resultados = []
    predicciones = []
    all_preds  = np.zeros(len(y))   # para curva ROC agregada
    all_true   = y.copy()

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            model = entrenar_maxent(X_train, y_train, feature_types,
                                    beta_multiplier, random_state)
            preds = model.predict(X_test)
            predicciones.append((y_test.copy(), preds.copy()))
            all_preds[test_idx] = preds

            auc   = roc_auc_score(y_test, preds)
            tss   = calcular_tss(y_test, preds)
            boyce = calcular_boyce(preds[y_test == 1], preds[y_test == 0])

            print(f"    Fold {fold_idx+1}: AUC={auc:.4f}  TSS={tss:.4f}  "
                  f"Boyce={boyce:.4f}")
        except Exception as e:
            print(f"    Fold {fold_idx+1}: ERROR — {e}")
            auc, tss, boyce = np.nan, np.nan, np.nan
            predicciones.append((None, None))

        resultados.append({
            "fold"  : fold_idx + 1,
            "tipo"  : "aleatorio",
            "n_pres_train": int(y_train.sum()),
            "n_pres_test" : int(y_test.sum()),
            "auc"   : auc,
            "tss"   : tss,
            "boyce" : boyce,
        })

    return pd.DataFrame(resultados), all_preds, all_true, predicciones


def plot_roc_por_fold(predicciones: list, titulo: str,
                       nombre_archivo: str, output_dir: Path):


    fig, ax = plt.subplots(figsize=(7, 6))

    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []
    colores = plt.cm.viridis(np.linspace(0, 0.85, len(predicciones)))

    for fold_idx, (y_true, y_pred) in enumerate(predicciones):
        if y_true is None:
            print(f"    Fold {fold_idx+1}: sin datos, omitido en la gráfica")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_fold = sklearn_auc(fpr, tpr)
        aucs.append(auc_fold)

        ax.plot(fpr, tpr, color=colores[fold_idx], linewidth=1.3, alpha=0.6,
                label=f"Fold {fold_idx+1} (AUC = {auc_fold:.4f})")

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr  = np.std(tprs, axis=0)
    mean_auc, std_auc = np.mean(aucs), np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color="black", linewidth=2.5,
            label=f"Media (AUC = {mean_auc:.4f} ± {std_auc:.4f})")

    ax.fill_between(mean_fpr,
                     np.maximum(mean_tpr - std_tpr, 0),
                     np.minimum(mean_tpr + std_tpr, 1),
                     color="grey", alpha=0.2, label="± 1 desv. estándar")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Clasificador aleatorio")

    ax.set_title(f"Curvas ROC por fold — {titulo}", fontsize=12)
    ax.set_xlabel("Tasa de Falsos Positivos (1 − Especificidad)", fontsize=11)
    ax.set_ylabel("Tasa de Verdaderos Positivos (Sensitividad)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_dir / nombre_archivo
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curvas ROC por fold guardadas en: {path}")


# 5. MÉTRICAS DE EVALUACIÓN
def calcular_tss(y_true: np.ndarray, y_pred: np.ndarray,
                 umbral: float = None) -> float:

    if umbral is None:
        umbrales = np.linspace(0.01, 0.99, 200)
        mejor_tss = -np.inf
        for u in umbrales:
            y_bin = (y_pred >= u).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_bin,
                                              labels=[0,1]).ravel()
            tss_u = tp/(tp+fn) + tn/(tn+fp) - 1 if (tp+fn)>0 and (tn+fp)>0 else -1
            if tss_u > mejor_tss:
                mejor_tss = tss_u
        return mejor_tss
    else:
        y_bin = (y_pred >= umbral).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0,1]).ravel()
        return tp/(tp+fn) + tn/(tn+fp) - 1


def calcular_boyce(preds_pres: np.ndarray,
                   preds_back: np.ndarray,
                   window: float = 0.1,
                   step: float = 0.01) -> float:

    if len(preds_pres) < 5:
        return np.nan

    centros  = []
    f_ratios = []

    posiciones = np.arange(0, 1 - window + step, step)

    for lo in posiciones:
        hi = lo + window
        n_p = np.sum((preds_pres >= lo) & (preds_pres <= hi))
        n_b = np.sum((preds_back >= lo) & (preds_back <= hi))

        prop_p = n_p / len(preds_pres)
        prop_b = n_b / len(preds_back)

        if prop_b == 0:
            continue         
        
        centros.append(lo + window / 2)
        f_ratios.append(prop_p / prop_b)   # 0 es válido si prop_p==0

    if len(centros) < 5:
        return np.nan

    r, _ = spearmanr(centros, f_ratios)
    return float(r)


def auc_and_variance(y_true, y_score):

    desc = np.argsort(y_score)
    y_true = np.asarray(y_true)[desc]
    y_score = np.asarray(y_score)[desc]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos < 2 or n_neg < 2:
        raise ValueError("Se necesitan casos positivos y negativos.")

    ranks = np.empty(len(y_true))
    i = 0
    while i < len(y_true):
        j = i
        while j < len(y_true) and y_score[j] == y_score[i]:
            j += 1
        mean_rank = (i + j - 1) / 2.0 + 1  # rangos desde 1
        ranks[i:j] = mean_rank
        i = j


    pos_ranks = ranks[y_true == 1]
    AUC = (np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    V01 = (ranks[y_true == 1] - np.arange(1, n_pos + 1) - AUC * n_neg) / n_neg
    V10 = (ranks[y_true == 0] - np.arange(1, n_neg + 1) - (1 - AUC) * n_pos) / n_pos

    S01 = np.sum(V01**2) / (n_pos - 1) 
    S10 = np.sum(V10**2) / (n_neg - 1) 

    var_AUC = S01 / n_pos + S10 / n_neg
    return AUC, var_AUC


def delong_test(y_true, y_score1, y_score2):

    AUC1, var1 = auc_and_variance(y_true, y_score1)
    AUC2, var2 = auc_and_variance(y_true, y_score2)

    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    n_pos = len(pos)
    n_neg = len(neg)

    def get_components(y_true, y_score):
        desc = np.argsort(y_score)
        y_true_ord = y_true[desc]
        y_score_ord = y_score[desc]
        n_pos = np.sum(y_true_ord == 1)
        n_neg = np.sum(y_true_ord == 0)
        ranks = np.empty(len(y_true_ord))
        i = 0
        while i < len(y_true_ord):
            j = i
            while j < len(y_true_ord) and y_score_ord[j] == y_score_ord[i]:
                j += 1
            mean_rank = (i + j - 1) / 2.0 + 1
            ranks[i:j] = mean_rank
            i = j
        AUC = (np.sum(ranks[y_true_ord == 1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
        V01 = (ranks[y_true_ord == 1] - np.arange(1, n_pos+1) - AUC * n_neg) / n_neg
        V10 = (ranks[y_true_ord == 0] - np.arange(1, n_neg+1) - (1-AUC) * n_pos) / n_pos
        return V01, V10, AUC

    V01_1, V10_1, AUC1 = get_components(y_true, y_score1)
    V01_2, V10_2, AUC2 = get_components(y_true, y_score2)

    cov = (np.sum(V01_1 * V01_2) / (n_pos - 1) if n_pos > 1 else 0) / n_pos + \
          (np.sum(V10_1 * V10_2) / (n_neg - 1) if n_neg > 1 else 0) / n_neg

    se_diff = np.sqrt(var1 + var2 - 2 * cov)
    if se_diff == 0:
        return AUC1, AUC2, 0.0, 0.0, 1.0

    z = (AUC1 - AUC2) / se_diff
    p = 2 * norm.cdf(-abs(z))
    return AUC1, AUC2, AUC1 - AUC2, z, p

def random_scores(y_true, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(size=len(y_true))

# 6. INTERPRETACIÓN DEL MODELO

def curvas_de_respuesta(model: MaxentModel, X: pd.DataFrame,
                         var_cols: list, output_dir: Path,
                         y: np.array = None):

    n_vars = len(var_cols)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5*n_cols, 3.5*n_rows))
    axes = np.array(axes).flatten() if n_vars > 1 else [axes]
    
    if y is not None and (y == 1).any():
        X_media = X[var_cols][y == 1].mean()
    else:
        X_media = X[var_cols].mean()  # fallback si no se pasa y 

    for i, var in enumerate(var_cols):
        rango = np.linspace(
            np.percentile(X[var], 2),
            np.percentile(X[var], 98),
            200
        )

        X_pred = pd.DataFrame(
            np.tile(X_media.values, (200, 1)),
            columns=var_cols
        )
        X_pred[var] = rango

        idoneidad = model.predict(X_pred)

        ax = axes[i]
        ax.plot(rango, idoneidad, color="#2c7bb6", linewidth=2)
        ax.fill_between(rango, idoneidad, alpha=0.15, color="#2c7bb6")
        ax.set_xlabel(var, fontsize=10)
        ax.set_ylabel("Idoneidad (cloglog)", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"Respuesta marginal: {var}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Curvas de Respuesta Marginal — MaxEnt\n"
                 "(cada variable varía en su rango; resto fijado en la media)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    path = output_dir / "curvas_respuesta.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curvas de respuesta guardadas en: {path}")


def importancia_variables_jackknife(model_full: MaxentModel,
                                     X: pd.DataFrame,
                                     y: np.ndarray,
                                     var_cols: list,
                                     feature_types: list,
                                     beta_multiplier: float,
                                     output_dir: Path,
                                     random_state: int = 42):

    print("\n  Calculando importancia de variables (jackknife)...")

    preds_full  = model_full.predict(X)
    auc_full    = roc_auc_score(y, preds_full)

    solo   = {}  
    sin    = {}

    for var in var_cols:

        try:
            m_solo = entrenar_maxent(X[[var]], y, feature_types,
                                     beta_multiplier, random_state)
            solo[var] = roc_auc_score(y, m_solo.predict(X[[var]]))
        except Exception:
            solo[var] = np.nan
        resto = [v for v in var_cols if v != var]
        try:
            m_sin = entrenar_maxent(X[resto], y, feature_types,
                                    beta_multiplier, random_state)
            sin[var] = roc_auc_score(y, m_sin.predict(X[resto]))
        except Exception:
            sin[var] = np.nan

        print(f"    {var:<20} solo={solo[var]:.4f}   sin={sin[var]:.4f}")


    importancia_df = pd.DataFrame({
        "variable"        : var_cols,
        "auc_solo"        : [solo[v] for v in var_cols],
        "auc_sin"         : [sin[v] for v in var_cols],
        "contribucion"    : [auc_full - sin[v] for v in var_cols],
    }).sort_values("contribucion", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(var_cols)*0.5)))

    axes[0].barh(importancia_df["variable"], importancia_df["auc_solo"],
                 color="#3498db", edgecolor="white")
    axes[0].axvline(0.5, color="red", linestyle="--", linewidth=1,
                    label="AUC = 0.5 (azar)")
    axes[0].axvline(auc_full, color="green", linestyle="--", linewidth=1.5,
                    label=f"AUC modelo completo = {auc_full:.4f}")
    axes[0].set_xlabel("AUC", fontsize=11)
    axes[0].set_title("AUC con solo esta variable", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(0.4, 1.0)


    colores = ["#e74c3c" if c > 0 else "#95a5a6"
               for c in importancia_df["contribucion"]]
    axes[1].barh(importancia_df["variable"], importancia_df["contribucion"],
                 color=colores, edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Contribución marginal (ΔAUC)", fontsize=11)
    axes[1].set_title("Contribución única\n(AUC_completo − AUC_sin_variable)",
                      fontsize=11)

    plt.suptitle("Importancia de Variables — Análisis Jackknife",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = output_dir / "importancia_jackknife.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Jackknife guardado en: {path}")

    importancia_df.to_csv(output_dir / "importancia_variables.csv", index=False)
    return importancia_df

# 7. VISUALIZACIONES


def plot_roc_con_cv(all_preds: np.ndarray, all_true: np.ndarray,
                    resultados_cv: pd.DataFrame, output_dir: Path):

    fig, ax = plt.subplots(figsize=(7, 6))

    fpr, tpr, _ = roc_curve(all_true, all_preds)
    auc_total   = sklearn_auc(fpr, tpr)
    ax.plot(fpr, tpr, color="#2c7bb6", linewidth=2,
            label=f"ROC out-of-fold (AUC = {auc_total:.4f})")

    ax.plot([0,1], [0,1], "k--", linewidth=1, label="Clasificador aleatorio")

    auc_media = resultados_cv["auc"].mean()
    auc_std   = resultados_cv["auc"].std()
    ax.set_title(f"Curva ROC — Validación Cruzada Aleatoria\n"
                 f"AUC = {auc_media:.4f} ± {auc_std:.4f}", fontsize=12)
    ax.set_xlabel("Tasa de Falsos Positivos (1 − Especificidad)", fontsize=11)
    ax.set_ylabel("Tasa de Verdaderos Positivos (Sensitividad)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "curva_roc.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curva ROC guardada en: {path}")


def plot_tabla_metricas(df_geo: pd.DataFrame, df_rand: pd.DataFrame,
                         output_dir: Path):

    resumen = pd.DataFrame({
        "Métrica"        : ["AUC", "TSS", "Boyce"],
        "CV Geográfica"  : [
            f"{df_geo['auc'].mean():.4f} ± {df_geo['auc'].std():.4f}",
            f"{df_geo['tss'].mean():.4f} ± {df_geo['tss'].std():.4f}",
            f"{df_geo['boyce'].mean():.4f} ± {df_geo['boyce'].std():.4f}",
        ],
        "CV Aleatoria"   : [
            f"{df_rand['auc'].mean():.4f} ± {df_rand['auc'].std():.4f}",
            f"{df_rand['tss'].mean():.4f} ± {df_rand['tss'].std():.4f}",
            f"{df_rand['boyce'].mean():.4f} ± {df_rand['boyce'].std():.4f}",
        ],
    })

    print("\n  ┌─ RESUMEN DE MÉTRICAS DE VALIDACIÓN ─────────────────────┐")
    print(resumen.to_string(index=False))
    print("  └──────────────────────────────────────────────────────────┘")

    resumen.to_csv(output_dir / "metricas_validacion.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")
    tabla = ax.table(
        cellText    = resumen.values,
        colLabels   = resumen.columns,
        cellLoc     = "center",
        loc         = "center",
        colColours  = ["#2c3e50"]*3,
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(11)
    tabla.scale(1.2, 2)
    for (r, c), cell in tabla.get_celld().items():
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#ecf0f1" if r % 2 == 0 else "white")
    plt.title("Métricas de Validación — MaxEnt", pad=20, fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "tabla_metricas.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_distribucion_idoneidad(model: MaxentModel,
                                 X: pd.DataFrame,
                                 y: np.ndarray,
                                 output_dir: Path):

    preds      = model.predict(X)
    preds_pres = preds[y == 1]
    preds_back = preds[y == 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(preds_back, bins=50, alpha=0.6, color="#e74c3c",
            density=True, label=f"Background (n={len(preds_back)})")
    ax.hist(preds_pres, bins=30, alpha=0.7, color="#2ecc71",
            density=True, label=f"Presencias (n={len(preds_pres)})")
    ax.set_xlabel("Idoneidad predicha (cloglog)", fontsize=12)
    ax.set_ylabel("Densidad", fontsize=12)
    ax.set_title("Distribución de Idoneidad — Presencias vs Background\n"
                 "(mayor separación = mejor discriminación)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "distribucion_idoneidad.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Distribución de idoneidad guardada en: {path}")



# 8. GENERACIÓN DE DATOS DEMO


def generar_datos_demo(n_pres: int = 100, n_back: int = 2000,
                        n_vars: int = 5, seed: int = 42):

    rng      = np.random.default_rng(seed)
    var_cols = [f"var_{i+1}" for i in range(n_vars)]

    back = {v: rng.uniform(0, 1, n_back) for v in var_cols}
    back[COL_LAT] = rng.uniform(19.8, 23.2, n_back)
    back[COL_LON] = rng.uniform(-85.0, -74.0, n_back)


    pres = {}
    pres["var_1"] = np.clip(rng.normal(0.75, 0.10, n_pres), 0, 1)
    pres["var_2"] = np.clip(rng.normal(0.70, 0.12, n_pres), 0, 1)
    pres["var_3"] = np.clip(rng.normal(0.25, 0.12, n_pres), 0, 1)
    for v in var_cols[3:]:
        pres[v] = rng.uniform(0, 1, n_pres)
    pres[COL_LAT] = rng.uniform(19.8, 23.2, n_pres)
    pres[COL_LON] = rng.uniform(-85.0, -74.0, n_pres)

    pd.DataFrame(pres).to_csv(PRESENCIAS_CSV, index=False)
    pd.DataFrame(back).to_csv(BACKGROUND_CSV, index=False)
    print(f"  Demo: '{PRESENCIAS_CSV}' y '{BACKGROUND_CSV}' generados.")
    return var_cols



# 9. PIPELINE PRINCIPAL


def main(modo_demo: bool = False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "═"*60)
    print("  MaxEnt — Aplicación completa (elapid)")
    print("  Distribución potencial de manglares en Cuba")
    print("═"*60)


    var_cols_demo = None
    if modo_demo:
        print("\n[DEMO] Generando datos sintéticos...")
        var_cols_demo = generar_datos_demo()


    print("\n[1] Cargando presencias...")
    pres_gdf = csv_a_geodataframe(PRESENCIAS_CSV, COL_LAT, COL_LON, CRS)
    print(f"    {len(pres_gdf)} puntos de presencia cargados")


    print("\n[2] Cargando background...")
    if Path(BACKGROUND_CSV).exists():
        back_gdf = csv_a_geodataframe(BACKGROUND_CSV, COL_LAT, COL_LON, CRS)
        print(f"    {len(back_gdf)} puntos de background desde CSV")
    else:

        print(f"    No encontrado '{BACKGROUND_CSV}'. Muestreando desde rasters...")
        raster_paths = sorted(Path(RASTER_DIR).glob("*.tif"))
        if not raster_paths:
            raise FileNotFoundError(
                f"No se encontraron rasters en '{RASTER_DIR}'. "
                "Proporciona background.csv o archivos .tif."
            )
        back_gdf = muestrear_background(
            [str(p) for p in raster_paths], N_BACKGROUND, RANDOM_STATE
        )
        print(f"    {len(back_gdf)} puntos de background muestreados")


    print("\n[3] Preparando variables ambientales...")

    raster_paths = sorted(Path(RASTER_DIR).glob("*.tif")) if Path(RASTER_DIR).exists() else []

    if MODO_ENTRADA in ("raster", "ambos") and raster_paths:
        # Extraer valores de rasters en cada punto
        print(f"    Extrayendo valores de {len(raster_paths)} rasters...")
        nombres_vars = [Path(p).stem for p in raster_paths]

        pres_gdf = extraer_variables_de_rasters(
            pres_gdf, [str(p) for p in raster_paths], nombres_vars
        )
        back_gdf = extraer_variables_de_rasters(
            back_gdf, [str(p) for p in raster_paths], nombres_vars
        )
        var_cols = nombres_vars

    else:
        # Variables ya están en el CSV
        coord_cols = [COL_LAT, COL_LON]
        var_cols   = var_cols_demo or detectar_variable_cols(
            pres_gdf, coord_cols, VARIABLE_COLS
        )
        print(f"    Variables desde CSV: {var_cols}")

    print(f"    Variables finales ({len(var_cols)}): {var_cols}")


    # 3.5 Guardar background con variables (CSV)

    print("\n[3.5] Guardando puntos de background con sus variables...")
    back_to_save = back_gdf.copy()

    back_to_save['lat'] = back_to_save.geometry.y
    back_to_save['lon'] = back_to_save.geometry.x

    back_to_save = back_to_save.drop(columns=['geometry'])

    back_csv_path = OUTPUT_DIR / "background_con_variables.csv"
    back_to_save.to_csv(back_csv_path, index=False)
    print(f"    Background guardado en: {back_csv_path}")



    print("\n[4] Construyendo matrices de entrenamiento...")


    pres_clean = pres_gdf.dropna(subset=var_cols)
    back_clean = back_gdf.dropna(subset=var_cols)
    if len(pres_clean) < len(pres_gdf):
        print(f"    Eliminados {len(pres_gdf)-len(pres_clean)} puntos de presencia con NaN")
    if len(back_clean) < len(back_gdf):
        print(f"    Eliminados {len(back_gdf)-len(back_clean)} puntos de background con NaN")

    pres_gdf = pres_clean.reset_index(drop=True)
    back_gdf = back_clean.reset_index(drop=True)


    X = pd.concat([pres_gdf[var_cols], back_gdf[var_cols]], ignore_index=True)
    y = np.array([1]*len(pres_gdf) + [0]*len(back_gdf))
    print(f"    X shape: {X.shape}  |  presencias: {y.sum()}  background: {(y==0).sum()}")

    print("\n[5] Entrenando modelo MaxEnt final...")
    print(f"    Feature types   : {FEATURE_TYPES}")
    print(f"    Beta multiplier : {BETA_MULTIPLIER}")

    model_final = entrenar_maxent(X, y, FEATURE_TYPES, BETA_MULTIPLIER, RANDOM_STATE)
    print("    Modelo entrenado.")


    coef = model_final.estimator.coef_.flatten()
    n_nonzero = np.sum(np.abs(coef) > 1e-10)
    print(f"    Coeficientes no nulos (λ_j ≠ 0): {n_nonzero} de {len(coef)}")


    print("\n[6] Validación cruzada...")


    df_geo, preds_geo = validacion_cruzada_geografica(
        pres_gdf, back_gdf, var_cols,
        FEATURE_TYPES, BETA_MULTIPLIER,
        N_FOLDS_GEO, RANDOM_STATE
    )

    df_rand, all_preds, all_true, preds_rand = validacion_cruzada_aleatoria(
        X, y, FEATURE_TYPES, BETA_MULTIPLIER,
        N_FOLDS_RANDOM, RANDOM_STATE
    )

    # TEST DE DELONG – comparación contra modelo aleatorio

    print("\n" + "=" * 65)
    print("  TEST DE DELONG – Comparación AUC del modelo vs. Azar (AUC=0.5)")
    print("=" * 65)


    print("\n  ► Validación Geográfica")
    for fold_idx, (y_true, preds_model) in enumerate(preds_geo):
        if y_true is None:
            print(f"    Fold {fold_idx+1}: Error – sin datos")
            continue
        preds_random = random_scores(y_true, seed=fold_idx * 100)
        auc_model, auc_rand, diff, z, p = delong_test(y_true, preds_model, preds_random)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    Fold {fold_idx+1}: AUC_model={auc_model:.4f}  AUC_rand={auc_rand:.4f}  "
            f"Δ={diff:.4f}  z={z:+.2f}  p={p:.4f} {sig}")


    print("\n  ► Validación Aleatoria")
    for fold_idx, (y_true, preds_model) in enumerate(preds_rand):
        if y_true is None:
            print(f"    Fold {fold_idx+1}: Error – sin datos")
            continue
        preds_random = random_scores(y_true, seed=fold_idx * 200)  # semilla distinta para no coincidir
        auc_model, auc_rand, diff, z, p = delong_test(y_true, preds_model, preds_random)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    Fold {fold_idx+1}: AUC_model={auc_model:.4f}  AUC_rand={auc_rand:.4f}  "
            f"Δ={diff:.4f}  z={z:+.2f}  p={p:.4f} {sig}")



    plot_tabla_metricas(df_geo, df_rand, OUTPUT_DIR)
    df_geo.to_csv(OUTPUT_DIR  / "cv_geografica.csv",  index=False)
    df_rand.to_csv(OUTPUT_DIR / "cv_aleatoria.csv",   index=False)


    print("\n[7] Interpretación del modelo...")

    curvas_de_respuesta(model_final, X, var_cols, OUTPUT_DIR, y)

    importancia_df = importancia_variables_jackknife(
        model_final, X, y, var_cols,
        FEATURE_TYPES, BETA_MULTIPLIER, OUTPUT_DIR, RANDOM_STATE
    )


    print("\n[8] Generando visualizaciones...")

    plot_roc_con_cv(all_preds, all_true, df_rand, OUTPUT_DIR)
    plot_roc_por_fold(preds_geo,  "Validación Geográfica",
                   "curva_roc_por_fold_geografica.png", OUTPUT_DIR)
    plot_roc_por_fold(preds_rand, "Validación Aleatoria",
                   "curva_roc_por_fold_aleatoria.png", OUTPUT_DIR)
    plot_distribucion_idoneidad(model_final, X, y, OUTPUT_DIR)

    if raster_paths:
        print("\n[9] Generando mapa de idoneidad sobre los rasters...")
        mapa_output = str(OUTPUT_DIR / "idoneidad_manglares_cuba.tif")
        apply_model_to_rasters(
            model        = model_final,
            raster_paths = [str(p) for p in raster_paths],
            output_path  = mapa_output,
            quiet        = False,
        )
        print(f"    Mapa guardado en: {mapa_output}")
    else:
        print("\n[9] Sin rasters disponibles — saltando predicción espacial.")
        print("    Para generar el mapa, coloca archivos .tif en la carpeta "
              f"'{RASTER_DIR}'.")


    print("\n[10] Guardando modelo final...")
    model_path = OUTPUT_DIR / "maxent_final.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_final, f)
    print(f"     Modelo guardado en: {model_path}")


    print("\n" + "═"*60)
    print("  RESUMEN FINAL")
    print("═"*60)
    print(f"  Presencias     : {len(pres_gdf)}")
    print(f"  Background     : {len(back_gdf)}")
    print(f"  Variables      : {len(var_cols)}")
    print(f"  Features       : {FEATURE_TYPES}")
    print(f"  Beta           : {BETA_MULTIPLIER}")
    print(f"  λ no nulos     : {n_nonzero}")
    print(f"\n  AUC (CV geo)   : {df_geo['auc'].mean():.4f} ± {df_geo['auc'].std():.4f}")
    print(f"  TSS (CV geo)   : {df_geo['tss'].mean():.4f} ± {df_geo['tss'].std():.4f}")
    print(f"  Boyce (CV geo) : {df_geo['boyce'].mean():.4f} ± {df_geo['boyce'].std():.4f}")
    print(f"\n  AUC (CV rand)  : {df_rand['auc'].mean():.4f} ± {df_rand['auc'].std():.4f}")
    print(f"  TSS (CV rand)  : {df_rand['tss'].mean():.4f} ± {df_rand['tss'].std():.4f}")
    print(f"  Boyce (CV rand): {df_rand['boyce'].mean():.4f} ± {df_rand['boyce'].std():.4f}")
    print(f"\n  Archivos en    : {OUTPUT_DIR}/")
    print("═"*60 + "\n")



# ENTRADA

if __name__ == "__main__":
    import sys
    main(modo_demo="--demo" in sys.argv)
