"""
================================================================================
OPTIMIZACIÓN DE HIPERPARÁMETROS DE MAXENT
Framework matemático-estadístico para SDMs
--------------------------------------------------------------------------------
Autor     : [Tu nombre]
Tesis     : Distribución potencial de manglares en Cuba — MaxEnt
Lenguaje  : Python 3.10+
Librerías : elapid, numpy, pandas, scikit-learn, matplotlib, seaborn
--------------------------------------------------------------------------------
Descripción:
    Este script realiza una búsqueda exhaustiva (grid search) sobre el espacio
    de hiperparámetros del modelo MaxEnt:
        - Feature classes (FC): combinaciones de L, Q, P, H, T
        - Regularization multiplier (RM): grilla en [0.5, 4.0]

    Criterios de selección:
        1. AICc      → parsimonia formal (Akaike corregido), calculado sobre
                       el modelo completo (todos los datos).
        2. AUC diff  → sobreajuste (AUC_train - AUC_test, por CV k-fold).
                       Valores cercanos a 0 indican buena generalización;
                       valores grandes y positivos indican sobreajuste.
        3. AUC test  → poder discriminativo medio en el conjunto de prueba
                       de cada fold.

    La selección del modelo óptimo balancea tener un AICc bajo (parsimonia)
    y un AUC diff bajo (buena generalización, ausencia de sobreajuste).

    Salidas:
        - Tabla completa de resultados (CSV)
        - Heatmaps AICc, AUC y AUC diff (PNG)
        - Gráfico de Pareto AICc vs AUC diff (PNG)
        - Modelo óptimo serializado (pkl)
        - Reporte resumen en consola
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from elapid import MaxentModel

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

PRESENCE_CSV   = "/Users/eryuer/Desktop/tesis/Final/Modelo/Maxent_input/presencias_mrmr.csv"
BACKGROUND_CSV = "/Users/eryuer/Desktop/tesis/Final/Modelo/Maxent_input/background_mrmr.csv"
COORD_COLS     = ["lat", "lon"]
OUTPUT_DIR     = Path("/Users/eryuer/Desktop/tesis/Final/Modelo/resultados_optimizacionv6")

RM_VALUES          = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
FC_CANDIDATES      = ["linear", "quadratic", "product", "hinge", "threshold"]
FC_MIN_SIZE        = 1
FC_MAX_SIZE        = 5
N_FOLDS            = 5
RANDOM_STATE       = 42


# ─────────────────────────────────────────────────────────────────────────────
# 2. FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def load_data(presence_csv: str, background_csv: str, coord_cols: list):
    pres = pd.read_csv(presence_csv)
    back = pd.read_csv(background_csv)
    exclude      = set(coord_cols + ["species", "id", "PXLVAL"])
    feature_cols = [c for c in pres.columns if c not in exclude]
    missing      = [c for c in feature_cols if c not in back.columns]
    if missing:
        raise KeyError(f"Background sin columnas requeridas: {missing}")
    X = pd.concat([pres[feature_cols], back[feature_cols]], ignore_index=True)
    y = np.array([1] * len(pres) + [0] * len(back))
    # ── NUEVO: coordenadas concatenadas en el mismo orden que X, y ──
    coords_pres = pres[coord_cols].values
    coords_back = back[coord_cols].values
    coords = np.vstack([coords_pres, coords_back])   # shape (N, 2)

    print(f"  Presencias : {len(pres)}")
    print(f"  Background : {len(back)}")
    print(f"  Variables  : {feature_cols}")
    return X, y, coords

from sklearn.cluster import KMeans

def make_spatial_folds(coords: np.ndarray, y: np.ndarray,
                       n_folds: int, random_state: int) -> np.ndarray:
    """
    Asigna cada punto a un fold geográfico usando k-means sobre las
    coordenadas. Puntos del mismo bloque espacial van siempre al mismo fold,
    evitando la autocorrelación espacial entre train y test.

    Retorna un array fold_ids de longitud N con valores en {0, ..., n_folds-1}.
    """
    km = KMeans(n_clusters=n_folds, random_state=random_state, n_init=10)
    fold_ids = km.fit_predict(coords)          # cluster por geografía

    # Verificar que cada fold tenga presencias y background
    for f in range(n_folds):
        mask = fold_ids == f
        if y[mask].sum() == 0 or (1 - y[mask]).sum() == 0:
            print(f"  ⚠️  Fold geográfico {f} sin presencias o sin background. "
                  f"Considera reducir n_folds.")
    return fold_ids

def compute_aicc(log_likelihood: float, n_params: int, n_presence: int) -> float:
    """
    AICc = 2k - 2·LL + 2k(k+1)/(n-k-1)

    Referencia: Warren & Seifert (2011). Ecological Applications, 21(2), 335–342.
    """
    k = n_params
    n = n_presence
    if k == 0 or n - k - 1 <= 0:
        return np.inf
    aic  = 2 * k - 2 * log_likelihood
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    return aicc


def compute_log_likelihood(raw_model: MaxentModel,
                           X: pd.DataFrame, y: np.ndarray) -> float:
    """
    Log-verosimilitud de MaxEnt sobre los datos (X, y) usando un modelo
    ya entrenado con transform='raw'.

    LL = Σ_{pres} log(f_raw(x)) - n_pres · log( mean_{bg}(f_raw(x)) )

    Separa presencias y background según la etiqueta y, de modo que puede
    usarse tanto en datos de entrenamiento como en datos de test del mismo
    fold sin reentrenar.

    Referencia: Phillips & Dudík (2008), Warren & Seifert (2011).
    """
    raw_scores = np.asarray(raw_model.predict(X), dtype=float)
    y_arr      = np.asarray(y)

    pres_scores = raw_scores[y_arr == 1]
    back_scores = raw_scores[y_arr == 0]

    if len(pres_scores) == 0 or len(back_scores) == 0:
        return -np.inf

    eps = 1e-9
    pres_scores = np.clip(pres_scores, eps, None)
    back_scores = np.clip(back_scores, eps, None)

    return float(np.sum(np.log(pres_scores))
                 - len(pres_scores) * np.log(np.mean(back_scores)))


def get_n_nonzero_params(model: MaxentModel) -> int:
    """Número de coeficientes no nulos (k para AICc)."""
    for attr in ("coef_", "estimator.coef_", "estimator_.coef_"):
        try:
            obj = model
            for part in attr.split("."):
                obj = getattr(obj, part)
            return int(np.sum(np.abs(obj) > 1e-10))
        except AttributeError:
            continue
    return 1


def fc_label(feature_types: list) -> str:
    abbrev = {"linear": "L", "quadratic": "Q", "product": "P",
              "hinge": "H", "threshold": "T"}
    return "".join(abbrev[f] for f in feature_types if f in abbrev)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MÉTRICAS DE VALIDACIÓN CRUZADA  (AUC test + AUC diff)
# ─────────────────────────────────────────────────────────────────────────────

def cv_metrics(X: pd.DataFrame, y: np.ndarray,
               feature_types: list, beta_multiplier: float,
               fold_ids: np.ndarray,               # ← NUEVO parámetro
               random_state: int) -> dict:
    """
    Validación cruzada con folds geográficos precomputados.
    fold_ids[i] indica a qué fold pertenece el punto i.
    """
    n_folds   = len(np.unique(fold_ids))
    aucs_test = []
    auc_diffs = []

    for f in range(n_folds):
        test_idx  = np.where(fold_ids == f)[0]
        train_idx = np.where(fold_ids != f)[0]

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if (y_train.sum() == 0 or (1 - y_train).sum() == 0 or
                y_test.sum() == 0 or (1 - y_test).sum() == 0):
            continue

        model_cll = MaxentModel(
            feature_types   = feature_types,
            beta_multiplier = beta_multiplier,
            transform       = "raw",
            use_sklearn     = True,
            random_state    = random_state,
        )
        try:
            model_cll.fit(X_train, y_train)
            preds_train = model_cll.predict(X_train)
            preds_test  = model_cll.predict(X_test)
            auc_train   = roc_auc_score(y_train, preds_train)
            auc_test    = roc_auc_score(y_test,  preds_test)
            aucs_test.append(auc_test)
            auc_diffs.append(auc_train - auc_test)
        except Exception:
            continue

    def safe_stats(arr):
        arr = [a for a in arr if not np.isnan(a)]
        if len(arr) == 0:
            return np.nan, np.nan
        return float(np.mean(arr)), float(np.std(arr))

    auc_mean,      auc_std      = safe_stats(aucs_test)
    auc_diff_mean, auc_diff_std = safe_stats(auc_diffs)
    return {"auc_mean": auc_mean, "auc_std": auc_std,
            "auc_diff_mean": auc_diff_mean, "auc_diff_std": auc_diff_std}





# ─────────────────────────────────────────────────────────────────────────────
# 4. GRID SEARCH PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_search(X, y, coords, rm_values, fc_candidates, fc_min, fc_max,
                    n_folds, random_state, output_dir) -> pd.DataFrame:

    output_dir.mkdir(parents=True, exist_ok=True)
    n_presence = int(y.sum())
    
    #------ Folds geograficos--------
    print("\n Calculando folds geograficos")
    folds_ids = make_spatial_folds(coords, y, n_folds, random_state)

    fc_combos = []
    for size in range(fc_min, fc_max + 1):
        for combo in itertools.combinations(fc_candidates, size):
            fc_combos.append(list(combo))

    total = len(fc_combos) * len(rm_values)
    print(f"\n{'─'*70}")
    print(f"  Grid search: {len(fc_combos)} FC × {len(rm_values)} RM = {total} modelos")
    print(f"{'─'*70}")

    records = []
    for counter, (fc, rm) in enumerate(
            itertools.product(fc_combos, rm_values), start=1):
        label = fc_label(fc)
        print(f"  [{counter:>4}/{total}]  FC={label:<8}  RM={rm:.1f}  ", end="", flush=True)

        # ── AICc sobre todos los datos (modelo completo) ──────────────────
        model_raw_full = MaxentModel(
            feature_types   = fc,
            beta_multiplier = rm,
            transform       = "raw",
            use_sklearn     = True,
            random_state    = random_state,
        )
        model_cll_full = MaxentModel(
            feature_types   = fc,
            beta_multiplier = rm,
            transform       = "raw",
            use_sklearn     = True,
            random_state    = random_state,
        )
        try:
            model_raw_full.fit(X, y)
            model_cll_full.fit(X, y)
            ll       = compute_log_likelihood(model_raw_full, X, y)
            n_params = get_n_nonzero_params(model_raw_full)
            aicc_val = compute_aicc(ll, n_params, n_presence)
        except Exception as e:
            print(f"ERROR train: {e}")
            aicc_val, n_params, ll = np.inf, 0, -np.inf
            model_cll_full = None

        # ── AUC test y AUC diff por CV ────────────────────────────────────
        cv = cv_metrics(X, y, fc, rm, folds_ids, random_state)

        print(f"AICc={aicc_val:>10.2f}  "
              f"AUC_diff={cv['auc_diff_mean']:>7.4f}  "
              f"AUC_test={cv['auc_mean']:.4f} ± {cv['auc_std']:.4f}")

        records.append({
            "fc_label"       : label,
            "fc_list"        : str(fc),
            "rm"             : rm,
            "n_params"       : n_params,
            "log_lik"        : ll,
            "aicc"           : aicc_val,
            "delta_aicc"     : np.nan,
            "auc_diff_mean"  : cv["auc_diff_mean"],
            "auc_diff_std"   : cv["auc_diff_std"],
            "auc_mean"       : cv["auc_mean"],
            "auc_std"        : cv["auc_std"],
            "model"          : model_cll_full,
        })

    results_df = pd.DataFrame(records)
    min_aicc = results_df["aicc"].min()
    results_df["delta_aicc"] = results_df["aicc"] - min_aicc

    cols_csv = ["fc_label", "fc_list", "rm", "n_params", "log_lik",
                "aicc", "delta_aicc",
                "auc_diff_mean", "auc_diff_std",
                "auc_mean", "auc_std"]
    results_df[cols_csv].to_csv(output_dir / "grid_search_results.csv", index=False)
    print(f"\n  Resultados guardados en: {output_dir / 'grid_search_results.csv'}")
    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. SELECCIÓN DEL MODELO ÓPTIMO
# ─────────────────────────────────────────────────────────────────────────────

def select_best_models(results_df: pd.DataFrame) -> dict:
    valid = results_df.dropna(subset=["aicc", "auc_mean", "auc_diff_mean"])
    valid = valid[valid["aicc"] < np.inf].copy()

    idx_aicc = valid["aicc"].idxmin()
    idx_auc  = valid["auc_mean"].idxmax()

    # ── Modelo balanceado: AICc + AUC diff ────────────────────────────────
    # Normaliza ambos criterios al rango [0, 1] y selecciona el mínimo
    # de la suma ponderada (igual peso por defecto).
    aicc_norm = (valid["aicc"] - valid["aicc"].min()) / \
                (valid["aicc"].max() - valid["aicc"].min() + 1e-12)
    diff_norm = (valid["auc_diff_mean"] - valid["auc_diff_mean"].min()) / \
                (valid["auc_diff_mean"].max() - valid["auc_diff_mean"].min() + 1e-12)
    valid = valid.copy()
    valid["balance_score"] = 0.5 * aicc_norm + 0.5 * diff_norm
    idx_balanced = valid["balance_score"].idxmin()

    # ── Frontera de Pareto: minimizar AICc y auc_diff_mean ────────────────
    aicc_arr = valid["aicc"].values
    auc_arr  = valid["auc_mean"].values
    diff_arr = valid["auc_diff_mean"].fillna(np.inf).values

    pareto_mask = np.ones(len(valid), dtype=bool)
    for i in range(len(valid)):
        for j in range(len(valid)):
            if i == j:
                continue
            better_aicc = aicc_arr[j] <= aicc_arr[i]
            better_auc  = auc_arr[j]  >= auc_arr[i]
            better_diff = diff_arr[j] <= diff_arr[i]
            strictly    = (aicc_arr[j] < aicc_arr[i] or
                           auc_arr[j]  > auc_arr[i]  or
                           diff_arr[j] < diff_arr[i])
            if better_aicc and better_auc and better_diff and strictly:
                pareto_mask[i] = False
                break

    pareto_front = valid[pareto_mask].sort_values("aicc")

    return {
        "best_aicc"     : valid.loc[idx_aicc],
        "best_auc"      : valid.loc[idx_auc],
        "best_balanced" : valid.loc[idx_balanced],
        "pareto_front"  : pareto_front,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALIZACIONES
# ─────────────────────────────────────────────────────────────────────────────

def _heatmap(pivot, title, cbar_label, output_path,
             cmap="RdYlGn_r", vmin=None, vmax=None, fmt=".1f",
             mark_min=True):
    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap,
                linewidths=0.5, linecolor="white",
                vmin=vmin, vmax=vmax,
                cbar_kws={"label": cbar_label}, ax=ax)
    extreme = pivot.min().min() if mark_min else pivot.max().max()
    for i, row in enumerate(pivot.index):
        for j, col in enumerate(pivot.columns):
            if pivot.loc[row, col] == extreme:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, fill=False, edgecolor="blue", lw=3))
    ax.set_title(title, fontsize=13, pad=15)
    ax.set_xlabel("Regularization Multiplier (RM)", fontsize=11)
    ax.set_ylabel("Feature Classes (FC)", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_path}")


def plot_heatmap_aicc(results_df, output_dir):
    valid = results_df[results_df["aicc"] < np.inf].copy()
    pivot = valid.pivot_table(index="fc_label", columns="rm",
                              values="aicc", aggfunc="min")
    _heatmap(pivot,
             title="Paisaje de AICc — espacio de hiperparámetros MaxEnt\n"
                   "(mínimo resaltado en azul)",
             cbar_label="AICc",
             output_path=output_dir / "heatmap_aicc.png",
             cmap="RdYlGn_r", mark_min=True)


def plot_heatmap_auc(results_df, output_dir):
    valid = results_df.dropna(subset=["auc_mean"]).copy()
    pivot = valid.pivot_table(index="fc_label", columns="rm",
                              values="auc_mean", aggfunc="max")
    _heatmap(pivot,
             title="Paisaje de AUC-ROC (CV) — espacio de hiperparámetros MaxEnt\n"
                   "(máximo resaltado en azul)",
             cbar_label="AUC medio (CV)",
             output_path=output_dir / "heatmap_auc.png",
             cmap="RdYlGn", vmin=0.5, vmax=1.0, fmt=".4f",
             mark_min=False)


def plot_heatmap_auc_diff(results_df, output_dir):
    """
    Heatmap del AUC diff medio (AUC_train - AUC_test) sobre (FC × RM).
    Valores cercanos a 0 indican buena generalización; >> 0 indica sobreajuste.
    """
    valid = results_df.dropna(subset=["auc_diff_mean"]).copy()
    pivot = valid.pivot_table(index="fc_label", columns="rm",
                              values="auc_diff_mean", aggfunc="mean")
    _heatmap(pivot,
             title="AUC diff medio (AUC_train − AUC_test) por CV\n"
                   "≈ 0 → buena generalización  |  >> 0 → sobreajuste\n"
                   "(mínimo resaltado en azul)",
             cbar_label="AUC diff medio",
             output_path=output_dir / "heatmap_auc_diff.png",
             cmap="RdYlGn_r", fmt=".4f", mark_min=True)


def plot_pareto(results_df, best, output_dir):
    valid  = results_df[results_df["aicc"] < np.inf].dropna(
        subset=["auc_diff_mean"]).copy()
    pareto = best["pareto_front"]

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(valid["aicc"], valid["auc_diff_mean"],
                         c=valid["rm"], cmap="plasma",
                         alpha=0.6, s=60, edgecolors="none")
    plt.colorbar(scatter, ax=ax, label="Regularization Multiplier (RM)")

    pareto_s = pareto.sort_values("aicc")
    ax.plot(pareto_s["aicc"], pareto_s["auc_diff_mean"],
            "b-o", lw=2, ms=8, label="Frontera de Pareto", zorder=5)

    b = best["best_aicc"]
    ax.scatter(b["aicc"], b["auc_diff_mean"], marker="*", s=300,
               color="red", zorder=10,
               label=f"Mejor AICc: FC={b['fc_label']}, RM={b['rm']}")
    b = best["best_balanced"]
    ax.scatter(b["aicc"], b["auc_diff_mean"], marker="D", s=180,
               color="green", zorder=10,
               label=f"Mejor balance: FC={b['fc_label']}, RM={b['rm']}")

    ax.set_xlabel("AICc  (↓ mejor)", fontsize=12)
    ax.set_ylabel("AUC diff  AUC_train − AUC_test  (↓ mejor)", fontsize=12)
    ax.set_title("Trade-off AICc vs AUC diff — todos los modelos MaxEnt\n"
                 "Frontera de Pareto en azul", fontsize=13, pad=15)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "pareto_aicc_aucdiff.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")


def plot_delta_aicc(results_df, output_dir, top_n=20):
    valid = results_df[results_df["aicc"] < np.inf].copy()
    top   = valid.nsmallest(top_n, "aicc").copy()
    top["label"] = top["fc_label"] + "\nRM=" + top["rm"].astype(str)

    colors = ["#2ecc71" if d < 2 else "#f39c12" if d <= 10 else "#e74c3c"
              for d in top["delta_aicc"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(top)), top["delta_aicc"], color=colors, edgecolor="white")
    ax.axhline(2,  color="green", linestyle="--", lw=1.5,
               label="ΔAICc = 2 (soporte sustancial)")
    ax.axhline(10, color="red",   linestyle="--", lw=1.5,
               label="ΔAICc = 10 (sin soporte)")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ΔAICc", fontsize=12)
    ax.set_title(f"ΔAICc — Top {top_n} modelos  "
                 f"(verde < 2, naranja 2–10, rojo > 10)", fontsize=13, pad=15)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = output_dir / "delta_aicc_barplot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. REPORTE FINAL
# ─────────────────────────────────────────────────────────────────────────────

def print_report(best, results_df):
    sep = "─" * 65
    print(f"\n{'═'*65}")
    print("  REPORTE DE OPTIMIZACIÓN DE HIPERPARÁMETROS — MaxEnt")
    print(f"{'═'*65}")

    for titulo, key in [("MEJOR AICc (parsimonia)",        "best_aicc"),
                         ("MEJOR AUC TEST (discriminación)", "best_auc"),
                         ("MEJOR BALANCE (AICc + AUC diff)", "best_balanced")]:
        b = best[key]
        print(f"\n  ★ MODELO ÓPTIMO POR {titulo}")
        print(f"  {sep}")
        print(f"    Feature classes      : {b['fc_list']}")
        print(f"    RM                   : {b['rm']}")
        print(f"    AICc                 : {b['aicc']:.4f}")
        print(f"    ΔAICc                : {b['delta_aicc']:.4f}")
        print(f"    AUC diff (train-test): {b['auc_diff_mean']:.4f} ± {b['auc_diff_std']:.4f}")
        print(f"    AUC test (CV)        : {b['auc_mean']:.4f} ± {b['auc_std']:.4f}")
        print(f"    Parámetros no nulos  : {int(b['n_params'])}")

    pareto = best["pareto_front"]
    print(f"\n  ★ FRONTERA DE PARETO ({len(pareto)} modelos no dominados)")
    print(f"  {sep}")
    cols = ["fc_label", "rm", "aicc", "delta_aicc",
            "auc_diff_mean", "auc_diff_std", "auc_mean", "auc_std"]
    print(pareto[cols].to_string(index=False))

    supported = results_df[results_df["delta_aicc"] < 2]
    print(f"\n  Modelos con ΔAICc < 2 (soporte sustancial): {len(supported)}")
    print(f"\n{'═'*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "═"*65)
    print("  MaxEnt — Optimización de Hiperparámetros")
    print("  Tesis: Distribución potencial de manglares en Cuba")
    print("═"*65)

    print("\n[1] Cargando datos...")
    X, y, coords = load_data(PRESENCE_CSV, BACKGROUND_CSV, COORD_COLS)

    print("\n[2] Ejecutando grid search...")
    results_df = run_grid_search(
        X, y, coords,
        rm_values     = RM_VALUES,
        fc_candidates = FC_CANDIDATES,
        fc_min        = FC_MIN_SIZE,
        fc_max        = FC_MAX_SIZE,
        n_folds       = N_FOLDS,
        random_state  = RANDOM_STATE,
        output_dir    = OUTPUT_DIR,
    )

    print("\n[3] Seleccionando modelos óptimos...")
    best = select_best_models(results_df)

    print("\n[4] Generando visualizaciones...")
    plot_heatmap_aicc(results_df, OUTPUT_DIR)
    plot_heatmap_auc(results_df, OUTPUT_DIR)
    plot_heatmap_auc_diff(results_df, OUTPUT_DIR)
    plot_pareto(results_df, best, OUTPUT_DIR)
    plot_delta_aicc(results_df, OUTPUT_DIR)

    print("\n[5] Guardando modelo óptimo (mejor balance AICc + AUC diff)...")
    best_model = best["best_balanced"]["model"]
    if best_model is not None:
        model_path = OUTPUT_DIR / "maxent_optimal_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        print(f"  Modelo guardado en: {model_path}")

    print_report(best, results_df)


if __name__ == "__main__":
    main()
