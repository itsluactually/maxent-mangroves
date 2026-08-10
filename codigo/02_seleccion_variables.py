import pandas as pd
import numpy as np
import seaborn as sns
import rasterio
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression



# 1. CONFIGURACIÓN 

CSV_PATH             = ""
RASTER_DIR           = ""
VAR_NAMES            = []
N_BG_POINTS          = 10_000   # puntos de background deseados
N_BG_OVERSAMPLE      = 40_000   # se generan más para compensar mar y NoData
N_FEATURES_TO_SELECT = 8
K_NUMBER = 3

# 2. CARGAR PRESENCIAS

df = pd.read_csv(CSV_PATH)
pres_coords = df[['lat', 'lon']].values
pres_values = df[VAR_NAMES].values
print(f"Presencias cargadas: {len(pres_values)}")


# 3. EXTRAER VALORES DE RASTER CON MANEJO CORRECTO DE NODATA

def extract_raster_values(raster_path, coords):

    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        # rasterio.sample espera (x, y) → (lon, lat)
        values = np.array([val[0] for val in src.sample(coords[:, [1, 0]])])
        values = values.astype(float)
        if nodata is not None:
            values[values == nodata] = np.nan
    return values

# 4. GENERAR PUNTOS DE BACKGROUND VÁLIDOS
#    Se generan N_BG_OVERSAMPLE puntos en el bounding box y se descartan
#    los que caen en el mar o en celdas NoData, quedándose con N_BG_POINTS.

with rasterio.open(f"{RASTER_DIR}/{VAR_NAMES[0]}_cuba.tif") as src:
    bounds = src.bounds

np.random.seed(42)
bg_lon = np.random.uniform(bounds.left,   bounds.right, N_BG_OVERSAMPLE)
bg_lat = np.random.uniform(bounds.bottom, bounds.top,   N_BG_OVERSAMPLE)
bg_coords_all = np.column_stack([bg_lat, bg_lon])

print(f"Extrayendo valores para {N_BG_OVERSAMPLE} puntos candidatos de background...")
bg_values_all = np.zeros((N_BG_OVERSAMPLE, len(VAR_NAMES)))
for i, var in enumerate(VAR_NAMES):
    bg_values_all[:, i] = extract_raster_values(
        f"{RASTER_DIR}/{var}_cuba.tif", bg_coords_all
    )

# Filtrar puntos con cualquier NoData
valid_bg = ~np.isnan(bg_values_all).any(axis=1)
bg_values_filtered = bg_values_all[valid_bg]
bg_coords_filtered = bg_coords_all[valid_bg]

print(f"Puntos de background válidos tras filtrar NoData: {valid_bg.sum()}")

if valid_bg.sum() < N_BG_POINTS:
    print(f"Solo hay {valid_bg.sum()} puntos válidos. "
          f"Considera aumentar N_BG_OVERSAMPLE.")
    bg_values = bg_values_filtered
    bg_coords = bg_coords_filtered
else:
    bg_values = bg_values_filtered[:N_BG_POINTS]
    bg_coords = bg_coords_filtered[:N_BG_POINTS]

print(f"Background final: {len(bg_values)} puntos")


# 5. COMBINAR PRESENCIAS Y BACKGROUND

X_pres = pres_values
X_bg   = bg_values
y_pres = np.ones(len(X_pres))
y_bg   = np.zeros(len(X_bg))

X = np.vstack([X_pres, X_bg])
y = np.hstack([y_pres, y_bg])

# Filtrar filas con NaN en las presencias (el background ya está limpio)
valid = ~np.isnan(X).any(axis=1)
X = X[valid]
y = y[valid]

print(f"\nTotal de puntos utilizables: {X.shape[0]} "
      f"(presencias: {int(y.sum())}, background: {int((1-y).sum())})")


# 7. CALCULAR RELEVANCIA Y REDUNDANCIA

# Relevancia: MI entre cada variable y la presencia/background (target binario)
print("\nCalculando relevancia...")
relevance = mutual_info_classif(X, y, n_neighbors=K_NUMBER, discrete_features=False, random_state=42)

# Redundancia: MI entre pares de variables (ambas continuas)
print("Calculando matriz de redundancia...")
n_features = len(VAR_NAMES)
redundancy = np.zeros((n_features, n_features))

for i in range(n_features):
    # mutual_info_regression estima MI(X[:,j], X[:,i]) para todo j a la vez
    mi_row = mutual_info_regression(X, X[:, i], n_neighbors=K_NUMBER, discrete_features=False, random_state=42)
    redundancy[i, :] = mi_row
    if i % 5 == 0:
        print(f"  variable {i+1}/{n_features}...")

# Simetrizar (por consistencia numérica)
redundancy = (redundancy + redundancy.T) / 2


# 8. SELECCIÓN SECUENCIAL mRMR (criterio MID)

selected  = []
remaining = set(range(n_features))

selected_red = []
selected_rel = []

for k in range(N_FEATURES_TO_SELECT):
    best_score = -np.inf
    best_idx   = None
    for idx in remaining:
        rel = relevance[idx]
        if selected:
            red = np.mean([redundancy[idx, s] for s in selected]) 
        else:
            red = 0.0    
        score = rel - red
        if score > best_score:
            best_score = score
            best_idx   = idx
            best_rel = rel
            best_red = red
    selected.append(best_idx)
    remaining.remove(best_idx)
    selected_red.append(best_red)
    selected_rel.append(best_rel)
    

selected_names = [VAR_NAMES[i] for i in selected]
print("\nVariables seleccionadas por mRMR (orden de incorporación):")

for i in range(len(selected_names)):
    name = selected_names[i]
    rel  = selected_rel[i]
    red  = selected_red[i]
    if i == 0:
        print(f"{i+1:2d}. {name}   (relevancia: {rel:.4f}). (redundancia: --)")
    else:
        print(f"{i+1:2d}. {name}   (relevancia: {rel:.4f}). (redundancia: {red:.4f})")
 

# 9. GUARDAR CSVs CON SOLO LAS VARIABLES SELECCIONADAS

OUTPUT_DIR = ""
 
# Presencias
df_pres_out = df[['species', 'lon', 'lat'] + selected_names].copy()
pres_out_path = f"{OUTPUT_DIR}/presencias_mrmr.csv"
df_pres_out.to_csv(pres_out_path, index=False)
print(f"\nPresencias guardadas : {pres_out_path}  ({len(df_pres_out)} filas)")
 
# Background
df_bg = pd.DataFrame(bg_coords, columns=['lat', 'lon'])
for name in selected_names:
    idx = VAR_NAMES.index(name)
    df_bg[name] = bg_values[:, idx]
df_bg.insert(0, 'species', 'background')
bg_out_path = f"{OUTPUT_DIR}/background_mrmr.csv"
df_bg.to_csv(bg_out_path, index=False)
print(f"Background guardado  : {bg_out_path}  ({len(df_bg)} filas)")



# 10. VISUALIZACIONES

# Relevancia de todas las variables
fig, ax = plt.subplots(figsize=(12, 4))
colors = ['#e07b54' if VAR_NAMES[i] in selected_names else '#5b8db8'
          for i in range(n_features)]
ax.bar(VAR_NAMES, relevance, color=colors)
ax.set_xticks(range(n_features))
ax.set_xticklabels(VAR_NAMES, rotation=45, ha='right', fontsize=8)
ax.set_ylabel("MI con presencia/background")
ax.set_title("Relevancia de variables  (naranja = seleccionadas por mRMR)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("mrmr_relevancia.png", dpi=150, bbox_inches="tight")
plt.show()

# Matriz de redundancia
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(redundancy, cmap='YlOrRd', interpolation='nearest')
ax.set_xticks(range(n_features))
ax.set_yticks(range(n_features))
ax.set_xticklabels(VAR_NAMES, rotation=90, fontsize=7)
ax.set_yticklabels(VAR_NAMES, fontsize=7)
plt.colorbar(im, ax=ax, label='MI')
ax.set_title("Redundancia entre variables")
plt.tight_layout()
plt.savefig("mrmr_redundancia.png", dpi=150, bbox_inches="tight")
plt.show()

# 11. MATRIZ DE INFORMACIÓN MUTUA ENTRE SELECCIONADAS

selected_indices = [VAR_NAMES.index(name) for name in selected_names]
mi_selected = redundancy[np.ix_(selected_indices, selected_indices)]
print("\nMatriz de información mutua entre variables seleccionadas:")
print("            " + " ".join(f"{name:>8}" for name in selected_names))
for i, name_row in enumerate(selected_names):
    print(f"{name_row:>10} " + " ".join(f"{mi_selected[i, j]:8.4f}" for j in range(len(selected_names))))


# 12. MAPA DE CALOR DE INFORMACIÓN MUTUA ENTRE SELECCIONADAS

df_mi = pd.DataFrame(mi_selected, index=selected_names, columns=selected_names)

nombres_cortos = {}
df_mi = df_mi.rename(index=nombres_cortos, columns=nombres_cortos)
mask = np.triu(np.ones_like(mi_selected, dtype=bool), k=0)

plt.figure(figsize=(9, 7))
heatmap = sns.heatmap(
    df_mi,
    mask=mask,                
    annot=True,                   
    fmt=".4f",
    cmap="YlOrRd",              
    linewidths=0.5,            
    cbar_kws={'label': 'Información mutua (MI)'},
    vmin=0,                      
    square=True                   
)

plt.title("Redundancia entre variables seleccionadas (MI)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("mrmr_heatmap_seleccionadas.png", dpi=200, bbox_inches="tight")
plt.show()

