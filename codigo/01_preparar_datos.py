import os
import glob

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterstats import zonal_stats
from sklearn.neighbors import BallTree, KDTree
from scipy.special import digamma
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy.stats import gaussian_kde

CUBA_SHP    = ""
MANGROVE_SHP   = ""
BIOCLIM_DIR    = ""
SOILGRIDS_DIR  = ""
DISTANCIA_DIR  = ""
OUTPUT_CSV     = ""
CROPPED_DIR    = ""
ALIGNED_DIR    = ""
OUTPUT_CLEAN   = ""
OUTPUT_FINAL   = ""

REF_RASTER     = os.path.join(CROPPED_DIR, "")

os.makedirs(CROPPED_DIR, exist_ok=True)
os.makedirs(ALIGNED_DIR, exist_ok=True)


cuba      = gpd.read_file(CUBA_SHP).to_crs("EPSG:4326")
mangroves = gpd.read_file(MANGROVE_SHP).to_crs("EPSG:4326")
print("Capas cargadas.")

print("Recortando manglares a Cuba...")
mangroves_cuba = gpd.clip(mangroves, cuba)

shp_polygons = os.path.join(CROPPED_DIR, "manglares_cuba.shp")
mangroves_cuba.to_file(shp_polygons, driver="ESRI Shapefile")
print(f"Polígonos guardados: {shp_polygons}")

# Calcular centroides en UTM y volver a WGS84
points = mangroves_cuba.copy()
utm_crs = points.estimate_utm_crs()
points  = points.to_crs(utm_crs)
points["geometry"] = points.geometry.centroid
points  = points.to_crs(mangroves_cuba.crs)
points["id"] = range(len(points))

print(f"{len(points)} centroides generados.")
print("CRS actual:", points.crs)

#SPATIAL THINNING

KDE_BANDWIDTH = Silverman
RANDOM_SEED   = 42

def kde_probabilistic_thinning(gdf, bandwidth=None, random_seed=0):
    rng = np.random.default_rng(random_seed)
    gdf_geo = gdf.to_crs("EPSG:32617")

    lon = np.array(gdf_geo.geometry.x)
    lat = np.array(gdf_geo.geometry.y)
    coords = np.vstack([lon, lat])

    kde     = gaussian_kde(coords, bw_method=bandwidth or "silverman")
    density = kde(coords)

    c        = density.min()
    p_retain = np.minimum(c / density, 1.0)

    u    = rng.uniform(0.0, 1.0, size=len(gdf_geo))
    mask = u < p_retain

    return gdf[mask].to_crs(gdf.crs), p_retain


points_thinned, p_retention = kde_probabilistic_thinning(
    points, bandwidth=KDE_BANDWIDTH, random_seed=RANDOM_SEED
)
points_thinned = points_thinned.to_crs("EPSG:4326")
print(f"Puntos originales   : {len(points)}")
print(f"Puntos tras thinning: {len(points_thinned)}")
print(f"   Eliminados          : {len(points) - len(points_thinned)}")

#METRICAS DE SESGO

def _utm_from_gdf(gdf):
    """ reproyectar a UTM zona 17N (EPSG:32617) - adecuado para Cuba"""
    return gdf.to_crs("EPSG:32617")

def cv_densidad_kde(gdf, bandwidth="silverman"):
    """Coeficiente de Variación de la densidad KDE estimada en cada punto."""
    g = _utm_from_gdf(gdf)
    lon = np.array(g.geometry.x)
    lat = np.array(g.geometry.y)
    if len(lat) < 5:
        return {"cv": np.nan, "densidades": np.array([]), "n": len(lat)}
    
    kde = gaussian_kde(np.vstack([lon, lat]), bw_method=bandwidth)
    densidades = kde(np.vstack([lon, lat]))
    cv = densidades.std() / densidades.mean() if densidades.mean() > 0 else np.nan
    return {"cv": cv, "densidades": densidades, "n": len(lat)}

def clark_evans(gdf, area_km2, perimetro_km = None, alpha=0.05):
    """Índice R de Clark–Evans con corrección de borde de Donnelly (1978)."""
    from scipy.stats import norm
    g = gdf.to_crs("EPSG:4326")
    coords_rad = np.radians(np.column_stack([g.geometry.y, g.geometry.x]))
    n = len(coords_rad)
    
    if n < 5:
        return {"R": np.nan, "z": np.nan, "p_valor": np.nan,
                "d_obs_km": np.nan, "d_esp_km": np.nan,
                "interpretacion": "Muestra insuficiente"}
        
    tree = BallTree(coords_rad, metric="haversine")
    dist_rad, _ = tree.query(coords_rad, k=2)
    dist_km = dist_rad[:, 1] * 6371.0
    d_obs = dist_km.mean()
    rho = n / area_km2
    d_esp = 1.0 / (2.0 * np.sqrt(rho))
    
    if perimetro_km is None:
        perimetro_km = 2.0 * np.sqrt(np.pi * area_km2)
    d_esp_corr = d_esp + (0.0514 + 0.041 / np.sqrt(n)) * perimetro_km / n
    se = 0.26136 / np.sqrt(n * rho)
    R = d_obs / d_esp_corr
    z = (d_obs - d_esp_corr) / se
    p_valor = 2.0 * (1.0 - norm.cdf(abs(z)))
    if p_valor > alpha:
        interp = "aleatoriedad espacial (CSR)"
    elif z < 0:
        interp = "agrupamiento significativo"
    else:
        interp = "dispersión regular significativa"
    return {"R": R, "z": z, "p_valor": p_valor, "d_obs_km": d_obs,
            "d_esp_km": d_esp_corr, "interpretacion": interp, "n": n}

def entropia_espacial(gdf, bbox = None, grid_size=20):
    """Entropía espacial discreta normalizada sobre cuadrícula G×G."""
    g = _utm_from_gdf(gdf)
    lon = np.array(g.geometry.x)
    lat = np.array(g.geometry.y)
    
    if bbox is None:
        xmin, ymin, xmax, ymax = lon.min(), lat.min(), lon.max(), lat.max()
    else:
        xmin, ymin, xmax, ymax = bbox

    # Evitar división por cero si todos los puntos caen exactamente en la misma coordenada
    if xmax - xmin == 0 or ymax - ymin == 0:
        # Caso degenerado: una sola celda ocupa todos los puntos
        counts = np.array([[len(lon)]])
        H_norm = 0.0
        H = 0.0
        ocupadas = 1
        total = 1
        return {"H_norm": H_norm, "H_nats": H, "H_max": 0.0,
                "grid_counts": counts, "grid_size": grid_size, "n": len(lon),
                "celdas_ocupadas": ocupadas, "celdas_totales": total}

    # Índices de celda (de 0 a grid_size-1)
    col_idx = np.floor((lon - xmin) / (xmax - xmin) * grid_size).astype(int).clip(0, grid_size - 1)
    row_idx = np.floor((lat - ymin) / (ymax - ymin) * grid_size).astype(int).clip(0, grid_size - 1)

    # Matriz de conteos
    counts = np.zeros((grid_size, grid_size), dtype=int)
    for r, c in zip(row_idx, col_idx):
        counts[r, c] += 1

    n = len(lon)
    p = counts.flatten() / n
    p_pos = p[p > 0]
    H = -np.sum(p_pos * np.log(p_pos))
    H_max = np.log(grid_size ** 2)
    H_norm = H / H_max if H_max > 0 else 0.0

    return {"H_norm": H_norm, "H_nats": H, "H_max": H_max,
            "grid_counts": counts, "grid_size": grid_size, "n": n,
            "celdas_ocupadas": int((counts > 0).sum()),
            "celdas_totales": grid_size ** 2}



AREA_CUBA_KM2 = 109_884

gdf_orig_utm = points.to_crs("EPSG:32617")
BBOX_CUBA     = gdf_orig_utm.total_bounds
GRID_SIZE     = 20   # 20×20 = 400 celdas

datasets_bias = [
    ("Original",    "#5b8db8", points),
    ("KDE thinning", "#5db87a", points_thinned),
]

print("\n" + "=" * 65)
print("  MÉTRICAS DE SESGO DE MUESTREO ESPACIAL")
print("=" * 65)
print(f"{'Conjunto':<22} {'N':>5}  {'CV densidad':>11}  "
      f"{'R Clark-Evans':>13}  {'H norm':>8}  {'Celdas ocup.':>12}")
print("-" * 65)

resultados_bias = []
for nombre, color, gdf in datasets_bias:
    cv  = cv_densidad_kde(gdf)
    ce  = clark_evans(gdf, AREA_CUBA_KM2)
    ent = entropia_espacial(gdf, BBOX_CUBA, GRID_SIZE)
    resultados_bias.append({"nombre": nombre, "color": color,
                             "cv": cv, "ce": ce, "ent": ent})
    print(f"{nombre:<22} {cv['n']:>5}  "
          f"{cv['cv']:>11.4f}  "
          f"{ce['R']:>13.4f}  "
          f"{ent['H_norm']:>8.4f}  "
          f"{ent['celdas_ocupadas']:>5}/{ent['celdas_totales']:<5}")

print("=" * 65)
print("CV densidad   : ")
print("R Clark-Evans : ")
print("H normalizada : ")
for r in resultados_bias:
    ce = r['ce']
    print(f"  {r['nombre']}: R = {ce['R']:.3f}, "
          f"z = {ce['z']:.2f}, p = {ce['p_valor']:.4f} → {ce['interpretacion']}")




def crop_raster(input_path, cuba_geom, output_dir):
    """Recorta un raster al polígono de Cuba y lo guarda como GeoTIFF comprimido."""
    filename = os.path.basename(input_path).replace(".tif", "")
    out_path = os.path.join(output_dir, f"{filename}_cuba.tif")
    if os.path.exists(out_path):
        return out_path
    with rasterio.open(input_path) as src:
        out_img, out_transform = mask(src, cuba_geom, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height"   : out_img.shape[1],
            "width"    : out_img.shape[2],
            "transform": out_transform,
            "compress" : "lzw"
        })
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_img)
    return out_path


all_tifs = (glob.glob(os.path.join(BIOCLIM_DIR,   "*.tif")) +
            glob.glob(os.path.join(DISTANCIA_DIR,  "*.tif")) +
            glob.glob(os.path.join(SOILGRIDS_DIR,  "*.tif")))

cropped_paths = [crop_raster(t, cuba.geometry, CROPPED_DIR) for t in all_tifs]
print(f"{len(cropped_paths)} rásteres recortados a Cuba.")

print("\n Rásteres recortados:")
for p in cropped_paths:
    if p:
        print(f"  • {os.path.basename(p)}")

  # Prefijos que identifican variables categóricas → nearest; el resto → bilinear
CATEGORICAL_PREFIXES = ("TAXOUSDA", "TAXNWRB") 

def es_categorica(fname):
    return any(fname.upper().startswith(p.upper()) for p in CATEGORICAL_PREFIXES)


def align_rasters_to_reference(input_folder, output_folder, ref_raster_path):
    """
    Reproyecta y alinea todos los .tif de input_folder al grid del raster de
    referencia, guardando los resultados en output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    with rasterio.open(ref_raster_path) as ref:
        ref_crs       = ref.crs
        ref_transform = ref.transform
        ref_width     = ref.width       
        ref_height    = ref.height   

    print(f"Referencia : {os.path.basename(ref_raster_path)}")
    print(f"Resolución : {ref_transform.a:.6f}° | Tamaño: {ref_width}×{ref_height} | CRS: {ref_crs}")

    for tif_path in glob.glob(os.path.join(input_folder, "*.tif")):
        fname    = os.path.basename(tif_path)
        out_path = os.path.join(output_folder, fname)

        # Elegir método de remuestreo según tipo de variable
        metodo = Resampling.nearest if es_categorica(fname) else Resampling.bilinear

        with rasterio.open(tif_path) as src:
            profile = src.profile.copy()
            profile.update(
                crs       = ref_crs,
                transform = ref_transform,
                width     = ref_width,
                height    = ref_height,
                driver    = "GTiff",
                compress  = "lzw"
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                reproject(
                    source      = rasterio.band(src, 1),
                    destination = rasterio.band(dst, 1),
                    src_transform = src.transform,
                    src_crs       = src.crs,
                    dst_transform = ref_transform,
                    dst_crs       = ref_crs,
                    resampling    = metodo
                )

    print("Rásteres alineados correctamente.")


align_rasters_to_reference(CROPPED_DIR, ALIGNED_DIR, REF_RASTER)


def leer_nodata(raster_path):
    """Lee el valor NoData del metadata del raster. Devuelve None si no está definido."""
    with rasterio.open(raster_path) as src:
        return src.nodata

archivos_tif = sorted([
    os.path.join(ALIGNED_DIR, f)
    for f in os.listdir(ALIGNED_DIR)
    if f.lower().endswith(".tif")
])

for r_path in archivos_tif:
    var_name  = os.path.basename(r_path).replace("_cuba.tif", "").replace(".tif", "")
    nodata_val = leer_nodata(r_path)

    stats = zonal_stats(points_thinned, r_path, stats=["mean"], nodata=nodata_val)
    points_thinned[var_name] = [s["mean"] for s in stats]
    print(f"{var_name} extraído.")

df = pd.DataFrame({
    "species": "Manglar",
    "id"     : points_thinned["id"].values,
    "lon"    : points_thinned.geometry.x.values,
    "lat"    : points_thinned.geometry.y.values,
})

env_vars = [c for c in points_thinned.columns if c not in ["id", "geometry"]]
df = pd.concat([df, points_thinned[env_vars].reset_index(drop=True)], axis=1)

# Eliminar filas con NaN (MaxEnt no acepta valores faltantes)
n_antes = len(df)
df = df.dropna()
print(f"Puntos eliminados por NaN : {n_antes - len(df)}")
print(f"Puntos válidos            : {len(df)}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"CSV guardado: {OUTPUT_CSV}")

#LIMPIEZA ADICIONAL

df_clean = pd.read_csv(OUTPUT_CSV)

no_ambientales = ["id", "lon", "lat", "species", "Manglar"]
env_cols = [c for c in df_clean.columns if c not in no_ambientales]

# Valores NoData conocidos de WorldClim, SoilGrids y otros productos estándar
nodata_values = [-3.4e+38, -9999.0, -9999, -32768, 255, 65535]
df_clean = df_clean.replace(nodata_values, np.nan)
df_clean = df_clean.dropna(subset=env_cols)

print(f"Puntos originales              : {len(pd.read_csv(OUTPUT_CSV))}")
print(f"Puntos tras limpieza adicional : {len(df_clean)}")

df_clean.to_csv(OUTPUT_CLEAN, index=False)
print(f"CSV limpio guardado: {OUTPUT_CLEAN}")
