# ML1 — Examen Aplicado: Predicción del valor de viviendas en California

Examen aplicado de **Machine Learning I** — Universidad Mayor, Escuela de Ingeniería.
Autor: **Rodrigo Soto**.

---

## Dataset

| Campo | Valor |
|---|---|
| Nombre | California Housing Prices (censo de 1990) |
| Fuente | Kaggle |
| URL | https://www.kaggle.com/datasets/camnugent/california-housing-prices |
| Licencia | CC0 — Dominio público |
| Filas | 20.640 |
| Columnas | 10 (9 predictoras + 1 objetivo) |
| Variable objetivo | `median_house_value` (valor mediano de vivienda por distrito, USD) |
| Tipo de tarea | **Regresión** |

El dataset agrega información por distrito censal de California: ubicación geográfica,
antigüedad del parque habitacional, número de habitaciones, dormitorios, población, hogares,
ingreso mediano de los residentes y proximidad al océano.

---

## Metodología

1. **EDA** — `shape`, `dtypes`, `info()`, `describe()`, `head()`; análisis de nulos con heatmap;
   detección de outliers por método IQR con boxplots antes/después; distribución de la variable
   objetivo con skewness; heatmap de correlación de Pearson, scatter de las dos variables más
   correlacionadas y violin plot por `ocean_proximity`.
2. **Preprocesamiento sin data leakage** — split train/test (80/20, `random_state=42`) **antes**
   de cualquier transformación; `ColumnTransformer` con `SimpleImputer` (mediana/moda),
   `OneHotEncoder` (nominal), `OrdinalEncoder` (variable ordinal de antigüedad creada por tramos)
   y `StandardScaler`; `fit_transform` sólo sobre train, `transform` sobre test.
3. **No supervisado** — PCA con scree plot y líneas de referencia en 80% y 90%; selección de
   4 componentes (90,7% de varianza acumulada); análisis de loadings; K-Means con K de 2 a 10,
   método del codo y Silhouette Score; perfil de clusters.
4. **Modelado supervisado** — Ridge (`GridSearchCV`, alphas `[0.001, 0.01, 0.1, 1, 10, 100]`,
   cv=5) y Random Forest (`GridSearchCV` sobre `n_estimators`, `max_depth`, `min_samples_split`,
   cv=5). `random_state=42` en todos los estimadores. Métricas sobre **test**: RMSE, MAE, R², MAPE.
5. **Interpretación** — importancia de variables, análisis de las 10 observaciones con mayor
   error y conclusiones ejecutivas.

---

## Resultados

Tabla comparativa completa en [`tabla_comparativa_modelos.csv`](tabla_comparativa_modelos.csv).

| Modelo | RMSE | MAE | R² (test) | MAPE | R² (train) | Tiempo entren. (s) | Tiempo infer. (s) |
|---|---|---|---|---|---|---|---|
| **Random Forest** | **48.901,48** | **31.683,39** | **0,8175** | **0,1772** | 0,9760 | 813,84 | 0,2724 |
| Ridge | 70.093,84 | 51.452,83 | 0,6251 | 0,3013 | 0,6695 | 1,55 | 0,0005 |

**Modelo seleccionado: Random Forest** (`n_estimators=300`, `max_depth=None`,
`min_samples_split=2`), superior en las cuatro métricas. Las variables más importantes son
`median_income` (0,49), `ocean_proximity_INLAND` (0,14) y las coordenadas geográficas
(`longitude` 0,11, `latitude` 0,10).

---

## Estructura del repositorio

```
├── ML1_ExamenAplicado_Soto_Rodrigo.ipynb   # notebook ejecutado
├── data/housing.csv                        # dataset
├── figures/                                # 13 gráficos (dpi=150)
├── tabla_comparativa_modelos.csv           # tabla comparativa de modelos
├── varianza_pca.csv                        # varianza explicada por componente
├── perfil_clusters.csv                     # perfil de clusters K-Means
├── peores_predicciones.csv                 # 10 observaciones con mayor error
├── requirements.txt
└── README.md
```

---

## Reproducir el análisis

```bash
pip install -r requirements.txt
jupyter notebook ML1_ExamenAplicado_Soto_Rodrigo.ipynb
```

El notebook se ejecuta de inicio a fin sin errores. Todos los estimadores usan
`random_state=42`, por lo que los resultados son reproducibles. El `GridSearchCV` del Random
Forest tarda aproximadamente 15 minutos en una máquina de 2 núcleos.

---

## Video

Enlace al video (máx. 8 min, YouTube no listado): **https://youtu.be/8RDWZytHPs8**

---

## Declaración de uso de IA generativa

En el desarrollo de este examen se utilizó IA generativa (Claude, de Anthropic) como
herramienta de apoyo para la estructuración del notebook, la redacción de las celdas de
documentación en Markdown y la revisión del código. La selección del dataset, las decisiones
metodológicas (tratamiento de nulos y outliers, elección de modelos e hiperparámetros) y la
interpretación de los resultados fueron revisadas y validadas por el autor. Todos los
resultados numéricos, tablas y gráficos provienen de la ejecución real del notebook sobre el
dataset declarado.
