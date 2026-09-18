# Diablos Rojos — Hackathon ISAC 2026

Proyecto de análisis de lanzamientos para estudiar su calidad física (Physical Stuff) y las características del arsenal de cada pitcher.

## Antes de empezar

1. Clona el repositorio y abre su carpeta en VS Code.
2. Coloca el dataset original **`stuff_model_df.parquet`** en esa misma carpeta. Se obtiene del material compartido del hackathon: **no viene en GitHub ni lo generan los scripts**.
3. Instala las dependencias en un entorno virtual:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install pandas numpy pyarrow scikit-learn
```

En VS Code, selecciona el intérprete de `.venv` mediante **Python: Select Interpreter**. Si tu equipo usa `python3` en lugar de `python`, sustituye el comando.

## Ejecutar en orden

Abre la terminal **en la carpeta del repositorio**, donde están los scripts, y ejecuta uno por uno. Espera a que termine cada paso; si aparece un error, resuélvelo antes de continuar.

```bash
python 01_eda.py
python 02_feature_engineering.py
python 03_stuff_baseline.py
python 04_fold_safe_arsenal.py
```

| Archivo | Qué hace |
| --- | --- |
| `01_eda.py` | Explora los datos, su calidad, cobertura y relaciones iniciales. |
| `02_feature_engineering.py` | Construye las features exploratorias de arsenal y genera `stuff_model_features_v1.parquet`. |
| `03_stuff_baseline.py` | Compara el baseline global, el baseline por tipo de lanzamiento y un modelo físico para `is_whiff`, con pitchers separados entre train y test. Incluye diagnósticos numéricos. |
| `04_fold_safe_arsenal.py` | Construye y audita referencias de arsenal sin leakage, tanto en pitcher holdout como en ventanas históricas 2024 → 2025 y 2024–2025 → 2026. No entrena modelos. Genera `04_primary_fastball_changes.csv`. |
| `explorar_datos.py` | Inspección rápida del tamaño del dataset y del diccionario. Opcional; no forma parte de la secuencia principal. |

**`02` debe ejecutarse antes de `03` y `04`**, porque ambos leen el parquet que genera. Los resultados principales se imprimen en la terminal. Volver a ejecutar `02` o `04` reemplaza sus respectivos archivos generados locales.

## Datos y archivos generados

- `.gitignore` excluye los archivos `.parquet`, `.pkl`, `.rds` y los CSV generados.
- `stuff_plus_data_dictionary.csv` es la excepción: describe las variables, está versionado y lo utilizan `01` y `explorar_datos.py`.
- `04_fold_safe_arsenal_audit.txt` conserva una salida de auditoría de referencia; `04` imprime la auditoría en la terminal, no actualiza ese TXT automáticamente.

## Dos decisiones importantes

- Las features de temporada completa de `02` sirven para exploración. Las referencias de arsenal para validación deben construirse con la lógica de `04`, usando únicamente datos permitidos de entrenamiento o historial previo.
- Para pitchers completamente nuevos, las features de arsenal quedan ausentes por diseño. No se rellenan con información de test. Con historial previo se utiliza la validación temporal independiente.

`04` está cerrado. El siguiente paso previsto es `05_arsenal_incremental_value.py` (todavía no incluido): evaluar si el arsenal histórico aporta valor adicional sobre Physical Stuff, comparando exactamente las mismas filas elegibles. Esa mejora aún no se ha demostrado.

## Documentación del proyecto

- **`ISAC_Diablos_Rojos_Guia_Base_Modelado*.docx`**: explicación metodológica y decisiones técnicas.
- **`Bitacora_Modelado_ISAC*.xlsx`**: avances, experimentos, estado y próximos pasos.

Consulta la versión más reciente disponible de cada documento para el detalle. Este README es solo la guía rápida para empezar.
