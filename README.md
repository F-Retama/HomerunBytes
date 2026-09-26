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
python 05_target_definitions.py
python 06_arsenal_incremental_value.py
python 07_boosting_models.py
python 08_park_holdout.py
python 09_pitching_plus.py
python 10_altitude_counterfactual.py
python 11_altitude_total_effect.py
python 12_weight_sensitivity.py
python 13_count_value_weights.py
python 14_v1_v2_altitude_matrix.py
python 15_profile_and_recommendations.py
```

| Archivo | Qué hace |
| --- | --- |
| `01_eda.py` | Explora los datos, su calidad, cobertura y relaciones iniciales. |
| `02_feature_engineering.py` | Construye las features exploratorias de arsenal y genera `stuff_model_features_v1.parquet`. |
| `03_stuff_baseline.py` | Compara el baseline global, el baseline por tipo de lanzamiento y un modelo físico para `is_whiff`, con pitchers separados entre train y test. Incluye diagnósticos numéricos. |
| `04_fold_safe_arsenal.py` | Construye y audita referencias de arsenal sin leakage, tanto en pitcher holdout como en ventanas históricas 2024 → 2025 y 2024–2025 → 2026. No entrena modelos. Genera `04_primary_fastball_changes.csv`. |
| `05_target_definitions.py` | Audita categorías reales, targets, consistencia y distribuciones de contacto en el dataset original completo. Guarda `05_target_definitions_audit.txt`. No entrena modelos ni fija thresholds de contacto. |
| `06_arsenal_incremental_value.py` | Compara Null, Physical Stuff y Physical Stuff + Historical Arsenal para `is_whiff` sobre las mismas filas elegibles de cada ventana season-forward (2024→2025, 2024–2025→2026), reutilizando las funciones de `04`. Genera `06_arsenal_incremental_results.csv`. |
| `07_boosting_models.py` | Entrena boosting (`HistGradientBoostingClassifier`; LightGBM/XGBoost no cargan en este entorno por falta de `libomp`, ver nota abajo) para 5 targets V1 con el mismo pitcher holdout de `03`/`04`: `whiff_per_pitch`, `whiff_given_swing`, `groundball` (familia Stuff+, solo físicas) y `called_strike`, `chase` (familia Location+, ubicación+count+BatterSide). No incluye arsenal ni altitud. Genera `07_boosting_results.csv`. |
| `08_park_holdout.py` | Entrena en No Altitude + Medium Altitude y evalúa en Extreme Altitude, reutilizando los targets y `build_model()` de `07`. `altitude_category` **no** se usa como feature (decisión documentada en la guía, sección 25.1). Genera `08_park_holdout_results.csv` y compara contra `07_boosting_results.csv`. |
| `09_pitching_plus.py` | Pitching+/Out+ V1: reentrena los 5 targets con el set de features completo (físicas + ubicación + arsenal + `altitude_category` + count), season-forward. Anota cada modelo sobre la población común elegible (no solo su población condicional de entrenamiento), combina con z-score y pesos iguales, y normaliza a escala Stuff+ (100+10z). Genera `09_pitching_plus_scores.csv` (score por pitch, con `altitude_category`/`pitcher_anon_id`/`year` — insumo del contrafactual de altitud) y `09_pitching_plus_component_metrics.csv`. |
| `10_altitude_counterfactual.py` | Contrafactual de altitud sobre el score de `09`: carga los modelos exactos de `09_pitching_plus_bundle.pkl` (correr `09` primero), asigna `altitude_category` = No / Medium / Extreme (proxy del entorno Harp Helú) manteniendo todo lo demás fijo y reporta ΔStuff+ y los 5 deltas de componentes por pitch type, shape bin y pitcher (n, media, mediana, SD, p10/25/75/90). Es efecto **directo**, no total (guía, sección 27). Incluye contraste observacional within-pitcher. Genera `10_altitude_*.csv`. |
| `11_altitude_total_effect.py` | Delta **total** de altitud (bloque separado de `10`): transformación física conjunta por pitch type (solo IVB, HB y ángulos de aproximación; release como placebo) estimada con el historial y validada en la temporada futura, aplicada a pitches observados en No Altitude; bootstrap por pitcher (`--bootstrap`, default 200; se corrió con 500) con IC95%, contribución por componente y robustez a quitar groundball. Requiere `09` (bundle) y el parquet. Genera `11_altitude_total_*.csv`. |
| `12_weight_sensitivity.py` | V2 paso 1: curva del delta total de altitud contra el peso relativo de groundball y punto de quiebre por pitch type (IC95% bootstrap por pitcher), sin tocar V1. Usa `11_altitude_total_reps.pkl` (correr `11` primero). Genera `12_gb_weight_*.csv`. |
| `13_count_value_weights.py` | V2 pasos 2-5: valor por conteo (cadena de Markov de 12 estados; pagos terminales de The Book/FanGraphs 2006, excepción documentada en la guía 29.1) y pesos no negativos del compuesto, con estabilidad entre ventanas y RMSE fuera de muestra contra el baseline nulo. Genera `13_*.csv` y `13_v2_bundle.pkl`. |
| `14_v1_v2_altitude_matrix.py` | V2 paso 6: matriz V1 vs V2 del delta total por pitch type y niveles de evidencia (Robusto / Condicional / Incierto). Genera `14_v1_v2_matrix_by_window.csv` y `14_evidence_levels.csv`. |
| `15_profile_and_recommendations.py` | Perfil físico/arsenal por pitch type en Extreme (zonas P25-P75 del quintil superior de Stuff+ V1, lift con IC95% por pitcher; aprendido con 2025 y validado en 2026 contra valor realizado por pitch), sensibilidad V2, recomendaciones por nivel de evidencia y `pitcher_fit_scores`. **Ningún perfil valida contra valor realizado**: es exploratorio. Genera `15_pitch_type_profiles.csv`, `15_profile_validation.csv`, `15_altitude_recommendations.csv`, `15_pitcher_fit_scores.csv`. |
| `explorar_datos.py` | Inspección rápida del tamaño del dataset y del diccionario. Opcional; no forma parte de la secuencia principal. |

**`02` debe ejecutarse antes de `03` y `04`**, porque ambos leen el parquet que genera. Los resultados principales se imprimen en la terminal. Volver a ejecutar `02` o `04` reemplaza sus respectivos archivos generados locales.

`05` usa directamente `stuff_model_df.parquet`, sin los filtros V1 de `02`/`03`. Sus tasas describen todo el dataset y no son métricas de test del baseline. Al repetirlo se actualiza su reporte TXT. Puede ejecutarse desde cualquier carpeta; las rutas predeterminadas se resuelven junto al script.

## Datos y archivos generados

- `.gitignore` excluye los archivos `.parquet`, `.pkl`, `.rds` y los CSV generados.
- `stuff_plus_data_dictionary.csv` es la excepción: describe las variables, está versionado y lo utilizan `01` y `explorar_datos.py`.
- `04_fold_safe_arsenal_audit.txt` conserva una salida de auditoría de referencia; `04` imprime la auditoría en la terminal, no actualiza ese TXT automáticamente.

## Dos decisiones importantes

- Las features de temporada completa de `02` sirven para exploración. Las referencias de arsenal para validación deben construirse con la lógica de `04`, usando únicamente datos permitidos de entrenamiento o historial previo.
- Para pitchers completamente nuevos, las features de arsenal quedan ausentes por diseño. No se rellenan con información de test. Con historial previo se utiliza la validación temporal independiente.

`04` y `05_target_definitions.py` están cerrados. Targets V1: whiff por pitch, whiff dado swing, called strike desde PitchCall, chase fuera de zona y groundball en BIP con hit_type conocido y sin bunts. Weak/hard contact, out-within-PA y xRunValue quedan fuera de V1; no usar el flag is_hit_by_pitch. Las contradicciones se documentan sin corregir etiquetas.

`06_arsenal_incremental_value.py` está cerrado (24/09/2026): compara Null, Physical Stuff y Physical Stuff + Historical Arsenal para is_whiff sobre exactamente las mismas filas elegibles season-forward, reutilizando `04_fold_safe_arsenal.py` sin reimplementar su lógica de leakage. El arsenal histórico aporta valor incremental positivo y consistente (AUC +0.0027 en 2024→2025 y +0.0032 en 2024-2025→2026, con mejora también en Log Loss y Brier). Detalle en la guía, sección 23, y en `06_arsenal_incremental_results.csv`.

`07_boosting_models.py` está cerrado (24/09/2026, V1 sin arsenal ni altitud): boosting con el mismo pitcher holdout mejora fuertemente sobre el baseline logístico de `03` (whiff_per_pitch AUC 0.6136 → 0.7268) y produce el primer Location+ funcional del proyecto (called_strike AUC 0.8790, chase AUC 0.8419). Nota de entorno: LightGBM y XGBoost no cargan aquí por falta de `libomp` (el host no puede compilarlo); se usa `sklearn.ensemble.HistGradientBoostingClassifier` como sustituto directo. Detalle en la guía, sección 24, y en `07_boosting_results.csv`.

`08_park_holdout.py` está cerrado (24/09/2026): el AUC bajo park holdout (train en No Altitude + Medium Altitude, test en Extreme Altitude, sin `altitude_category` como feature) es prácticamente idéntico al pitcher holdout de `07` en los 5 targets (deltas entre -0.0083 y +0.0003) — evidencia real de robustez ambiental. Detalle e interpretación física de por qué generaliza tan bien en la guía, sección 25.

`09_pitching_plus.py` está cerrado (24/09/2026, V1 con pesos iguales): combina los 5 componentes (con arsenal+ubicación+altitud+contexto ya integrados) en un solo score Pitching+/Out+, normalizado a escala Stuff+ (100+10z). El sanity check por pitch type reproduce exactamente el orden de `03` (Splitter > Slider > Curveball > Changeup > Cutter > Sinker > Four-Seam), y agregar arsenal+altitud+contexto mejora cada componente sobre sus versiones puras de `07`/`08` (p.ej. whiff_per_pitch: 0.7268 → 0.7531 AUC). Metodología de la ponderación (por qué no se usaron pesos de valor de carrera todavía) en la guía, sección 26.

## Documentación del proyecto

- **`ISAC_Diablos_Rojos_Guia_Base_Modelado*.docx`**: explicación metodológica y decisiones técnicas.
- **`Bitacora_Modelado_ISAC*.xlsx`**: avances, experimentos, estado y próximos pasos.

Consulta la versión más reciente disponible de cada documento para el detalle. Este README es solo la guía rápida para empezar.

`10_altitude_counterfactual.py` (24/09/2026, V1 efecto directo): el delta directo es ≈0 y sin signo estable entre ventanas; aún no hay un ΔStuff+ por pitch shape reportable. Falta el delta total (movimiento por ambiente) y bootstrap. Detalle en la guía, sección 27.

`11_altitude_total_effect.py` (24/09/2026, **corregido el 25/09/2026**: la transformación ahora respeta la lateralidad, ya que HorzBreak/HorzApprAngle/RelSide tienen signo absoluto; ver guía, sección 30). Vigente: salvo Four-Seam en V1 (+0.30, por groundball) todos los tipos con muestra suficiente pierden valor; Curveball (−1.16 en V1) y Slider (−0.55) son robustos; Cutter y Splitter no permiten concluir.

`12`-`14` (V2 como sensibilidad; recalculados el 25/09/2026 tras la corrección de lateralidad, guía secciones 29-30): con pesos ligados a valor de carrera, Four-Seam pasa de +0.30 (V1) a −1.43 (V2); Sinker ≈ 0 en V1 y −0.64 en V2; Changeup, Slider y Curveball pierden en ambos lentes. La estabilidad estricta de los pesos no se cumple, así que V1 permanece como escala oficial. La regla de pesos externos se enmendó solo para eventos terminales.

`15` (25/09/2026): V1 es la escala oficial y V2 una sensibilidad; el perfil por pitch type es casi todo compartido con nivel del mar y no valida contra valor realizado (sí predice whiff observado en Four-Seam, Slider, Changeup y Cutter). Guía, sección 31.
