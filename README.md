# Diablos Rojos — Hackathon ISAC 2026

Proyecto de análisis de lanzamientos para estudiar su calidad física (Physical Stuff) y las características del arsenal de cada pitcher.

## Antes de empezar

1. Clona el repositorio y abre su carpeta en VS Code.
2. Coloca el dataset original **`stuff_model_df.parquet`** en esa misma carpeta. Se obtiene del material compartido del hackathon: **no viene en GitHub ni lo generan los scripts**.
3. Instala las dependencias en un entorno virtual con **Python 3.9 a 3.13** (scikit-learn 1.6.x, fijado en `requirements.txt` para que los modelos guardados se abran en cualquier equipo, no existe para Python 3.14):

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Windows cmd:        .venv\Scripts\activate.bat

python -m pip install -r requirements.txt
```

En VS Code, selecciona el intérprete de `.venv` mediante **Python: Select Interpreter**. Si tu equipo usa `python3` en lugar de `python`, sustituye el comando (en Windows también funciona `py -3.13 -m venv .venv`).

## Probar el prototipo del dashboard (macOS, Windows o Linux)

Con el entorno activado y el parquet en la carpeta del repo:

```bash
python correr_prototipo.py
```

Corre la cadena mínima que necesita el dashboard (`02 → 17 → 18 → 19 --stage cv → 20 → 21 → 22 → 23 → 24`), omite los pasos cuyos resultados ya existen, verifica que el motor reproduce los scores oficiales de `18` (diferencia ~1e-14) y abre Streamlit en el navegador (normalmente `http://localhost:8501`).

- La primera corrida completa tarda del orden de 30 minutos (los pasos largos son `18`, `19` y `21`). Si alguien comparte sus salidas ya generadas (`17_*`, `18_*`, `19_cv_oof.parquet`, `20_*`, `21_*`, `22_*`), cópialas a la carpeta y esos pasos se omiten.
- `python correr_prototipo.py --todo` vuelve a correr todo; `--sin-abrir` prepara todo sin abrir Streamlit. Los logs de cada paso quedan en `prototipo_logs/`.
- La primera vez que se usa Streamlit en un equipo, pide un correo en la terminal antes de levantar el servidor: presiona Enter para dejarlo vacío.
- El pipeline reescribe `22_dashboard_manifest.json` (fechas y hashes); el runner avisa al final. No lo subas salvo que quieras actualizar el del equipo.

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
python 16_three_scores.py
python 17_sequence_and_contact.py
python 18_three_scores_v2.py
python 19_validation.py
python 20_altitude_physics.py
python 21_altitude_stuff_delta.py
python 22_profile_recommendations_export.py
python 23_dashboard_data.py
python 24_arsenal_harp.py
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
| `16_three_scores.py` | **Corrección post-auditoría (Bloque A).** Tres scores separados por pitch en escala 100+10z: **Stuff+** (solo física en release, sin ángulos de release/aproximación, que reconstruían la ubicación con R² 0.995), **Location+** y **Pitching+**. Horizontales en convención de diestro, variante sin arsenal para pitchers nuevos (cobertura 100%, columna `information_level`) y pesos aprendidos contra el valor por conteo con predicciones out-of-fold del historial (V1 queda como sensibilidad). Requiere `13` importable (no su salida). Genera `16_three_scores.csv`, `16_component_metrics.csv`, `16_calibration_by_altitude.csv`, `16_weights.csv`, `16_out_of_sample_value.csv`, `16_location_leak_check.csv` y `16_three_scores_bundle.pkl`. Guía, sección 32. |
| `17_sequence_and_contact.py` | **Bloque B.** Ordena los pitches con el timestamp del UUID v1, reconstruye plate appearances (98.8% empiezan en 0-0, 99.7% de conteos coherentes) y construye `prev_pitch_type`, `out_within_PA`, `hard_hit` (EV ≥ 95 mph), `weak_contact` (EV < 60 mph) y `hr_given_contact`. Estima el run expectancy LMB por conteo × outs, con las funciones `fit_re`/`realized_value_re` que usa `18`, y la evidencia descriptiva de H3 por altitud. Genera `17_pitch_sequence.parquet`, `17_sequence_validation.csv`, `17_run_expectancy.csv`, `17_contact_by_altitude.csv` y `17_contact_within_pitcher.csv`. Guía, sección 33. |
| `18_three_scores_v2.py` | **Scores oficiales** (arquitectura congelada, guía 34; incluye la rama de contacto como módulo explicativo y su ablación). Mismo pipeline de `16` con 9 componentes (se agregan weak/hard-hit/HR en contacto y out en la PA), Pitching+ con `Outs`/`Inning`/`prev_pitch_type`, valor LMB por conteo × outs (sin The Book) y Stuff+ ponderado con el valor por evento de Pitching+ sin called strike. Requiere correr `17` antes. Genera `18_*.csv`, `18_three_scores_bundle.pkl` y `18_pitcher_level_check.csv`. Guía, sección 33. |
| `19_validation.py` | **Bloque C.** Validación formal de los scores de `18` sin cambiar la arquitectura: CV por pitcher (GroupKFold 5), park holdout en 3 direcciones, estabilidad (split-half, año a año) y predicción del valor de la temporada siguiente, y calibración con recalibración 2025 → 2026. Se corre por etapas (`--stage cv|park|stability|calibration|summary`; `cv` y `park` tardan horas). Genera `19_*.csv`, `19_cv_oof.parquet` y `19_exit_table.csv`. Guía, sección 35. |
| `20_altitude_physics.py` | **Bloque D (D1–D2).** Compara transformaciones del movimiento por altitud (aditiva contra multiplicativas) prediciendo la temporada futura, y mide el gradiente No → Medium → Extreme. Elegida: multiplicativa global (IVB × 0.77, HB × 0.71 en Extreme). Genera `20_*.csv` y `20_transform_choice.json`. Guía 36. |
| `21_altitude_stuff_delta.py` | **Bloque D (D3–D4).** ΔStuff+ de altitud con el Stuff+ limpio (bootstrap por pitcher, chequeo inverso, sensibilidad a la forma y contraste observado del mismo pitcher), niveles de evidencia y tabla H1–H4. Genera `21_*.csv`, `21_delta_rows.parquet` y `21_transform_r.json`. Guía 36. |
| `22_profile_recommendations_export.py` | **Bloque D (D5–D7).** Perfil físico ideal en altitud desde el Stuff+ limpio (validado contra whiff/chase reales), recomendaciones por tipo, proyección de cada pitcher a nivel del mar contra Harp Helú, `simulate_pitch()` para el dashboard y manifiesto congelado `22_dashboard_manifest.json`. Guía 36. |
| `23_dashboard_data.py` | **Dashboard: capa de datos.** Convierte los outputs congelados (17–22) en las tablas de `dashboard/data/` (pitchers, arsenal, fingerprint, inbox, targets, tendencias, equipos reconstruidos, historial). Aquí viven todas las reglas de producto (Fit, semáforo, alertas, confianza, targets). Guía 37. |
| `24_arsenal_harp.py` | **Arsenal en Harp Helú.** Responde qué repertorio produce el Stuff+ más alto en altitud: Stuff+ por composición del arsenal, dependencia de la recta, descomposición mezcla contra calidad y los 10 mejores arsenales individuales. Lee la capa de datos de `23`. Guía 39. |
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

## Corrección post-auditoría (25/09/2026)

Una auditoría encontró que las features "físicas" de `03`/`07`/`09` incluían `VertRelAngle`, `HorzRelAngle`, `VertApprAngle` y `HorzApprAngle`, que junto con el movimiento reconstruyen la ubicación en el plato (R² 0.995 en pitchers no vistos). Consecuencias:

- El "Stuff+" de `07` no era físico (AUC de whiff 0.7268; sin ángulos ~0.63).
- El score de `09` es un **Pitching+**, no un Stuff+, y su mejora frente a `07` no se atribuye a arsenal ni altitud (ablación limpia: arsenal −0.0026 AUC en whiff, altitud ≈ 0).
- `07`–`15` quedan como **arquitectura experimental**. La metodología de altitud (`10`–`14`) y el perfil (`15`) se rehacen sobre `16` (Bloque D).

`16_three_scores.py` (Bloque A) produce los tres scores. En la temporada 2026: Stuff+ físico AUC de whiff 0.628 (pitchers nuevos 0.620); los tres scores superan al nulo por conteo (reducción de MSE: Pitching+ 2.10%, Location+ 1.93%, Stuff+ 0.10%, todos con IC que excluye 0) y los pesos aprendidos superan a V1 en las tres familias. Pendientes: Bloque B (orden de pitches por UUID, `Outs`, `prev_pitch_type`, hard-hit/HR), Bloque C (suite de validación y calibración) y Bloque D (altitud, perfil y recomendaciones). Detalle en la guía, sección 32.

## Bloque B (26/09/2026)

`17` reconstruye la secuencia y las PA a partir del UUID y estima un run expectancy LMB por conteo × outs, sin pagos externos. Evidencia de H3: la ventaja en valor de un roletazo sobre un elevado pasa de +0.14 runs (No Altitude) a +0.25 (Extreme); para el mismo pitcher, en Extreme el HR/FB sube 4.7 pp. `18` produce los **scores oficiales**:

- **Pitching+:** reduce 1.80% del MSE contra el nulo por conteo × outs en 2026.
- **Location+:** 1.70%.
- **Stuff+:** ~0% a nivel pitch, pero es muy estable por pitcher (split-half 0.99) y predice el valor de la temporada siguiente (corr 0.15, IC que excluye 0), sin superar al valor realizado (0.22).

Los componentes de contacto reciben peso ≈ 0, así que H3 queda cuantificada de forma descriptiva y no dentro del score (decisión abierta, guía 33.9). Siguiente: Bloque C (validación) y Bloque D (altitud).

## Rama de contacto y arquitectura congelada (26/09/2026)

`18` agrega una rama de valor de contacto, P(bola en juego) × E[valor | bola en juego], y la compara contra la versión sin rama en la misma corrida. Stuff+ y Location+ son neutrales al entorno; Pitching+ usa el run expectancy de su entorno. La rama **no mejora** el valor fuera de muestra (Pitching+ 2026: 1.56% con rama contra 1.66% sin ella; por pitcher 0.09 contra 0.19), así que queda como **columna explicativa de H3** (`*__p_contact_value`) y no entra al score. El contrafactual ambiental es coherente en signo: con la etiqueta Extreme, el mismo pitch tiene más daño esperado (−0.2 a −0.35 carreras por 100 pitches), menos si induce roletazos.

**Scores oficiales:** `stuff_plus`, `location_plus` y `pitching_plus` en `18_three_scores.csv`. Las versiones con rama están en `*_con_contacto`. La configuración de la guía 33 se reproduce con `python 18_three_scores_v2.py --sin-rama-contacto` (prefijo `18A`). La arquitectura queda congelada para el Bloque C. Guía, sección 34.

## Bloque C: validación (26/09/2026)

`19` valida la arquitectura congelada sin modificarla:

- **Stuff+:** es la medida más repetible (split-half 0.99; año a año 0.78/0.77 por pitcher, contra 0.26/0.22 del valor observado) y generaliza a pitchers no vistos y a Extreme sin verlo (mismo AUC).
- **Location+ y Pitching+:** reducen ~1.7% del error contra el nulo por conteo en la CV por pitcher, con calibración de pendiente ≈ 1.
- **Fallos conocidos:** ningún score supera al valor observado en correlación con la temporada siguiente, aunque sí en error fuera del par de ajuste, y lo mejor es combinarlos. La recalibración aprendida con la temporada anterior no ayuda (deriva de temporada). Algunas probabilidades físicas de Stuff+ no deben mostrarse como calibradas. Detalle en la guía, sección 35.

## Bloque D: altitud, perfil y recomendaciones (26/09/2026)

- **Transformación de altitud:** la multiplicativa predice mucho mejor la temporada futura que la aditiva. En Extreme, el mismo pitcher conserva ~77% de su IVB y ~71% de su HB. Medium cae entre No y Extreme en los siete tipos.
- **Efecto en el Stuff+ limpio (físico, neutral al entorno):** *(cifras de la variante oficial; el titular cambió en la auditoría 2, ver abajo)* Four-Seam pierde 4.8 puntos y Sinker 2.8, confirmado por el Stuff+ observado del mismo pitcher (−4.7 y −1.6). En slider, changeup y curva la pérdida modelada (1.2–2.3) no aparece en el neto observado, así que son **condicionales**. Cutter y Splitter quedan **inciertos**.
- **Hipótesis:** H1 soportada. H2 soportada en términos relativos (changeup/splitter pierden 0.8 menos que los tipos Magnus). H3 soportada de forma descriptiva. H4 parcial.
- **Perfil ideal:** sale solo del Stuff+ limpio y se valida contra whiff y chase reales en Extreme. Reemplaza al perfil de `15`.

Los outputs quedaron congelados en `22_dashboard_manifest.json`, con `simulate_pitch()` verificado para el simulador. Detalle en la guía, sección 36.

## Dashboard — Pitching Decision System (fase 1, 30/09/2026)

Diseño en `Design_Thinking_Dashboard_Diablos_Rojos.md` y en los wireframes; el análisis de lo que los modelos sostienen está en `Dashboard_Analisis_Factibilidad.md`. Para ejecutarlo, lo más simple es `python correr_prototipo.py` (sección "Probar el prototipo del dashboard"). A mano, desde la raíz del repo y después de correr la secuencia 17–22:

```bash
python 23_dashboard_data.py
python 24_arsenal_harp.py
python -m streamlit run dashboard/app.py
```

Pantallas listas: **Roster** (inbox, mapa Stuff+ × adaptación, tabla con tendencia y comparador), **Player Overview** (recomendación, arsenal, fingerprint de movimiento a nivel del mar vs. Harp Helú, adaptación, Why? y development targets) y **Simulator** (el mismo pitch a nivel del mar, en Medium y en Harp Helú; what-if de física; heatmap de Location+ por count y lado; experimentos guardados) y **Scouting** (perfil ideal por tipo, explorador de la liga con señales Stuff+ contra resultados, evaluación de prospectos con arsenal completo, comparador y shortlist) y **Game** (lectura táctica por count y lado del bateador: calidad esperada por pitch y heatmaps de Pitching+ y del costo del contacto, a nivel del mar vs. Harp Helú). El motor (`dashboard/lib/engine.py`) reproduce exactamente los scores oficiales; se verifica con `python dashboard/lib/engine.py`. Las cinco pantallas del diseño están integradas.

El dataset es anónimo: el roster se elige entre equipos reconstruidos por co-aparición (4–5 son locales en Extreme Altitude) o como lista personalizada, que se guarda en `dashboard/state/`. Guía, sección 37.

## Auditoría 2: Fit, alertas y titular de altitud (30/09/2026)

- **Titular de altitud = variante física.** El Stuff+ se evalúa sin contexto de arsenal para todos los pitches: mide solo lo que el aire le hace al shape propio. Coincide con lo observado en el mismo pitcher (2026: Four-Seam −5.4 vs. −5.4 observado; Sinker −2.9 vs. −2.5; Curveball −1.3 vs. −1.1; Changeup −0.6 vs. −0.9; Slider −0.2 vs. +0.3). Niveles de evidencia: Robusto FF y SI; Condicional CU y CH; Incierto SL, FC y FS. La variante con arsenal (el Stuff+ oficial de 18) también castiga que la separación contra la recta se encoja, y por eso exagera la pérdida de slider y changeup. Se reporta como **sensibilidad**.
- **Nivel proyectado en Harp Helú** = Stuff+ oficial a nivel del mar + Δ físico. Así lo calculan `22` y `23`, y así lo reproduce el motor del dashboard (`population_table` y `prospect` con `hybrid=True`).
- **Harp Helú Fit** = calidad proyectada en Harp Helú (percentil del Stuff+ proyectado entre pitches del mismo tipo o entre pitchers). Ya no es "cuánto pierde", porque eso premiaba a los pitchers con poco Stuff+.
- **Alertas y acciones** = el pitch pierde *más de lo esperado* para su nivel y su tipo: residuo del Δ después de controlar por el Stuff+ a nivel del mar dentro de cada tipo. Adjust ≤ p10 con evidencia Robusta; Review ≤ p25. El mapa del Roster usa ese percentil en el eje vertical.
- **Bug corregido:** el Simulator ya no falla con pitchers que solo tienen datos de 2025. Dependencias fijadas en `requirements.txt`.

Detalle en la guía, sección 38.

## Huecos frente al reto: arsenal y mapa de movimiento (05/10/2026)

- **¿Qué arsenal produce el Stuff+ más alto en Harp Helú?** El que depende menos de la recta y la acompaña con slider o cutter. A nivel del mar el porcentaje de rectas no cambia el Stuff+; en Harp Helú cada +10 puntos porcentuales de rectas cuestan 0.57 de Stuff+ adicionales [0.45, 0.68] en 2026 (0.56 en 2025). En 2026 la composición más alta es FF+FC+SL (103.5) y la más baja es FF+SI+CU+CH, sin slider (95.8).
- **Matiz:** la mezcla de tipos explica solo 7% de las diferencias de Stuff+ entre pitchers en Harp Helú (1% a nivel del mar). Pesa más la calidad de cada pitch que el repertorio. El análisis es descriptivo, no causal.
- **Mapa de movimiento de los pitches más y menos efectivos:** en Scouting › Perfil ideal (por tipo) y como fondo del mapa del Simulator. Más y menos efectivos = quintil alto y bajo de Stuff+ proyectado en Harp Helú dentro de cada tipo.
- **Dashboard:** Scouting tiene la pestaña nueva "Arsenal ideal".

Detalle en la guía, sección 39.

## Consistencia semántica del dashboard (05/10/2026)

- **Regla transversal:** cada número pertenece a un estado explícito (No Altitude, Harp Helú o todos los parques) y nunca se mezclan en una misma comparación.
- **Resiliencia a la altitud (Scouting):** ahora es "pierde menos o más de lo esperado para su nivel y tipo", no los puntos brutos que pierde. En el Four-Seam la velocidad pasa de −0.47 a +0.21.
- **Development Targets (Player):** columnas "Baseline (No Altitude)" y "En Harp Helú". El movimiento se compara en Harp Helú, porque la zona objetivo se midió en pitches lanzados en altitud.
- **Net effect (Player):** Neutral si el cambio es menor a 1 punto; Mixed si hay señales en conflicto (el modelo de arsenal da el signo contrario, o whiff o chase van en contra). En 2026, de los 37 pitches que ganan 1 punto o más en altitud por sí solos, 24 pierden 1 punto o más según el modelo de arsenal: su separación contra la recta se encoge.
- **Whiff y chase por entorno (`23`):** ahora usan el mismo esquema que el Stuff+ proyectado (nivel oficial + efecto físico). Antes salían de la variante con arsenal y podían contradecir al Δ mostrado.
- `Dashboard_Analisis_Factibilidad.md` y el manifiesto de `22` quedaron al día.

Detalle en la guía, sección 40.

## Arreglos para el video y datos incompletos (12/10/2026)

- **Game:** sin el borde rojo que señalaba "los dos mejores" pitches; cada tarjeta muestra el Pitching+ máximo junto al costo del contacto en esa misma zona (Pitching+ no penaliza el contacto, guía 34).
- **Roster:** KPI, mapa, orden y columna principal usan el Stuff+ a nivel del mar. El de temporada baja con cada pitch lanzado en altitud y castigaba a los equipos locales en Extreme (equipo por defecto: 99.5 de temporada contra 102.2 a nivel del mar).
- **Player:** "Primary risk" y "Siguiente paso" señalan el pitch que pierde más de lo esperado para su nivel (exceso), igual que las alertas; antes era casi siempre la recta (90% → 47% de los pitchers en 2026). `23` alinea `worst_pitch`, `most_resilient_pitch` y `best_pitch` (calidad en Harp Helú).
- **Scouting › Evaluar prospecto:** selector "Medido en" que lleva el movimiento a nivel del mar antes de simular; los rasgos del perfil se comparan en Harp Helú.
- **Datos incompletos (`23`):** un tipo sin proyección ya no anula el promedio del arsenal; un valor faltante queda "Sin muestra" en lugar de contar como percentil 0; los pitches sin movimiento medido (433 en 2026) quedan sin proyección en lugar de Δ = 0. Alertas 2026: 38 rojas, 66 amarillas y 267 verdes (antes 39/65/267).
- **Multiplataforma:** `correr_prototipo.py` reemplaza los pasos manuales en macOS, Windows y Linux; `requirements.txt` acota pandas < 3, numpy < 2.4 y scikit-learn 1.6.x; el estado del dashboard se guarda en UTF-8.

Pendiente de decidir antes del guion: `Revision_Pre_Guion.md`. Falta registrar estos cambios en la guía y la bitácora.

## Ajustes posteriores al PR 6 (14/10/2026)

- **Player:** la tarjeta principal y la tabla del arsenal muestran el Stuff+ a nivel del mar, igual que el Roster; el de temporada queda como dato secundario.
- **Exceso del pitcher (`23`):** promedia solo los tipos con valor; antes rellenaba con 0 los tipos sin exceso.
- **`correr_prototipo.py`:** guarda un sello por paso (`prototipo_logs/sellos.json`) con el hash del script que generó cada salida. Si un script cambió, vuelve a correr ese paso y los siguientes. Las salidas sin sello (copiadas de otro equipo o anteriores al runner) se reutilizan con un aviso; `--todo` las regenera.
- **Cifras 2026 regeneradas en esta máquina:** 369 pitchers con proyección; alertas 38 rojas, 65 amarillas y 266 verdes. Detalle en la guía, sección 41.

## Revisión previa al guion: A2, A4, A5, A6 y A7 (14/10/2026)

- **A2 · Game y Simulator › Ubicación:** el Pitching+ por entorno usa el mismo esquema híbrido que el resto (nivel oficial a nivel del mar + efecto físico). El slider ya no pierde en Harp Helú por la separación contra la recta.
- **A4 · Mapa de más y menos efectivos:** se define por el whiff real de la temporada (o el chase real), no por el propio Stuff+. Validación no circular: en los seis tipos, el grupo con más whiff real tiene más Stuff+ proyectado.
- **A5 · Alertas:** Adjust y Review exigen además perder al menos 1 punto de más. En 2026, Review baja de 176 a 126 y las alertas quedan en 38 rojas, 51 amarillas y 280 verdes.
- **A6 · Oportunidades y mejor pitch:** por percentil dentro del tipo, porque la escala de Stuff+ no se recentra por tipo. Las rectas pasan de 1 a 31 Four-Seam y 25 Sinker en oportunidades.
- **A7 · Arsenal ideal:** contrastado con resultados reales. En el valor realizado en altitud no se detecta el costo de las rectas, y los sinkerballers inducen más roletazos (sinker ≥ 30% del arsenal: 51.9% [50.2, 53.6] en 2025 y 51.2% [49.2, 53.1] en 2026, contra 46.2% y 46.4% del resto; roletazos entre bolas en juego sin toques, promedio por pitcher; corregido el 14/10: antes se contaban también los fouls, que traen `hit_type` en 74,638 filas, y daba 41% contra 35–36%). La brecha casi no cambia (+5.7 y +4.8 pp contra +5.6 y +5.0), pero la tasa por cuartil de % de rectas deja de bajar (38% → 35% en 2025) y queda plana (47% → 49%). Se presenta como lectura de swing-and-miss, no como recomendación de roster.

- **A3 · Game y Simulator › Ubicación:** promedian los pitches reales del pitcher (muestra fija de hasta 150 por tipo) en lugar de un "pitch promedio", igual que Player y el What-if.

- **A8 · Confianza:** se etiqueta "Confianza · altitud", porque mide la evidencia del efecto de altitud, no la del Stuff+.
- **A9 · Probabilidades:** whiff, chase y strike cantado se muestran como niveles aproximados (≈), con una nota de que los cambios entre entornos son más confiables que los niveles.
- **A10 · Documentos:** `Design_Thinking_Dashboard_Diablos_Rojos.md` queda marcado como histórico.
- **B1–B3 · Entrevista:** el peso del chase (68% del Stuff+, AUC físico 0.547), la elección a posteriori de la variante física (H2: +2.30 física, +0.76 arsenal) y los límites quedan declarados en el documento técnico; las respuestas, en `Revision_Pre_Guion.md`.

Detalle en la guía, sección 42.

## Corrección del What-if (14/10/2026)

Al preparar el guion del video apareció un error: en el Simulator, un cambio de física (por ejemplo, +2 in de IVB) subía el Stuff+ a nivel del mar pero no en Harp Helú. El esquema híbrido mezclaba la sensibilidad de dos variantes del modelo. Ahora el efecto del cambio se mide con la variante oficial en cada entorno. Detalle en la guía, sección 42.9.
