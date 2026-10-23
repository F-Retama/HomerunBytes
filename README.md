# Stuff+ ajustado por altitud para Diablos Rojos

**ISAC Sports Analytics Hackathon 2026 · presentado por Diablos Rojos del México**
Equipo: Daniela Biel Luna, Luis Fernando Rodríguez Retama y Alexa Morales Pérez Vargas.

Construimos un sistema que califica cada lanzamiento de la LMB con tres escalas separadas (**Stuff+**, **Location+** y **Pitching+**, donde 100 es el promedio de la liga y 10 puntos una desviación estándar) y estima cuánto cambia el Stuff+ de un pitch en la altitud del Harp Helú (2,240 m). Se entrenó con 635,002 lanzamientos de 1,134 pitchers (2024–2026), y todos los resultados reportados son fuera de muestra. Encima corre un dashboard para el cuerpo técnico, el *Pitching Decision System*.

## Empieza aquí

| Qué | Dónde |
| --- | --- |
| Documento técnico (10 páginas) | [`Documento_Tecnico_Diablos_Rojos.pdf`](Documento_Tecnico_Diablos_Rojos.pdf) |
| Video del dashboard | [Ver en Google Drive](https://drive.google.com/file/d/19y09JaeqQ2c27uvklJmGwdqo_vco85b7/view?usp=sharing) |
| Dashboard en línea | Link en el formulario de entrega (se hostea desde un repositorio privado aparte, ver abajo) |
| Todo el código (pipeline, dashboard y video) | [`codigo/`](codigo/) |
| Metodología completa, paso a paso | [`ISAC_Diablos_Rojos_Guia_Base_Modelado.docx`](ISAC_Diablos_Rojos_Guia_Base_Modelado.docx) |
| Historial de decisiones y correcciones | [`HISTORIAL.md`](HISTORIAL.md) |

## Resultados principales

- **Generaliza.** El AUC de whiff de Stuff+ es 0.625 / 0.628 / 0.630 con pitchers no vistos, en la temporada siguiente y en un entorno de altitud no visto.
- **La altitud quita movimiento.** El mismo pitcher conserva 77% de su movimiento vertical inducido y 71% del horizontal en altitud extrema.
- **El costo se concentra en las rectas.** Four-Seam −5.4 y Sinker −2.9 puntos de Stuff+, confirmados por lo observado en los mismos pitchers. Curveball −1.2 y Changeup −0.6 son condicionales; Slider, Cutter y Splitter no se distinguen de cero.
- **Comprime, no reordena.** Por pitcher el Δ medio es −2.6 (de −8.3 a +0.4), pero el orden de la liga casi no cambia (correlación de rangos 0.94).
- **Stuff+ es estable.** Correlaciona 0.77–0.78 de un año al siguiente, contra 0.22–0.26 del valor observado por pitch.

## Dónde está cada entregable del reto

| Entregable | En el documento | Scripts que lo producen |
| --- | --- | --- |
| 1. Stuff+ por pitch, pitcher y arsenal; CDMX contra otros estadios | Secciones 3–4, 6 y 7.3 | `18`, `21`, `22`, `24` |
| 2. Perfil físico ideal para Diablos Rojos | Sección 7.1 | `22` |
| 3. Recomendaciones operativas | Secciones 6.3 y 7.2 | `21`, `22`, `17` (contacto) |
| 4. Dashboard simulador | Sección 7.4 y el video | `23`, `codigo/dashboard/` |
| 5. Documento técnico | El PDF | `19` (validación), `20` (física) |

## Estructura del repositorio

```
README.md                            este archivo
Documento_Tecnico_Diablos_Rojos.pdf  documento técnico
ISAC_…_Guia_Base_Modelado.docx       metodología completa
HISTORIAL.md                         decisiones y correcciones, en orden cronológico
codigo/
  01_eda.py … 24_arsenal_harp.py     pipeline numerado (abajo, cuál es la ruta oficial)
  correr_prototipo.py                corre la cadena mínima y abre el dashboard
  dashboard/                         aplicación Streamlit (Roster, Player, Simulator, Scouting y Game)
  video/                             guion, narración y grabación automatizada del video
  *_audit.txt, *.json                salidas de auditoría y parámetros congelados, versionados a propósito
  requirements.txt                   dependencias
  stuff_plus_data_dictionary.csv     diccionario de variables del dataset
```

Todos los scripts resuelven sus rutas respecto a su propia carpeta, así que **todo se corre desde `codigo/`**.

### Ruta oficial y scripts de exploración

Los resultados del documento salen de la cadena **`02 → 17 → 18 → 19 → 20 → 21 → 22 → 23 → 24`**:

| Script | Qué hace |
| --- | --- |
| `02_feature_engineering.py` | Prepara las variables base (`stuff_model_features_v1.parquet`). |
| `17_sequence_and_contact.py` | Ordena los lanzamientos con el timestamp del UUID, reconstruye los turnos y estima la expectativa de carreras LMB por conteo × outs. |
| `18_three_scores_v2.py` | **Scores oficiales** Stuff+, Location+ y Pitching+ (arquitectura congelada antes de validar). |
| `19_validation.py` | Validación fuera de muestra: pitcher, season y park holdout, estabilidad y calibración. |
| `20_altitude_physics.py` | Transformación física del movimiento por altitud (IVB × 0.77, HB × 0.71). |
| `21_altitude_stuff_delta.py` | Δ de Stuff+ por altitud, niveles de evidencia e hipótesis H1–H4. |
| `22_profile_recommendations_export.py` | Perfil ideal, recomendaciones, proyección por pitcher y motor del simulador. |
| `23_dashboard_data.py` | Tablas del dashboard y reglas de producto (Fit, alertas). |
| `24_arsenal_harp.py` | Qué arsenal produce el Stuff+ más alto en Harp Helú. |
| `25_dashboard_lite.py` | Paquete ligero del dashboard para la versión en línea (sin el bundle completo de `18`). |

Los scripts **`01` a `16`** son las etapas previas: exploración, baselines, auditoría de targets y versiones anteriores de los scores y del efecto de altitud. Se conservan por dos razones. Primero, algunos son dependencias de la ruta oficial: `18` importa funciones de `16`, que a su vez usa `04`, `06`, `07` y `13`. Segundo, muestran cómo se llegó al resultado. Qué aportó y qué se descartó en cada uno está en [`HISTORIAL.md`](HISTORIAL.md) y en la guía.

## Cómo reproducir

**Los datos no vienen en el repositorio.** Las reglas del concurso permiten presentar resultados, no los datos. Quien tenga el dataset de ISAC debe colocar `stuff_model_df.parquet` dentro de `codigo/`; ningún script lo descarga ni lo genera.

1. Entorno con **Python 3.9 a 3.13**. scikit-learn está fijado en 1.6.x porque los modelos se guardan con pickle:

   ```bash
   cd codigo
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

2. Cadena mínima y dashboard en un solo paso (desde `codigo/`):

   ```bash
   python correr_prototipo.py
   ```

   Corre la ruta oficial omitiendo los pasos cuyos resultados ya existen, verifica que el motor del dashboard reproduce los scores oficiales de `18` (diferencia máxima ~1e-14) y abre Streamlit (normalmente en `http://localhost:8501`). La primera corrida tarda unos 30 minutos. `--todo` rehace toda la cadena y `--sin-abrir` no abre el navegador. Los logs quedan en `prototipo_logs/`.

3. Validación completa (las tablas de la sección 5 del documento). `19` se corre por etapas, y `cv` y `park` tardan horas:

   ```bash
   python 19_validation.py --stage cv
   python 19_validation.py --stage park
   python 19_validation.py --stage stability
   python 19_validation.py --stage calibration
   python 19_validation.py --stage summary
   ```

4. Versión en línea del dashboard: `python 25_dashboard_lite.py` (unos 10 minutos) genera `codigo/dashboard/lite/engine_lite.pkl`, compara contra la versión completa y guarda las diferencias en `25_dashboard_lite_check.csv`. Sin el bundle completo de `18`, el dashboard usa ese paquete automáticamente. Como el paquete trae una muestra de lanzamientos, la app se hostea en Streamlit Community Cloud desde un repositorio privado aparte con solo `dashboard/`, sus tablas y el paquete; este repositorio no los incluye.

5. Para correr un script suelto, el orden es el de su número. Cada uno imprime sus resultados y guarda sus salidas (`NN_*.csv`, `.parquet`, `.pkl`) en `codigo/`; `.gitignore` las excluye.

## Integridad del proceso

- **Arquitectura congelada antes de validar.** Variables, objetivos y pesos de `18` no cambiaron durante la validación de `19`; los fallos se reportan como hallazgos, por ejemplo que Stuff+ no supera al valor observado para predecir la temporada siguiente.
- **Sin fuga de información.** El arsenal de cada pitcher se mide contra su recta primaria calculada solo con temporadas anteriores (`04`), y la escala de 2026 usa como referencia 2024–2025.
- **Trazabilidad.** Cada decisión, corrección y auditoría está fechada en la guía, en `HISTORIAL.md` y en el historial de commits, incluida la corrección de lateralidad del 25/09/2026 y la auditoría previa al guion.
- **Los datos nunca se subieron.** El historial de git no contiene el dataset ni salidas con datos por lanzamiento. La versión en línea del dashboard usa un paquete ligero (`25`) con los modelos y una muestra de hasta 50 lanzamientos por pitcher y tipo (27 columnas, sin resultados ni IDs de juego, bateador o catcher). Ese paquete vive solo en el repositorio privado desde el que se hostea la app; su efecto en las cifras está en `25_dashboard_lite_check.csv` y en la sección 7.4 del documento.

## Uso de IA y trabajo de terceros

Se utilizó Claude (Anthropic) como apoyo en la programación, en la elaboración del video y en la redacción y gramática del documento técnico.

Librerías: scikit-learn (`HistGradientBoostingClassifier`), pandas, NumPy, SciPy, PyArrow, joblib, Streamlit, Plotly, openpyxl y python-docx. El video se grabó con Playwright e imageio-ffmpeg. Las referencias de la literatura (Tango, Lichtman y Dolphin; SABR; Alan Nathan; FanGraphs; Huang y Hsu; McBride) están en el documento técnico.

Los datos son de rendimiento deportivo, anónimos y sin datos personales, y el sistema no se usa para apuestas.
