# Revisión antes del guion: puntos que el jurado podría notar

Revisión del dashboard y del modelo hecha el 12/10/2026 sobre la rama `arreglos-dashboard-video`, con el pipeline regenerado localmente desde `stuff_model_df.parquet` (las cifras coinciden con la guía con diferencias de centésimas). **Estado al 14/10/2026: A2 a A10 resueltos (A11 queda fuera del relato del video) y B1 a B3 incorporados al documento técnico en la rama `feature/ajustes-post-pr6` (guía §42); el resto sigue abierto.** Al escribirse, ningún punto estaba corregido: la idea es decidir, antes de escribir el guion del video (entrega 22/10/2026), si cada uno se **arregla**, se **explica** o **no se muestra**.

Por qué importa: el reto evalúa cuatro criterios con 25% cada uno (calidad del modelo, robustez fuera de muestra, integración arsenal + altitud y aplicabilidad). Los puntos del bloque A afectan sobre todo **aplicabilidad** (lo que ve un coach o el director deportivo); los del bloque B son preguntas probables en la **entrevista de 10 minutos**.

Cifras de 2026, pitchers con ≥ 100 pitches, salvo que se indique otra cosa.

## Resumen

| # | Punto | Bloque | Dónde se ve | Costo de arreglarlo |
|---|---|---|---|---|
| A1 | Harp Helú Fit casi repite el Stuff+ | Dashboard | Roster, Player, Scouting | Bajo (nombre/texto) o medio (redefinir) |
| A2 | Game y Simulator › Ubicación usan otra variante del modelo | Dashboard | Game, Simulator | Medio |
| A3 | Game y Simulator › Ubicación califican un "pitch promedio" | Dashboard | Game, Simulator | Medio |
| A4 | "Más/menos efectivos" del mapa de movimiento es circular | Dashboard | Scouting › Perfil ideal, Simulator | Bajo (nombre) o medio (redefinir) |
| A5 | Alertas por cuantil, sin piso de magnitud | Dashboard | Roster, Player, Inbox | Bajo |
| A6 | Oportunidades y "mejor pitch" sesgados contra las rectas | Dashboard | Inbox, Player, comparador | Bajo |
| A7 | "Arsenal ideal" recomienda menos rectas con una métrica ciega a los roletazos | Dashboard + modelo | Scouting › Arsenal ideal | Bajo (explicar) o medio (cruzar con contacto) |
| A8 | "Confianza" ambigua | Dashboard | Roster, Player | Bajo |
| A9 | Probabilidades absolutas sin ajuste de nivel | Dashboard | Game, Player | Bajo |
| A10 | Documentos desalineados con el producto | Documentación | Design Thinking, guía §37.2 | Bajo |
| A11 | Partes del diseño sin construir | Dashboard | Player, Simulator | No arreglar; dejar fuera del relato |
| B1 | Stuff+ lo domina su componente más débil | Modelo | Entrevista | Explicar |
| B2 | La variante titular de altitud se eligió y se confirmó con el mismo dato | Modelo | Entrevista, documento técnico | Explicar |
| B3 | Límites ya documentados | Modelo | Entrevista, documento técnico | Explicar |

## A. Dashboard

### A1. Harp Helú Fit casi repite el Stuff+

- **Qué se ve:** cada pitcher y cada pitch tienen un "Harp Helú Fit" (Strong / Moderate / Low / Poor), que se lee como "qué tan bien encaja con CDMX".
- **Evidencia:** Fit = percentil del Stuff+ proyectado en Harp Helú (`23_dashboard_data.py:290`). La correlación de rangos entre ese Stuff+ proyectado y el Stuff+ a nivel del mar es **0.941** (369 pitchers); contra el Stuff+ de temporada, 0.943. La información propia de la altitud (cuánto pierde un pitch más o menos de lo esperado para su nivel) está en el eje vertical del mapa del Roster y en las alertas, no en el Fit.
- **Riesgo:** pregunta directa del jurado: "¿qué agrega el Fit sobre el Stuff+?".
- **Opciones:** (a) renombrarlo a "Stuff+ en Harp Helú (percentil)"; (b) redefinirlo combinando nivel y sensibilidad a la altitud; (c) mantenerlo y tener lista la respuesta: "el Fit es la calidad donde jugamos; la sensibilidad a la altitud se lee en el eje vertical del mapa".

### A2. Game y Simulator › Ubicación usan otra variante del modelo

- **Qué se ve:** un slider o un changeup puede verse casi intacto en Player, Roster y Simulator › What-if, y claramente peor en Harp Helú dentro de Game o de Simulator › Ubicación.
- **Evidencia:** desde la auditoría 2 (guía §38), el titular de altitud es la variante física (sin contexto de arsenal) combinada con el nivel oficial (`engine.population_table(..., hybrid=True)`). Game (`dashboard/lib/engine.py:231`, `game_grid`) y Simulator › Ubicación (`dashboard/views/simulator.py:242`, `altitude_table(..., ctx=ctx)`) califican Pitching+ por entorno con la variante con arsenal, la que también castiga que la separación contra la recta se encoja. Guía §37.6: con arsenal el slider pierde −2.32 y el changeup −2.87; sin arsenal, −0.22 y −0.76.
- **Opciones:** aplicar el mismo esquema híbrido al Pitching+ por entorno, o advertirlo en pantalla.

### A3. Game y Simulator › Ubicación califican un "pitch promedio"

- **Qué se ve:** para el mismo pitch, los números de Game y de Simulator › Ubicación no cuadran con los de Simulator › What-if o Player.
- **Evidencia:** Game usa `engine.pitch_means` (`engine.py:222`) y Simulator › Ubicación usa el escenario promedio (`simulator.py:228` y `:242`); What-if califica todos los pitches reales del tipo (`population_table`, `engine.py:129`). La guía §37.5 midió la diferencia por no linealidad del modelo: el mismo Four-Seam da 100 como "pitch promedio" y 107 como población. En las tarjetas de Game conviven ambos: el Stuff+ viene de la población y el Pitching+ del pitch promedio.
- **Opciones:** rejilla poblacional (promediar la rejilla sobre los pitches reales; más lento) o una nota en pantalla.

### A4. "Más/menos efectivos" del mapa de movimiento es circular

- **Qué se ve:** el mapa de movimiento (Scouting › Perfil ideal y fondo del Simulator) separa los pitches "más efectivos" y "menos efectivos" de la liga.
- **Evidencia:** los grupos son el quintil alto y bajo del Stuff+ proyectado en Harp Helú dentro de cada tipo (`dashboard/lib/data.py:124`). El mapa muestra, por tanto, el movimiento que el propio modelo premia. Hay respaldo parcial: el Stuff+ alto sí produce más whiff y chase reales en Extreme (guía §36.5), pero el reto pide "efectivos".
- **Opciones:** (a) renombrar a "mayor/menor Stuff+ proyectado"; (b) definir los grupos por resultados observados en Extreme (whiff, chase o valor realizado); (c) citar la validación en el pie del mapa.

### A5. Alertas por cuantil, sin piso de magnitud

- **Qué se ve:** siempre hay pitches en "Adjust" o "Review", aunque la pérdida de más sea pequeña.
- **Evidencia:** la acción usa el p10 y p25 del exceso dentro de cada tipo (`23_dashboard_data.py:237`), así que siempre marca alrededor del 10% y del 25% de cada tipo. Umbrales p25 en 2026: Four-Seam −1.00, Sinker −0.73, Changeup −0.79, Curveball −0.61, Slider −0.53 puntos de Stuff+. "Adjust" exige evidencia Robusta, que solo tienen Four-Seam y Sinker: las 42 acciones "Adjust" de 2026 son rectas (26 Four-Seam y 16 Sinker).
- **Opciones:** piso de magnitud (por ejemplo, exceso ≤ −1 punto) y un verbo menos prescriptivo que "Adjust" ("Revisar primero").

### A6. Oportunidades y "mejor pitch" sesgados contra las rectas

- **Qué se ve:** el Inbox casi nunca propone una recta como oportunidad, y el "mejor pitch en Harp Helú" casi nunca es una recta.
- **Evidencia:** la escala de Stuff+ no está normalizada por tipo (promedio de liga en Harp Helú 2026: Splitter 113, Slider 107, Changeup 106, Cutter 103, Curveball 99, Four-Seam 93, Sinker 92; guía §39.1). La oportunidad usa un umbral absoluto, Stuff+ en Harp Helú ≥ 105 (`23_dashboard_data.py:324`): de 144 oportunidades en 2026, 84 son sliders, 43 changeups, 8 curvas, 8 cutters y 1 Four-Seam. El "mejor pitch" (`23_dashboard_data.py:280`) es una recta en 1 de 371 pitchers (también lo era con el criterio anterior: 4 de 371).
- **Opciones:** usar el percentil dentro del tipo (el Fit por pitch) para oportunidades y para "mejor pitch".

### A7. "Arsenal ideal" recomienda menos rectas con una métrica ciega a los roletazos

- **Qué se ve:** Scouting › Arsenal ideal concluye que el arsenal que más rinde en Harp Helú "depende menos de la recta" (FF+FC+SL arriba con 103.5; FF+SI+CU+CH abajo con 95.8; −0.61 de Stuff+ en Harp Helú por cada +10 pp de rectas).
- **Evidencia:** los pesos oficiales de Stuff+ en 2026 son chase 0.680, whiff 0.313 y HR dado contacto 0.005; roletazo ≈ 0 (`18_weights.csv`). La evidencia propia de H3 dice que el roletazo vale casi el doble en altitud (ventaja GB − FB: +0.14 carreras a nivel del mar contra +0.25 en Extreme) y que el mismo pitcher sufre +4.7 pp de HR/FB en Extreme (guía §33.4). Un sinkerballer sale castigado por construcción. Además, la pendiente de "dependencia de la recta" es consecuencia aritmética del Δ por tipo (Four-Seam −5.5, Sinker −3.0, el resto ≈ 0) multiplicado por el porcentaje de uso, no una evidencia independiente.
- **Opciones:** presentarlo como "lectura de swing-and-miss" y cruzarlo con la tasa de roletazos y el costo del contacto (rama de contacto, ya en Game) antes de recomendar roster; evitar una recomendación categórica contra los sinkerballers.

### A8. "Confianza" ambigua

- **Qué se ve:** barras de confianza junto al Stuff+ y al Fit, como si calificaran al score.
- **Evidencia:** la confianza del pitcher es Robust si ≥ 60% de sus pitches son de tipos con evidencia Robusta y tiene ≥ 300 pitches (`23_dashboard_data.py:294`). Solo Four-Seam y Sinker son Robustos, así que un pitcher de rompientes nunca es Robust. En realidad mide la evidencia del efecto de altitud, no la del score.
- **Opciones:** etiquetarla "confianza del efecto de altitud".

### A9. Probabilidades absolutas sin ajuste de nivel

- **Qué se ve:** Game muestra whiff y strike cantado como porcentajes absolutos.
- **Evidencia:** la guía §35.4 documenta deriva de la tasa base entre temporadas y que la recalibración aprendida no ayuda; para mostrar probabilidades pedía un ajuste de nivel con la temporada en curso, que no se implementó. Los cambios (pp entre entornos) son más confiables que los niveles.
- **Opciones:** mostrar diferencias en lugar de niveles, o una nota de "nivel aproximado".

### A10. Documentos desalineados con el producto

- `Design_Thinking_Dashboard_Diablos_Rojos.md` no cambia desde el 30/09: sigue hablando de "Home vs Away" (los datos no tienen local/visitante), de Game como V2, de un desglose del Fit y de un what-if de uso.
- La guía §37.2 conserva las definiciones viejas del Fit y del semáforo sin anotación (las vigentes están en §38); es la fuente probable del documento técnico.
- `Dashboard_Analisis_Factibilidad.md` dice tendencia semanal con ≥ 20 pitches; el código usa ≥ 10 (`23_dashboard_data.py:298`).
- **Opciones:** anotar §37.2 y marcar el Design Thinking como histórico; en el video usar el vocabulario del producto.

### A11. Partes del diseño sin construir

Player › History (el wireframe tiene "View History"), comparación entre experimentos guardados y la Home/Away Strategy Card. `history_by_type` se genera pero ninguna vista lo usa. Para un video de 5 minutos no hacen falta: dejarlos fuera del relato.

## B. Modelo (preparar respuestas para la entrevista)

### B1. Stuff+ lo domina su componente más débil

- **Evidencia:** pesos oficiales de Stuff+ 2026: chase 0.680 y whiff 0.313. AUC de la familia física en 2026 (`18_component_metrics.csv`): chase **0.547**, whiff **0.628**. Es decir, 68% del Stuff+ viene de la predicción de chase que la física apenas logra (el chase depende sobre todo de la ubicación). La regla de pesos (guía §33.6) se adoptó después de descartar los pesos propios de la familia por un orden por tipo implausible.
- **Respuesta sugerida:** el Stuff+ se valida a nivel pitcher, no a nivel pitch: split-half 0.99, año a año 0.78/0.77 y generaliza a pitchers no vistos y a Extreme (guía §35). La versión con pesos iguales predice casi igual el valor de la temporada siguiente (correlación 0.12 [−0.02, 0.26] contra 0.15 [0.04, 0.27] con los pesos oficiales; `18_pitcher_level_check.csv`), así que el orden no depende de esa elección. Además, en altitud extrema el quintil alto de Stuff+ genera de 5.9 a 12.1 pp más chase real que el bajo (`22_profile_outcome_validation.csv`): el componente débil por pitch sí se traduce en chase real al agregar. Y al revés, sin usar el modelo: el 20% con más whiff real de cada tipo tiene más Stuff+ en los seis tipos (guía §42.2).
- **Estado (14/10):** incorporado al documento técnico, sección 3.5.

### B2. La variante titular de altitud se eligió y se confirmó con el mismo dato

- **Evidencia:** la variante física se volvió titular porque coincide con el contraste observado del mismo pitcher (guía §38.1), y el nivel "Robusto" exige confirmación observada (guía §36.3): el mismo dato se usa para elegir y para confirmar. H2 cambia de magnitud con esa elección: changeup y splitter pierden 0.76 menos que los pitches de spin con la variante con arsenal, y 2.30 menos con la física.
- **Respuesta sugerida:** decirlo como decisión a posteriori, mostrar ambas variantes para H2 y explicar el argumento físico (la variante con arsenal también castiga que la separación contra la recta se encoja, algo que en el mismo pitcher observado no aparece). Verificado: H2 = +2.30 [+1.99, +2.72] con la variante física y +0.76 con la de arsenal (`21_delta_rows.parquet`, 2026); en las dos, changeup y splitter pierden menos.
- **Estado (14/10):** incorporado al documento técnico (sección 6.2 y tabla de hipótesis).

### B3. Límites ya documentados

Extreme Altitude agrupa varios parques (4–5 equipos locales; no solo el Harp Helú); no hay corredores en base, así que no hay run expectancy de 24 estados; barrel no se implementa (faltan tablas de xBA/xSLG); el tunneling es solo distancia del punto de release; no hay SHAP (el "Why?" es narrativo); la calibración deriva entre temporadas. Están en la guía §38.4 y deben ir en "alcance y fronteras" del documento técnico.

**Estado (14/10):** todos están en la sección 8 del documento técnico; se agregó barrel y xBA/xSLG.

## Orden sugerido para decidir

1. **Lo que va a salir en pantalla en el video:** A1, A2, A5, A6 y A7.
2. **Arreglos de texto, baratos:** A4, A8 y A9.
3. **Respuestas para la entrevista y el documento técnico:** B1, B2 y B3.
4. **Higiene de documentos:** A10.
