# Propuesta: documento técnico recortado (de 21 a ~11 páginas)

Propuesta para revisar antes de pasarla al documento final. Parte de la exportación del 14/10/2026 (21 páginas) y la acerca al rango del brief (6 a 10 páginas) sin perder ninguno de los cinco requisitos del entregable 5: marco teórico; decisiones de modelado con arquitectura, features y función de pérdida; definición formal y normalización de la escala; validación fuera de muestra con pitcher, season y park holdout; alcance y fronteras.

**Cómo leerla.** Primero, la tabla de qué se recortó y por qué; después, las cifras que cambian; al final, el texto propuesto completo, listo para pegar. Las dos figuras del documento actual (diagrama de arquitectura y gráfica del Δ de Stuff+) se conservan sin cambios: en el texto aparecen como marcadores `[Figura …]`.

**Extensión estimada.** El documento actual tiene ~6,040 palabras en 21 páginas; descontando sus tres figuras (~1.8 páginas), son ~315 palabras por página de texto en la exportación actual. El texto propuesto tiene ~3,150 palabras (contadas igual, con tablas) más las dos figuras conservadas (~1.3 páginas): **~11 páginas** en el mismo formato, quizá 10 si las figuras se exportan más chicas. Si hace falta bajar a 10, lo siguiente que se puede quitar sin tocar requisitos es la tabla de hipótesis de 6.3 (pasarla a cuatro frases), la tabla del perfil de 7.1 (dejar Four-Seam, Slider y Changeup) y el párrafo de pantallas de 7.4 (el video las muestra).

**Sin anexo.** Un anexo contaría para el límite de páginas. El material retirado se remite a la guía metodológica y a los scripts del repo (línea final "Código y resultados").

## Qué se recortó y por qué

| Sección | Antes (pp.) | Propuesta (pp.) | Qué cambia | Por qué |
|---|---|---|---|---|
| Encabezado y resumen | 1.2 | 0.6 | Fecha nueva. Tabla de 6 preguntas en una línea por fila; la del arsenal dice que es una lectura de swing-and-miss y que en valor real no se detecta. | El resumen repetía las secciones; la fila del arsenal podía leerse como recomendación de roster (Revisión A7). |
| 1. Marco teórico | 1.3 | 0.8 | 1.1–1.3 en tres párrafos cortos; se conservan Nathan/Coors, H1–H4 y la tabla de entornos. | Requisito del brief: se condensa, no se elimina. |
| 2. Datos | 1.0 | 0.4 | Las cuatro decisiones de preparación en una línea cada una. | Detalle en la guía §33.1. |
| 3. Decisiones de modelado | 3.3 | 2.0 | Se conserva el diagrama (Figura 1). Se quita la tabla de pesos (los pesos ya están en el diagrama). 3.5 y 3.6 se unen en un párrafo. | Arquitectura, features y pérdida siguen completas. |
| 4. Escala Stuff+ | 1.3 | 0.8 | Las tres fórmulas sin cambio; propiedades en 4 viñetas. | Requisito del brief, sin cambios de fondo. |
| 5. Validación | 2.7 | 1.4 | Se conservan las tablas de AUC y modelo nulo; el diseño de las tres pruebas pasa a tres frases. La gráfica de estabilidad pasa a una frase con sus cifras; calibración en dos frases. | Las cifras de la gráfica se conservan en el texto. |
| 6. Efecto de la altitud | 3.3 | 1.9 | Tabla de formas → una frase con sus errores; se conserva la gráfica del Δ (Figura 2); niveles de evidencia, variante y Δ por pitcher en un párrafo cada uno; hipótesis con celdas cortas. | Criterio "integración arsenal + altitud": la gráfica es la evidencia central. |
| 7. Perfil, recomendaciones y dashboard | 5.0 | 2.3 | Tablas de recomendaciones y de pantallas → un párrafo cada una (la Figura 2 ya muestra Δ y nivel). Se quitan la "resiliencia a la altitud", "quién gana y quién pierde", la tabla de cuartiles, la tabla de ejemplos del simulador y las reglas de consistencia. **Se corrige la tasa de roletazos.** | Los ejemplos del simulador salen en el video; resiliencia y reglas de consistencia están en la guía §40. |
| 8. Alcance y fronteras | 1.5 | 0.8 | Mismos cuatro bloques; se unen puntos repetidos. | No se pierde ningún límite. |
| Referencias | 0.5 | 0.3 | Formato compacto; la línea final remite a la guía. | Sustituye al anexo. |
| **Total** | **21** | **~11** | | |

**Lo que deja de estar en el documento pero sigue en el repo:** pesos propios de la familia física y rama de valor de contacto (guía §33.6 y §34), gráfica de estabilidad y error de predicción fuera del par de ajuste (§35.2–35.3), formas aditiva común y Magnus de la transformación (§36.1), resiliencia ajustada por nivel (§40.2), regla de un estado por número y efecto neto Neutral/Mixed (§40.1 y §40.4), "quién gana y quién pierde" y celdas que ganan con un modelo y pierden con otro (§40.4), tabla de cuartiles de rectas (`24_arsenal_rectas.csv`) y ejemplos del simulador (§42.9; se muestran en el video).

## Cifras que cambian respecto al documento actual

Las demás cifras se copian tal cual del documento actual (cifras de la máquina de Dani).

| Dónde | Antes | Ahora | Fuente |
|---|---|---|---|
| Encabezado | Oct 5, 2026 | 22 de octubre de 2026 (fecha de entrega; ajustar si se envía antes) | — |
| §7.3, tasa de roletazos | Pitchers con sinker ≥ 30%: 41% de roletazos, contra 35–36% del resto | 51.9% contra 46.2% (2025) y 51.2% contra 46.4% (2026), sobre bolas en juego; por cuartil de % de rectas la tasa es plana (46.3% a 48.6%) | `24_arsenal_contraste.csv` y `24_arsenal_rectas.csv` (columna `gb_rate`); guía §42.5 y §42.10. La cifra anterior contaba los fouls como bolas en juego. |

---

# Stuff+ ajustado por altitud para Diablos Rojos — Documento técnico

22 de octubre de 2026 · Diablos Rojos

## Resumen

Construimos un sistema que califica cada lanzamiento de la LMB con tres escalas separadas (Stuff+, Location+ y Pitching+) y estima cuánto cambia su Stuff+ en la altitud del Harp Helú. Se entrenó con 635,002 lanzamientos de 1,134 pitchers (2024–2026), y todo resultado reportado es fuera de muestra.

| Pregunta | Resultado |
|---|---|
| ¿Generaliza? | AUC de whiff de Stuff+ 0.625 / 0.628 / 0.630 en pitcher, season y park holdout. |
| ¿Supera al nulo? | Pitching+ reduce 1.66% el error contra el valor medio por conteo × outs. |
| ¿Qué hace la altitud? | El mismo pitcher conserva 77% de su movimiento vertical y 71% del horizontal. |
| ¿Cuánto cuesta? | Four-Seam −5.4 y Sinker −2.9 de Stuff+, confirmados por lo observado. |
| ¿Qué es estable? | Stuff+ 0.77–0.78 de un año al siguiente; el valor observado, 0.22–0.26. |
| ¿Qué arsenal? | En swing-and-miss, el que depende menos de la recta; en valor real ese costo no se detecta. |

Límites de lectura: datos anónimos; "Extreme Altitude" es un proxy del Harp Helú que agrupa varios parques; el sistema estima calidad esperada y no prescribe qué pitch lanzar.

## 1. Marco teórico

**Sabermetría.** El resultado de un lanzamiento depende del bateador, del umpire y de la suerte; las métricas "plus" lo evalúan por lo que se espera que produzca. Seguimos las tres capas de FanGraphs: Stuff+ (física del pitch al salir de la mano), Location+ (ubicación dado el tipo, el conteo y el bateador) y Pitching+ (física, ubicación, contexto y entorno). En las tres, 100 es el promedio de la liga y 10 puntos, una desviación estándar.

**Física.** El movimiento proviene del efecto Magnus, una fuerza proporcional a la densidad del aire, que a 2,240 m es cerca de 20% menor que a nivel del mar. Alan Nathan estima que un pitch con 18 pulgadas de quiebre a nivel del mar se mueve 14–15 en Coors Field (1,609 m), y SABR documenta más carreras en estadios de altura. De ahí las hipótesis del reto: la altitud reduce el movimiento (H1), los pitches de diferencial de velocidad pierden menos (H2), los roletazos valen más (H3) y el spin rate solo no basta (H4).

**El entorno LMB.** Los datos no identifican estadios; usamos la categoría Extreme Altitude como proxy del Harp Helú.

| Categoría | Juegos | Lanzamientos | Carreras por juego |
|---|---|---|---|
| No Altitude | 1,004 | 293,875 | 9.05 |
| Medium Altitude | 509 | 154,840 | 12.52 |
| Extreme Altitude | 604 | 183,267 | 13.66 |

En altitud extrema se anota 51% más que a nivel del mar, y la categoría intermedia permite ver una relación dosis-respuesta.

## 2. Datos y preparación

635,002 lanzamientos, 1,134 pitchers y 2,127 juegos (2024–2026; 2026 hasta el 13 de junio), en siete tipos de pitch. Cuatro decisiones: (1) los zurdos (cerca de 24%) se reflejan a convención de diestro; (2) el UUID versión 1 de cada lanzamiento codifica su instante y ordena 99.95% de ellos dentro del juego; (3) con ese orden se reconstruyen 166,124 turnos (98.8% empieza en 0-0), lo que da el pitch anterior, el out en el turno y el valor en carreras; (4) el arsenal se mide contra la recta primaria calculada solo con temporadas anteriores. No hay corredores en base, local o visitante ni rol del pitcher.

## 3. Decisiones de modelado

### 3.1 Arquitectura

`[Figura 1 — se conserva la actual: "Cada familia ve información distinta y produce su propio score" (entradas → 9 clasificadores de boosting → combinación ponderada → Stuff+ / Location+ / Pitching+ con sus pesos).]`

El mismo procedimiento se aplica tres veces con distinta información de entrada. Pitching+ usa las variables crudas, no las salidas de Stuff+ y Location+, para no encadenar predicciones.

### 3.2 Features por familia

| Familia | Variables | Excluidas a propósito |
|---|---|---|
| Stuff+ | Velocidad, spin, eje, altura y lado de salida, extensión, movimiento vertical inducido (IVB) y horizontal (HB), tipo y mano; con historial, 7 de arsenal | Ubicación, conteo, altitud y los cuatro ángulos de salida y aproximación |
| Location+ | Altura y lado en el plato, bolas, strikes, lado del bateador, tipo y mano | Toda la física |
| Pitching+ | Las anteriores, más outs, entrada, pitch anterior y categoría de altitud | Los cuatro ángulos |

El arsenal son las diferencias de velocidad, IVB, HB, spin y extensión contra la recta primaria, la distancia entre puntos de salida y el uso; sin historial se usa la variante sin arsenal. Los ángulos se excluyen porque, con el movimiento, reconstruyen la ubicación en el plato (R² 0.995 en pitchers no vistos): con ellos el AUC de whiff "físico" era 0.727 y sin ellos 0.62.

### 3.3 Algoritmo y función de pérdida

Cada componente es un `HistGradientBoostingClassifier` de scikit-learn que minimiza la **pérdida logarítmica** (entropía cruzada binaria): tasa de aprendizaje 0.05, hasta 300 iteraciones y paro temprano con 10% de validación. El boosting capta interacciones (dos pulgadas de movimiento no valen lo mismo a 95 mph que a 88) y la pérdida logarítmica premia probabilidades calibradas. Cada componente se entrena donde su evento está definido (chase, fuera de la zona) y se aplica a todos los lanzamientos.

### 3.4 Valor en carreras y pesos

RE(b, s, o) es el promedio de carreras desde el conteo b-s con o outs hasta el fin de la media entrada (36 estados, solo LMB y solo historial). El valor de un pitch para el pitcher es

$$v = -\left(R + RE(b', s', o') - RE(b, s, o)\right)$$

con R las carreras que entran y el estado con prima el siguiente: en 0-0 sin outs RE es 0.93, un strike la lleva a 0.85 (v = +0.08) y una bola a 0.98 (v = −0.05). Pitching+ usa un RE por categoría de altitud; Stuff+ y Location+, el general.

Los pesos (Figura 1) salen de mínimos cuadrados no negativos con ridge leve de v sobre las probabilidades estandarizadas, con predicciones fuera de fold del historial. Stuff+ toma el valor por evento de Pitching+ sin el strike cantado, que con solo física mide qué tipo de pitch va a la zona. Así, el chase pesa 68% aunque la física lo predice mal por lanzamiento (AUC 0.547 contra 0.628 del whiff); el score se sostiene por pitcher (0.77–0.78 año a año, y en altitud extrema el quintil alto genera de 5.9 a 12.1 pp más chase real) y con pesos iguales el orden casi no cambia (correlación con el valor siguiente 0.12 contra 0.15). Una rama de valor de contacto no mejoró fuera de muestra (1.56% contra 1.66%) y se descartó.

## 4. Escala Stuff+: definición formal y normalización

Sea $p_k(x)$ la probabilidad del componente k para el lanzamiento x y H el historial. Tres pasos, iguales en las tres familias:

**Paso 1.** Estandarizar contra el historial ($s_k = -1$ en contacto fuerte y jonrón, para que más alto sea mejor):

$$z_k(x) = s_k \, \frac{p_k(x) - \mu_k^H}{\sigma_k^H}$$

**Paso 2.** Combinar con los pesos (no negativos, suman 1):

$$r(x) = \sum_k w_k \, z_k(x)$$

**Paso 3.** Llevar a la escala plus contra el historial:

$$\text{Stuff+}(x) = 100 + 10 \, \frac{r(x) - \mu_r^H}{\sigma_r^H}$$

- **Referencia histórica.** Para 2026, H es 2024–2025; los scores de 2026 salen con media 99.8 y desviación 10.0.
- **Una escala para todos los tipos,** sin recentrar: Splitter 112, Slider 106, Changeup 105, Cutter 102, Curveball 100, Four-Seam 96 y Sinker 93 (2026).
- **Neutral al entorno.** Stuff+ no ve la altitud: su diferencia entre entornos es el cambio físico del pitch.
- **Agregación.** Pitcher: promedio de sus lanzamientos; arsenal: tipos ponderados por uso.

## 5. Validación fuera de muestra

La arquitectura se congeló antes de validar; los fallos se reportan como hallazgos. **Pitcher holdout:** validación cruzada de 5 grupos por pitcher con las tres temporadas y variantes sin arsenal (las que recibe un prospecto). **Season holdout:** 2024 → 2025 y 2024–2025 → 2026 (scores oficiales). **Park holdout:** No + Medium → Extreme, con Pitching+ sin la variable de altitud.

| AUC por componente | Pitcher | Season 2026 | Park → Extreme |
|---|---|---|---|
| Stuff+ · whiff | 0.625 | 0.628 | 0.630 |
| Location+ · strike cantado | 0.879 | 0.874 | 0.878 |
| Location+ · chase | 0.841 | 0.841 | 0.840 |
| Pitching+ · whiff | 0.750 | 0.750 | 0.750 |
| Pitching+ · strike cantado | 0.884 | 0.878 | 0.881 |
| Pitching+ · chase | 0.850 | 0.852 | 0.850 |

El AUC casi no cambia: el modelo no depende de conocer al pitcher, la temporada ni el entorno (los pitchers nuevos de 2026 dan whiff de Stuff+ 0.620 contra 0.628).

| Reducción del error contra el nulo | Pitcher | Season 2025 / 2026 | Park → Extreme |
|---|---|---|---|
| Pitching+ | 1.66% ± 0.12% | 1.21% / 1.66% | 0.86% |
| Location+ | 1.74% ± 0.11% | 1.25% / 1.70% | 0.80% |
| Stuff+ | 0.01% ± 0.01% | 0.00% / 0.00% | 0.03% |

El nulo es el valor medio por conteo × outs, el "xRunValue promedio por conteo" del reto. Las mejoras son pequeñas pero consistentes: el valor de un lanzamiento es mayormente ruido, y la física no decide si un pitch es bola o strike, así que Stuff+ se valida por pitcher. Año a año (al menos 150 lanzamientos), Stuff+ correlaciona 0.78 y 0.77, Pitching+ 0.62 y 0.56, Location+ 0.60 y 0.55 y el valor observado 0.26 y 0.22. Para predecir el valor de la temporada siguiente, Stuff+ (0.17–0.20) no supera al valor observado (0.22–0.26), pero combinarlos da lo mejor (0.25–0.27). **Calibración:** pendientes de 0.94 a 1.09 en los componentes con peso; una recalibración aprendida con 2025 no mejora 2026 porque las tasas base cambian.

## 6. Efecto de la altitud

Separamos tres preguntas: cuánto cambia el movimiento, cuánto cuesta en Stuff+ y cuánto se encarece el contacto (esta, fuera del Stuff+).

### 6.1 Cambio físico del movimiento

Con el mismo pitcher, tipo y temporada en dos entornos (al menos 20 lanzamientos en cada uno), probamos cinco formas de transformar el movimiento, ajustadas con el historial y evaluadas en la temporada siguiente. La multiplicativa con un factor por eje (error de 1.16 pulgadas) supera a la aditiva por tipo (1.53) y a no transformar (3.37), y la regla fijada antes la prefiere a la multiplicativa por tipo (1.15 con 14 parámetros). El IVB se multiplica por 0.77 [0.76, 0.78] y el HB por 0.71 [0.70, 0.72]; en altitud media, 0.83 y 0.79. Velocidad, giro y salida no se transforman. La forma multiplicativa es la que predice el efecto Magnus, con magnitud coherente con la caída de densidad.

### 6.2 Efecto en Stuff+

Contrafactual: cada lanzamiento de 2026 hecho a nivel del mar se califica con el mismo Stuff+ antes y después de multiplicar su movimiento por esos factores; como Stuff+ no ve la altitud, la diferencia es solo de forma.

`[Figura 2 — se conserva la actual: "La pérdida de Stuff+ en altitud se concentra en las rectas y coincide con lo observado" (Δ modelado contra observado por tipo, 2026, con nivel de evidencia).]`

El contraste observado es independiente (Stuff+ real del mismo pitcher y tipo, altitud menos nivel del mar), y el chequeo inverso coincide en las rectas (Four-Seam −5.3, Sinker −3.0). **Robusto** exige intervalo sin cero y mismo signo en dos temporadas, chequeo inverso coherente, al menos 30 celdas y confirmación observada; **Condicional**, mismo signo sin confirmación; **Incierto**, pocas celdas o signo distinto.

El titular usa la variante de física pura. La variante con arsenal también penaliza que se encoja la separación contra la recta (Slider −1.6, Changeup −2.3), pérdida ausente en lo observado (+0.3 y −0.9). Esa elección se hizo después de ver el contraste, que también confirma el nivel Robusto; el argumento independiente es físico (el aire no encoge por sí solo esa separación), y H2 se reporta con ambas variantes. Por pitcher (2026, 369 con al menos 100 lanzamientos), el Δ medio es −2.6 y el orden casi no cambia (correlación de rangos 0.94): la altitud comprime ventajas, no reordena la liga.

### 6.3 Hipótesis del reto

| Hipótesis | Evidencia | Veredicto |
|---|---|---|
| H1. Menos movimiento | Se conserva 0.81 en altitud media y 0.74 en extrema, en los siete tipos | Soportada |
| H2. Diferencial de velocidad pierde menos | Changeup y Splitter −0.5 contra −2.8 de los tipos Magnus: +2.3 [+2.0, +2.7]; con arsenal, +0.8 | Soportada en términos relativos |
| H3. Roletazos valen más | Ventaja del roletazo sobre el elevado: +0.14 carreras a nivel del mar, +0.25 en extrema; jonrones por elevado +4.7 pp en el mismo pitcher | Soportada, descriptiva |
| H4. El spin solo no basta | Con el mismo spin, el movimiento por 1,000 rpm cae a 0.76; como predictor, spin y movimiento empatan (AUC 0.6205 y 0.6204) | Parcial |

## 7. Perfil ideal, recomendaciones y dashboard

### 7.1 Perfil físico ideal en altitud

Rango central (percentiles 25 a 75) del 20% de lanzamientos con mayor Stuff+ en altitud extrema; una característica entra si distingue al grupo en 2025 y repite el signo en 2026. Entre paréntesis, la mediana del tipo.

| Tipo | Perfil del grupo alto |
|---|---|
| Four-Seam | 92.8–95.1 mph (91.7); IVB 12.8–15.5 in (13.1); spin 2,258–2,459 rpm (2,308) |
| Sinker | 90.9–94.3 mph (91.1); IVB 4.2–9.0 in (8.3) |
| Slider | IVB −2.7 a 1.6 in (2.6); spin 2,345–2,700 rpm (2,409); 10–16 in menos de IVB que la recta |
| Changeup | 78.3–82.8 mph; IVB −1.8 a 3.0 in (4.7); 7–10 mph menos que la recta |
| Curveball | 78.3–81.8 mph (77.7); 9–11 mph menos que la recta |
| Cutter | IVB 3.6–8.1 in (7.6); spin 2,354–2,558 rpm (2,283) |

En altitud gana casi lo mismo que en el llano (el Splitter no tiene muestra). En altitud extrema 2026, el quintil alto de Stuff+ genera más whiff real que el bajo en seis tipos (de +2.7 pp en Sinker a +10.4 en Curveball). Sin usar el modelo, el 20% de pitchers con más whiff real de cada tipo tiene más Stuff+ proyectado en Harp Helú que el 20% con menos en los seis tipos (Four-Seam 96.5 contra 91.3).

### 7.2 Recomendaciones operativas

Al evaluar un pitch para CDMX: descontar 5.4 puntos de Stuff+ al Four-Seam y 2.9 al Sinker; ajustar parcialmente hasta 1.3 la Curveball y 0.6 el Changeup, sin usarlo para excluir; no descontar al Slider, pero revisar su separación contra la recta; no concluir en Cutter y Splitter. Ningún tipo gana Stuff+ de forma confiable; ganan en términos relativos los que dependen menos del efecto Magnus. **Riesgo de contacto:** en el mismo pitcher, en altitud extrema el contacto fuerte sube 2.7 pp y el jonrón por elevado 4.7 pp; Pitching+ estima 0.21 a 0.35 carreras más por 100 lanzamientos (un piso), menos en pitches de alta propensión a roletazo. **Prospecto:** su Stuff+ en Harp Helú es su Stuff+ a nivel del mar más el Δ físico de sus propios lanzamientos; los arsenales cargados de rectas son los más sensibles.

### 7.3 Arsenal con mayor Stuff+ en Harp Helú

A nivel del mar el porcentaje de rectas no se relaciona con el Stuff+ del arsenal; en Harp Helú, cada 10 puntos porcentuales más cuestan 0.56 puntos más de Δ [0.45, 0.68] en 2026. La composición más alta es Four-Seam + Cutter + Slider (103.5 [102.0, 105.1]) y la más baja Four-Seam + Sinker + Curveball + Changeup (95.8 [93.8, 97.7]), también la última en 2025. La mezcla explica 7% de las diferencias entre pitchers en Harp Helú: pesa más la calidad de cada pitch.

**Contraste con resultados reales.** El Stuff+ pondera sobre todo swing-and-miss y casi nada el roletazo. En el valor realizado por 100 pitches en altitud extrema el costo de las rectas no se detecta (0.00 [−0.17, 0.15] carreras por cada 10 pp en 2025; −0.15 [−0.43, 0.12] en 2026). Sobre bolas en juego, los pitchers con sinker en al menos 30% de su arsenal inducen 51.9% de roletazos contra 46.2% del resto en 2025, y 51.2% contra 46.4% en 2026; por cuartil de rectas la tasa es plana (46.3% a 48.6%). Es una lectura de swing-and-miss, no una recomendación de roster.

### 7.4 Dashboard: Pitching Decision System

Aplicación Streamlit que solo lee los resultados congelados (reproduce los scores oficiales con diferencia máxima de 1.4 × 10⁻¹⁴). Cinco pantallas: **Roster** (alertas para el cuerpo técnico), **Player** (ficha del pitcher y su movimiento en cada entorno), **Simulator** (el mismo pitch a nivel del mar, en altitud media y en Harp Helú, cambios de física y mapa de movimiento contra los pitches con más y menos whiff real), **Scouting** (perfil ideal, arsenal, explorador de la liga y prospectos) y **Game** (calidad esperada por conteo y bateador, con el costo del contacto). **Harp Helú Fit** es el percentil del Stuff+ proyectado en altitud, no de la pérdida, que premiaría a los peores. Las **alertas** marcan a quien pierde más de lo esperado para su tipo y nivel, con piso de 1 punto: en 2026, 38 rojas, 51 amarillas y 280 verdes de 369 pitchers. Toda lectura muestra su nivel de evidencia, y el sistema nunca afirma cuál es "el mejor pitch".

## 8. Alcance y fronteras del modelo

Sirve para comparar calidad esperada, descontar la altitud al evaluar candidatos y explorar escenarios físicos; no para prescribir qué pitch lanzar ni en qué proporción.

- **Datos.** "Extreme Altitude" agrupa 4 o 5 equipos locales, no solo el Harp Helú. Datos anónimos: equipos reconstruidos por co-aparición y rol inferido. Sin corredores en base (RE por conteo × outs), sin barrel ni xBA/xSLG. 2026 parcial y 190 a 202 pitchers en las pruebas año a año: diferencias de 0.05 en correlación no son distinguibles.
- **Modelo.** Stuff+ mide pitchers y tipos, no lanzamientos, y complementa al desempeño observado sin superarlo. El daño por contacto no entra a los scores; Pitching+ lo subestima en altitud extrema (−0.52 carreras por 100 lanzamientos predichas contra −1.01 observadas) y puede calificar alto el centro de la zona con pocos strikes. El arsenal aporta poco y el túnel solo se aproxima con la distancia entre salidas. La calibración deriva entre temporadas. Interpretación por ablación, sin SHAP. Fuga menor: los pesos se aprendieron con 2024–2025 y se usaron en toda la validación por pitcher.
- **Altitud.** Solo se transforma el movimiento, no la ubicación ni la adaptación del pitcher; un par de factores para todos los tipos; pocas comparaciones del mismo pitcher en Cutter y Splitter.
- **Uso.** Sin política causal de mezcla ni secuencia; los escenarios del simulador no garantizan que el pitcher los logre. Datos de rendimiento, sin uso en apuestas.

## Referencias

- Tango, T., Lichtman, M. y Dolphin, A. (2007). *The Book: Playing the Percentages in Baseball.* Potomac Books.
- SABR. High altitude offense. *Baseball Research Journal.*
- Nathan, A., y FanGraphs (Stuff+, Location+, Pitching+), citados en el documento del reto.
- Huang, G. y Hsu, M. (2021). Big data analytics in baseball: A review. *SAGE Open,* 11(4).
- McBride, J. (2026). Shapley value decomposition of pitching performance. *Games,* 17(2), 15.

**Código y resultados.** Scripts 01–24 del repositorio: 18 (scores), 19 (validación), 20–22 (altitud y perfil), 23 (dashboard) y 24 (arsenal). La guía metodológica del repositorio detalla lo resumido: pesos y rama de contacto (§33–34), validación (§35), altitud y perfil (§36 y §38), dashboard (§40) y correcciones previas a la entrega (§42).
