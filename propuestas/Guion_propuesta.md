# Propuesta: guion del video con holgura (4:32 de 5:00)

Propuesta de ajuste al guion del video (versión del 14/10/2026), hecha sobre el documento técnico recortado (`propuestas/Doc_Tecnico_propuesta.md`). Objetivos:

- **Holgura.** El guion actual dura exactamente 5:00. Esta versión tiene **523 palabras de narración** (~4:00 a 130 palabras por minuto, el ritmo del guion actual: ~610 palabras en 5:00), más ~30 s de pausas para los clics: **4:32 en total, con 28 s de margen**.
- **Simulator más corto:** de 95 s y cinco acciones a 55 s y tres.
- **Una sola demostración de prospecto:** se queda la captura manual en el Simulator y sale Scouting › Evaluar prospecto.
- **Cifras verificadas** contra el dashboard con `main` actualizado (tras el PR #8). Cada afirmación coincide con el documento recortado.

## Qué cambia respecto al guion actual

| Escena | Antes | Ahora | Por qué |
|---|---|---|---|
| Apertura | 0:30 | 0:40 | Misma narración, con una frase de física ("con menos aire, el mismo giro mueve menos la pelota"); Location+ queda presentado aquí, porque sale la pestaña Ubicación. |
| Roster | 0:30, "pierde 7.2 puntos" | 0:30, "pierde más de 7 puntos" | En `main` la cifra es 7.28 en Linux (se muestra 7.3) y 7.2 en la Mac de Dani: la redacción vale en ambas. |
| Player | 0:35, "coinciden: 5.4 y 5.4" | 0:35, "el modelo coincide con lo observado" | En Linux ambas son 5.5 y en la Mac 5.4. El 5.4 de la liga se conserva porque es la cifra del documento. |
| Simulator | 1:35, cinco acciones | 0:55, tres acciones | Se quita la pestaña Ubicación (Location+ 107.6 / 100.1) y "+1 mph" pasa a "casi 4 puntos más" (Linux 89.2, Mac 89.3). Quedan: el mismo pitch en tres entornos, el what-if y la captura de un prospecto con el mapa. |
| Scouting | 0:40, mapa + Evaluar prospecto | 0:37, solo el mapa | Una sola demostración de prospecto. Se agrega la cifra no circular del documento (96.5 contra 91.3). |
| Game | 0:35, "mejor calidad arriba de la zona" | 0:37, "en la parte alta de la zona" | El máximo de Pitching+ está a 3.3 ft, dentro de la zona (que llega a 3.5 ft): "arriba de la zona" sugiere fuera de ella. Las cifras 4.2 y 2.1 se confirman. |
| Cierre | 0:35 | 0:38 | Sin cambios de contenido; "promedio por conteo y outs" en lugar de "por conteo", como el documento. |

## Requisitos del entregable 4

| Requisito del reto | Dónde se cumple |
|---|---|
| Ingresar las características físicas de un pitch y obtener su Stuff+ | Escena 4: Simulator › Pitch manual, Four-Seam de 95 mph e IVB de 18 in → 109.5 |
| El mismo pitch en Harp Helú contra nivel del mar | Escena 4: tres tarjetas del 00964 (92.9 / 87.4 / 85.6) y el prospecto (109.5 / 101.4) |
| Mapa de movimiento de los más y menos efectivos | Escena 4 (fondo del mapa del Simulator) y escena 5 (Scouting › Perfil ideal, efectividad por whiff real) |

## Guion escena por escena

| # | Tiempo | Pantalla | En pantalla | Narración | Palabras |
|---|---|---|---|---|---|
| 1 | 0:00–0:40 | Roster | Abrir el dashboard en el Roster, sin moverse. | El Harp Helú está a 2,240 metros. Con menos aire, el mismo giro mueve menos la pelota: Alan Nathan estima que en Coors Field un pitch con 18 pulgadas de quiebre se mueve solo 14 o 15. Lo medimos con datos de la Liga Mexicana: en altitud extrema, el mismo pitcher conserva 77% de su movimiento vertical. Construimos un sistema que califica cada lanzamiento con Stuff+, Location+ y Pitching+, y proyecta cuánto cambia en altitud extrema, nuestro proxy del Harp Helú. | 81 |
| 2 | 0:40–1:10 | Roster | Señalar el KPI de Stuff+ y la primera alerta de la bandeja. | Este es un equipo local en altitud. Su Stuff+ comparable es 102.2; con lo que lanzaron en sus parques se vería 99.5, porque la altitud les baja el número. La bandeja responde a quién revisar primero: el Four-Seam del Pitcher 00964 pierde más de 7 puntos en Harp Helú, 3.3 más de lo normal para una recta de su nivel. | 60 |
| 3 | 1:10–1:45 | Player | Abrir al Pitcher 00964 desde la alerta (Review →); mostrar la tarjeta de adaptación del Four-Seam y la del Slider. | Su ficha explica la alerta. El mismo Four-Seam pasa de 93 a 86: el aire le quita 4 pulgadas de movimiento vertical y su whiff modelado cae 3.7 puntos. No todos los pitches pierden igual. En la liga, el Four-Seam pierde 5.4 puntos, el Sinker 2.9 y el Changeup 0.6; en el Slider el efecto es prácticamente nulo. En las rectas, el modelo coincide con lo observado en los mismos pitchers. | 71 |
| 4 | 1:45–2:40 | Simulator | (a) Mismo pitcher y pitch: señalar las tres tarjetas. (b) Mover velocidad de 94.0 a 95.0 mph y regresar con "Restablecer". (c) Cambiar a Pitch manual (Four-Seam, RHP), capturar 95.0 mph e IVB 18.0 y señalar el mapa con la liga. | Este es el simulador: el mismo pitch a nivel del mar, en altitud media y en Harp Helú, 92.9, 87.4 y 85.6. Si lograra una milla más de velocidad, el modelo estima casi 4 puntos más en Harp Helú. Para un prospecto, el coach captura sus métricas: un Four-Seam de 95 millas y 18 pulgadas de movimiento vertical da 109.5 a nivel del mar y 101.4 en Harp Helú. El mapa lo ubica frente a los Four-Seam de la liga con más y menos swing en falso real. | 88 |
| 5 | 2:40–3:17 | Scouting | Perfil ideal › Four-Seam; mapa con "Efectividad según: Whiff real" y movimiento en Harp Helú; señalar la tabla comparativa. | Para el director deportivo, el perfil ideal muestra ese mapa completo. Los Four-Seam con más whiff real superan a los de menos, 15.2% contra 4.1%, y los separa la velocidad, no el movimiento. El grupo con más whiff también tiene más Stuff+ proyectado en Harp Helú, 96.5 contra 91.3: el modelo coincide con resultados que no usa. Esa es la pregunta de reclutamiento: cuánto vale un pitcher aquí, no en otro parque. | 72 |
| 6 | 3:17–3:54 | Game | Pitcher 00776, Four-Seam, cuenta 0-2 contra bateador derecho, entorno Harp Helú; después cambiar a nivel del mar. | En el juego, la vista táctica muestra calidad y riesgo juntos. Con cuenta de 0-2 contra un derecho, la recta del Pitcher 00776 tiene su mayor calidad esperada en la parte alta de la zona. Pero en Harp Helú el contacto dentro de la zona cuesta el doble que a nivel del mar: 4.2 contra 2.1 carreras por cada 100 pitches. En altitud, dejarla en la zona sale el doble de caro. | 72 |
| 7 | 3:54–4:32 | Roster | Regresar al Roster. | Todo se validó fuera de muestra. El AUC de whiff casi no cambia con pitchers que el modelo no vio, con la temporada siguiente y con un entorno de altitud que no vio: 0.625, 0.628 y 0.630. Y Pitching+ predice el valor de cada lanzamiento mejor que el promedio por conteo y outs. Cada lectura muestra su nivel de evidencia. El sistema no dice qué pitch lanzar: dice cuánto vale cada pitch en el Harp Helú y por qué. | 79 |
| | **4:32** | | | | **523** |

Tiempo por escena = narración a 130 palabras por minuto más pausas para las acciones; la más larga es la captura manual del Simulator (~14 s de clics).

## Cifras dichas en voz alta, verificadas

Verificadas el 16/10/2026 en Linux con `main` en `ad0af9d` (PR #8 fusionado), tablas del dashboard regeneradas (`correr_prototipo.py --sin-abrir`: 23 y 24 regenerados; el motor reproduce los scores oficiales con 1.4e-14) y `propuestas/verificar_cifras_guion.py`, que replica el cálculo de cada vista. Las cifras de la Mac de Dani difieren en centésimas; cuando eso cambia el redondeo, la narración usa una forma que vale en ambas.

| Escena | Cifra dicha | Valor en `main` (Linux) | Fuente | ¿Cambió respecto al guion actual? | En el documento recortado |
|---|---|---|---|---|---|
| 1 | 2,240 m; 18 → 14–15 in (Nathan) | — | Documento del reto | No | §1 |
| 1 | 77% del movimiento vertical | IVB × 0.769 | `20_transform_params.csv`; guía §36.1 | No | Resumen y §6.1 |
| 2 | Stuff+ comparable 102.2; temporada tal cual 99.5 | 102.16 / 99.48 | Roster, KPI "Average Stuff+ · nivel del mar" (Equipo Extreme-1, 2026; `pitchers.parquet`) | No | — (cifra de pantalla) |
| 2 | 00964 Four-Seam pierde "más de 7"; 3.3 más de lo esperado | −7.28 (se muestra 7.3); exceso −3.28 | Roster › Inbox, `inbox.parquet` | **Sí: antes "7.2"** (cifra de la Mac; en Linux redondea a 7.3) | §7.4 (regla de alertas) |
| 3 | Four-Seam del 00964: 93 → 86 | 92.89 → 85.61 | Player, tarjeta Harp Helú Adaptation (`arsenal.parquet`) | No | — |
| 3 | 4 pulgadas menos de IVB | 17.82 → 13.70 in (−4.1) | Player, tarjeta de adaptación (`fingerprint.parquet`) | No | — |
| 3 | Whiff modelado −3.7 puntos | 8.91% → 5.18% (−3.73 pp) | Player, Outcomes (`arsenal.parquet`) | No | — |
| 3 | Liga: Four-Seam −5.4, Sinker −2.9, Changeup −0.6, Slider ≈ 0 | −5.49, −2.88, −0.59, −0.17 | `21_evidence_levels.csv` (variante física) | No; Linux da −5.5 en el Four-Seam, se conserva el 5.4 del documento (Mac) | Figura 2 y §7.2 |
| 3 | "El modelo coincide con lo observado" (rectas) | Four-Seam −5.49 / −5.46; Sinker −2.88 / −2.48 | `21_evidence_levels.csv` | **Sí: antes "5.4 y 5.4"**; sin cifra para no depender del redondeo | §6.2 |
| 4 | 92.9, 87.4 y 85.6 | 92.89 / 87.37 / 85.61 | Simulator, tarjetas (`Engine.population_table`) | No | — |
| 4 | +1 mph: "casi 4 puntos más en Harp Helú" | 85.61 → 89.20 (+3.6) | Simulator, what-if | **Sí: antes "de 85.6 a 89.3"** (Mac; en Linux 89.2) | §7.4 (cómo calcula el simulador) |
| 4 | Prospecto Four-Seam 95 mph e IVB 18 in (RHP): 109.5 y 101.4 | 109.48 / 101.37; dentro del rango observado | Simulator › Pitch manual | No | §7.2 (cómo perfilar un prospecto) |
| 5 | Four-Seam con más y menos whiff real: 15.2% contra 4.1% | 15.2 / 4.1 (53 y 53 de 264) | Scouting › Perfil ideal, tabla del mapa (`data.movement_map`) | No | — |
| 5 | Los separa la velocidad, no el movimiento | 93.2 vs 91.4 mph; IVB en Harp Helú 13.2 vs 12.5 in | Misma tabla | No | — |
| 5 | Stuff+ proyectado 96.5 contra 91.3 | 96.5 / 91.3 | Misma tabla; guía §42.2 | **Nueva** (estaba en el documento, no en el guion) | §7.1 |
| 6 | Mayor calidad "en la parte alta de la zona" | Pitching+ máximo 104.1 en z = 3.31 ft, x = −0.50 (zona hasta 3.5 ft) | Game (`Engine.game_grid`) | **Sí: antes "arriba de la zona"** | — |
| 6 | Costo del contacto en la zona: 4.2 contra 2.1 carreras por 100 | 4.16 / 2.06 | Game, lectura "Costo del contacto (zona)" | No (se confirma) | §7.2 (riesgo de contacto) |
| 7 | AUC de whiff 0.625 / 0.628 / 0.630 | 0.6249 (media de 5 grupos) / 0.6278 / 0.630 | `19_cv_component_metrics.csv`, `18_component_metrics.csv`, guía §35.2 | No | Resumen y §5 |
| 7 | Pitching+ mejor que el promedio por conteo y outs | −1.66% de error (2026) | `18_out_of_sample_value.csv`; guía §35.2 | Redacción ("y outs") | §5 |

**Retiradas del guion** (ya no se dicen): Location+ 107.6 arriba y 100.1 abajo (pestaña Ubicación) y el prospecto de Scouting de 103.5 a 100.6, percentil 66 (Evaluar prospecto).

## Preparación antes de grabar

- Correr `python correr_prototipo.py` con la versión final de `main` y confirmar que la verificación del motor da una diferencia cercana a 1e-14.
- Correr `python propuestas/verificar_cifras_guion.py` en la máquina donde se graba y comparar con la tabla de arriba. Si alguna cifra cambia de redondeo, ajustar la narración antes de la toma.
- Abrir una vez cada pantalla del video para que los cálculos queden en caché: 00964 Four-Seam en Player y Simulator; Simulator › Pitch manual con Four-Seam RHP; Scouting › Perfil ideal › Four-Seam; 00776 en Game con 0-2 contra derecho, en Harp Helú y a nivel del mar.
- Roster por defecto en "Equipo Extreme-1" y temporada 2026.
- Navegador a pantalla completa, zoom al 100% o 110%, tema oscuro del dashboard, sin otras pestañas ni notificaciones.
- Ensayar el cambio a Pitch manual y la captura de 95 mph y 18 in: es el paso con más clics y el que más tiempo toma (≈ 14 s previstos).
- Grabar la voz aparte y montarla sobre la pantalla, o grabar por escena. Ya no hace falta partir el Simulator en dos tomas.
- Exportar a 1080p, verificar que dure 5:00 o menos (objetivo 4:32) y subirlo como enlace.

## Lo que no se dice en el video

Sin cambios respecto al guion actual.

| No decir | Por qué | Decir en su lugar |
|---|---|---|
| "Este es el mejor pitch" o "tira aquí" | El modelo estima calidad esperada, no el efecto causal de elegir un pitch. | "Aquí tiene su mayor calidad esperada" |
| "El arsenal ideal tiene menos rectas" | El Stuff+ pondera sobre todo swing-and-miss; en el valor real en altitud el porcentaje de rectas no hace diferencia y los sinkerballers inducen más roletazos (51–52% contra 46% de las bolas en juego). | No mostrar Arsenal ideal en el video, o decir "en swing-and-miss" |
| "Esto es el Harp Helú" | "Extreme Altitude" agrupa varios parques de altitud, no solo el Harp Helú. | "Altitud extrema, nuestro proxy del Harp Helú" |
| "Con este ajuste el pitcher ganará X" | El simulador cambia la física dentro del modelo; no garantiza que el pitcher pueda lograrlo. | "Si lograra este cambio, el modelo estima…" |
