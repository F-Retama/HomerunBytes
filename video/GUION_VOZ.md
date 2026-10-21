# Grabar la voz del video

El video se arma con **un audio por segmento** (20 en total). No hace falta que coincidan con ningún tiempo: el grabador
recorre el dashboard y espera a que termine cada audio antes de pasar a la siguiente acción, así que el video se ajusta a
tu ritmo.

**Único límite:** que la suma de los 20 audios no pase de **3:40** (220 s). Los clics y las transiciones agregan ~75 s y
el video debe durar 5:00 o menos. Referencia: la voz sintética suma 3:36 (columna "Ref."), así que a un ritmo de
conversación tranquilo cabe sin problema.

## Cómo grabar

- Un archivo por segmento, con el nombre de la columna **Archivo** (`1a.m4a`, `2a.m4a`…). Sirve m4a (notas de voz del
  iPhone o QuickTime), wav o mp3.
- Lugar silencioso, el micrófono siempre a la misma distancia y el mismo tono en todos los segmentos.
- Puedes dejar silencio al inicio y al final: se recorta solo. Si te equivocas, vuelve a grabar ese segmento completo.
- Pon los 20 archivos en `video/voz/` y avísame. Lo que se corre es:

```bash
python3 video/grabar_video.py --voz-propia video/voz
```

## Segmentos

| Archivo | Pantalla | Qué se ve mientras hablas | Texto | Ref. |
|---|---|---|---|---|
| `1a` | Roster | Roster quieto; al final se resalta "Extreme Altitude" | El Harp Helú está a 2,240 metros. Con menos aire, el mismo giro mueve menos la pelota: Alan Nathan estima que en Coors Field un pitch con 18 pulgadas de quiebre se mueve solo 14 o 15. Lo medimos con datos de la Liga Mexicana: en altitud extrema, el mismo pitcher conserva 77% de su movimiento vertical. Construimos un sistema que califica cada lanzamiento con Stuff+, Location+ y Pitching+, y proyecta cuánto cambia en altitud extrema, nuestro proxy del Harp Helú. | 35 s |
| `2a` | Roster | KPI de Stuff+ resaltado | Este es un equipo local en altitud. Su Stuff+ comparable es 102.2; con lo que lanzaron en sus parques se vería 99.5, porque la altitud les baja el número. | 13 s |
| `2b` | Roster | Primera alerta de la bandeja resaltada | La bandeja responde a quién revisar primero: el Four-Seam del Pitcher 00964 pierde más de 7 puntos en Harp Helú, 3.3 más de lo normal para una recta de su nivel. | 12 s |
| `3a` | Player | (antes: clic en Review →) Recomendación resaltada | Su ficha explica la alerta. | 2 s |
| `3b` | Player | (antes: baja a Harp Helú Adaptation) Tarjeta del Four-Seam | El mismo Four-Seam pasa de 93 a 86: el aire le quita 4 pulgadas de movimiento vertical y su whiff modelado cae 3.7 puntos. | 10 s |
| `3c` | Player | Tarjeta del Slider | No todos los pitches pierden igual. En la liga, el Four-Seam pierde 5.4 puntos, el Sinker 2.9 y el Changeup 0.6; en el Slider el efecto es prácticamente nulo. | 13 s |
| `3d` | Player | Tarjeta del Four-Seam | En las rectas, el modelo coincide con lo observado en los mismos pitchers. | 5 s |
| `4a` | Simulator | (antes: abre el Simulator) Las tres tarjetas | Este es el simulador: el mismo pitch a nivel del mar, en altitud media y en Harp Helú, 92.9, 87.4 y 85.6. | 11 s |
| `4b` | Simulator | (antes: velocidad 94 → 95) Tarjeta de Harp Helú | Si lograra una milla más de velocidad, el modelo estima casi 4 puntos más en Harp Helú. | 6 s |
| `4c` | Simulator | (antes: Restablecer y Pitch manual) Selector de origen | Para un prospecto, el coach captura sus métricas: | 4 s |
| `4d` | Simulator | (antes: 95 mph e IVB 18) Controles y tarjetas | un Four-Seam de 95 millas y 18 pulgadas de movimiento vertical da 109.5 a nivel del mar y 101.4 en Harp Helú. | 10 s |
| `4e` | Simulator | (antes: baja al mapa) Mapa resaltado | El mapa lo ubica frente a los Four-Seam de la liga con más y menos swing en falso real. | 5 s |
| `5a` | Scouting | (antes: abre Scouting y baja al mapa) Mapa | Para el director deportivo, el perfil ideal muestra ese mapa completo. | 5 s |
| `5b` | Scouting | Tabla comparativa | Los Four-Seam con más whiff real superan a los de menos, 15.2% contra 4.1%, y los separa la velocidad, no el movimiento. | 10 s |
| `5c` | Scouting | Tabla comparativa | El grupo con más whiff también tiene más Stuff+ proyectado en Harp Helú, 96.5 contra 91.3: el modelo coincide con resultados que no usa. | 11 s |
| `5d` | Scouting | Mapa | Esa es la pregunta de reclutamiento: cuánto vale un pitcher aquí, no en otro parque. | 6 s |
| `6a` | Game | (antes: Pitcher 00776, cuenta 0-2) Contexto, tarjeta y mapa de calor | En el juego, la vista táctica muestra calidad y riesgo juntos. Con cuenta de 0-2 contra un derecho, la recta del Pitcher 00776 tiene su mayor calidad esperada en la parte alta de la zona. | 14 s |
| `6b` | Game | Panel de Lectura (4.2 y 2.1) | Pero en Harp Helú el contacto dentro de la zona cuesta el doble que a nivel del mar: 4.2 contra 2.1 carreras por cada 100 pitches. | 10 s |
| `6c` | Game | (antes: cambia a Nivel del mar) Panel de Lectura | En altitud, dejarla en la zona sale el doble de caro. | 4 s |
| `7a` | Roster | (antes: regresa al Roster) Roster quieto | Todo se validó fuera de muestra. El AUC de whiff casi no cambia con pitchers que el modelo no vio, con la temporada siguiente y con un entorno de altitud que no vio: 0.625, 0.628 y 0.630. Y Pitching+ predice el valor de cada lanzamiento mejor que el promedio por conteo y outs. Cada lectura muestra su nivel de evidencia. El sistema no dice qué pitch lanzar: dice cuánto vale cada pitch en el Harp Helú y por qué. | 32 s |

Los textos son los de `propuestas/Guion_propuesta.md` partidos en segmentos; las cifras se verificaron en esta Mac el
19/10/2026. Pronunciación: "Pitcher 00964" se lee "cero cero nueve seis cuatro"; "0-2" se lee "cero y dos".
