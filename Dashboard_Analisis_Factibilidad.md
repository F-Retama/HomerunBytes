# Análisis de factibilidad — Pitching Decision System vs. modelos actuales

Contraste del documento de Design Thinking y de los wireframes (Pitching Staff, Player Overview) contra lo que los modelos congelados (scripts 16–22, guía secciones 32–36) pueden sostener con evidencia.

**Leyenda:** ✅ se construye tal cual · ⚠️ se construye con una definición o un ajuste explícito · ❌ no lo sostiene el modelo (fuera de V1).

---

## 1. Hechos de los datos que cambian el diseño

| Supuesto del wireframe | Lo que dicen los datos | Decisión de producto |
|---|---|---|
| "Roster de Diablos" | El dataset es anónimo y no identifica equipos. Los equipos se reconstruyen por co-aparición en la misma mitad de entrada: salen 19 grupos y **4–5 son locales en Extreme Altitude**. | El roster es **configurable**: equipo reconstruido (anónimo, etiquetado por su entorno local), lista personalizada persistida (el club carga sus IDs reales) o liga completa. |
| "Extreme Altitude = Harp Helú" | Extreme agrupa varios parques locales (4–5 equipos locales). | Se mantiene como **proxy**, con tooltip explícito en todas las pantallas. |
| Home vs. Away | No existe local/visitante por pitcher; solo la categoría de altitud del juego. | "Home/Away Δ" se reemplaza por **Δ altitud**: el mismo pitch en Harp Helú contra nivel del mar (modelado) y, como secundario, el **Δ observado** (Extreme − No Altitude del mismo pitcher). |
| SP / RP | No viene en los datos. | Se **infiere** con pitches por aparición y la proporción de apariciones que empiezan en la 1.ª entrada (127 de 550 SP en 2026). Se etiqueta como inferido. |
| Trend · 6 semanas | No hay fecha, pero el UUID v1 del PitchUID trae timestamp (2026: 24-mar a 13-jun). | ✅ Tendencia semanal real (≥ 20 pitches por semana). |
| Historial por temporada | Scores fuera de muestra para 2024–2026 (CV por pitcher, guía 35). | ✅ Historial de 3 temporadas. Las temporadas oficiales 2025 y 2026 son season-forward. |

## 2. Definiciones necesarias (no estaban en el wireframe)

> **Regla transversal (guía 40):** cada número del dashboard pertenece a un estado explícito —No Altitude, Extreme Altitude (Harp Helú) o todos los parques— y nunca se mezclan dentro de una misma comparación.

Todas están implementadas en `23_dashboard_data.py`, con umbrales explícitos:

- **Harp Helú Fit (pitcher):** calidad proyectada en Harp Helú *(redefinido en la auditoría 2, guía 38)*. Es el percentil del Stuff+ proyectado en Harp Helú (Stuff+ oficial a nivel del mar + Δ físico), ponderado por uso, entre pitchers con ≥ 100 pitches: Strong ≥ p75, Moderate, Low, Poor < p25. La definición original (percentil del Δ) premiaba a pitchers con poco Stuff+, porque los mejores pierden más puntos.
- **Fit por pitch:** el mismo criterio, comparando contra pitches del mismo tipo en la liga.
- **Confianza:** la de cada tipo sale de los niveles de evidencia de la guía 36. La del pitcher es Robust si ≥ 60% de sus pitches son de tipos robustos y tiene ≥ 300 pitches; Uncertain si < 150 pitches o ≥ 40% en tipos inciertos.
- **Semáforo (acción por pitch):**
  - Adjust: el pitch pierde más de lo esperado para su nivel y tipo (exceso en el 10% inferior de su tipo) y el tipo tiene evidencia Robusta.
  - Review: exceso en el 25% inferior de su tipo.
  - Maintain: el resto.

  Son prioridades de revisión, no instrucciones de uso.
- **Alerta del pitcher:** roja si un pitch con ≥ 15% de uso es Adjust; amarilla si un pitch con ≥ 30% de uso no es Maintain.

## 3. Pantalla por pantalla

### Pantalla 1 — Pitching Staff (Roster)

| Elemento | Estado | Fuente / nota |
|---|---|---|
| KPIs (tamaño, Stuff+ medio, distribución de Fit, alertas) | ✅ | 18 (scores oficiales), 22 (proyección) |
| Recommendation Inbox | ⚠️ | Alertas por pitcher (su pitch más afectado) y "oportunidades": pitches con Stuff+ ≥ 105 que pierden < 1.5 en altitud. **Se quitó "margen para más uso"**: el modelo no sostiene recomendaciones de pitch mix (sin política causal; el mismo documento de diseño lo excluye de V1). |
| Harp Helú Roster Map (Stuff+ × adaptación) | ✅ | x = Stuff+; y = percentil de "pierde menos / más de lo esperado" para su nivel y tipo (guía 38) |
| Tabla: Stuff+, Location+, Fit, Δ, alerta, tendencia | ✅ | Se agrega Pitching+ y el Δ observado como secundario |
| Comparar (selección múltiple) | ✅ | Comparador rápido de 2 a 4 pitchers |
| Filtros (rol, mano, acción, fit) | ✅ | Rol inferido |

### Pantalla 2 — Player Overview

| Elemento | Estado | Fuente / nota |
|---|---|---|
| Stuff+, Location+, Pitching+, Fit | ✅ | 18, 23 |
| Current Recommendation / Strength / Risk | ⚠️ | Se generan con reglas a partir del Δ y la evidencia. **El texto del wireframe "menos Four-Seam arriba de la zona"** es una recomendación de ubicación y uso que el modelo no sostiene causalmente; se reformula como lectura de calidad ("pierde X de Stuff+; revisar en el Simulator"). |
| Arsenal Overview (uso, Stuff+, Location+, Δ, Fit, acción, confianza) | ✅ | 18, 22, 23 |
| Arsenal Fingerprint (movimiento No vs. Extreme) | ✅ | Observado si hay ≥ 20 pitches por entorno; si no, proyectado con la transformación validada (IVB × 0.77, HB × 0.71) |
| Harp Helú Adaptation (Stuff+, IVB, HB, outcomes) | ✅ | Outcomes = whiff y chase **modelados** con el componente físico de Stuff+ a nivel del mar contra Harp Helú |
| "Net effect: depende de cómo se pondere whiff vs. groundball" | ❌ → reemplazado | Esa lectura era de la arquitectura V1 (pesos iguales), invalidada en la guía 32–36. El neto ahora es el Δ Stuff+, más el nivel de evidencia y el contraste observado. Regla vigente (guía 40): Neutral si \|Δ\| < 1; Mixed si \|Δ\| ≥ 1 pero el modelo de arsenal da el signo contrario, o whiff o chase van en contra; Favorable / Unfavorable en el resto. |
| Why? | ✅ | Cambio físico, componentes, contraste observado y nivel de evidencia |
| Development Targets | ⚠️ | Zonas P25–P75 de los pitches de Stuff+ alto en Extreme (perfil de 22). "Release consistency" se quitó: no hay modelo que la ligue a resultados. Siempre exploratorios. |

**Valores demo del wireframe que contradicen la evidencia actual** (se reemplazan por datos reales):

- "Sinker favorable en altitud": en realidad el Sinker pierde 2.8 de Stuff+ (Robusto).
- "Slider: evidencia Robust": en realidad es **Conditional** (pérdida modelada no confirmada en lo observado).
- "Groundball mejora en altitud": es cierto (+5 pp del mismo pitcher), pero no entra al Stuff+ (guía 34).

### Pantallas siguientes (fases)

| Pantalla | Estado | Base técnica |
|---|---|---|
| Simulator — Altitude | ✅ | `simulate_pitch()` (22): mismo pitch a nivel del mar contra Harp Helú |
| Simulator — What-if (velocidad, IVB, HB, spin, release) | ✅ | Mismo motor; solo características físicas |
| Simulator — What-if de ubicación | ⚠️ | Location+ (18) evalúa ubicación × count × lado del bateador; se muestra como calidad esperada, no como "mejor pitch" |
| Simulator — What-if de uso / pitch mix | ❌ | Sin política causal (V2) |
| Saved Experiments / Shortlists / Historial | ✅ | Persistencia local (JSON/SQLite) |
| Scouting — Ideal Profile, Compare, Shortlist | ✅ | 22 (perfiles validados contra whiff y chase reales), 23 |
| Scouting — Undervalued | ⚠️ | Stuff+ alto con valor observado bajo; exploratorio (guía 35: Stuff+ complementa al observado, no lo reemplaza) |
| Game — heatmaps de Location+ por count y lado | ✅ | Modelo de Location+ (18) sobre una rejilla de ubicaciones |
| Recomendador de secuencias | ❌ | Fuera de V1 (lo excluye el propio documento de diseño) |

## 4. Plan de integración

1. **Fase 1 (esta entrega):** capa de datos (`23_dashboard_data.py`), app Streamlit con el sistema visual de los wireframes y las pantallas 1 (Roster) y 2 (Player Overview) con datos reales.
2. **Fase 2:** Simulator (Altitude + What-if) con experimentos guardados.
3. **Fase 3:** Scouting (perfil ideal, comparador, shortlist, proyección a Harp Helú).
4. **Fase 4:** Game (heatmaps de Location+), pulido de UX y pruebas con usuarios.

Ejecución: `python 23_dashboard_data.py` y después `streamlit run dashboard/app.py`, ambos desde la raíz del repo.
