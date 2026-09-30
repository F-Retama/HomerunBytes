# Design Thinking — Dashboard / Pitching Decision System para Harp Helú

## 1. Objetivo

Diseñar un producto que transforme la complejidad de **Stuff+**, **Location+**, **Pitching+**, arsenal y altitud en decisiones claras para el cuerpo técnico y la dirección deportiva de Diablos Rojos.

La intención no es construir únicamente un dashboard de métricas, sino un **sistema de apoyo a decisiones de pitcheo**.

---

## 2. Principio central de diseño

> **Diseña para el usuario. Presenta para el jurado.**

El producto debe ser útil para el sponsor y sus usuarios reales; la presentación debe demostrar al jurado que las recomendaciones están sustentadas por datos, validación y metodología.

También se adopta la lógica del taller:

- Primero entender al usuario.
- Después definir su problema real.
- Convertir ese entendimiento en una propuesta de valor.
- Idear sin filtrar prematuramente.
- Priorizar con criterios de deseabilidad, factibilidad, viabilidad, responsabilidad y diferenciación.
- Solo después diseñar pantallas.

---

## 3. How Might We

> **¿Cómo podríamos transformar la complejidad de Stuff+, ubicación, arsenal y altitud en decisiones claras sobre cómo desarrollar, utilizar y seleccionar pitchers para jugar en Harp Helú?**

Esta formulación cubre tres grandes tipos de decisiones:

- **Desarrollar** → coach de pitcheo.
- **Utilizar** → coach / catcher / cuerpo técnico.
- **Seleccionar** → director deportivo / scouting.

---

## 4. Usuarios principales


### Tabla resumen de empatía

| Usuario | Contexto | Job to Be Done | Dolor principal | Ganancia que busca |
|---|---|---|---|---|
| **Coach de pitcheo** | Desarrollo, preparación y estrategia del pitcher | **Entender cómo usar y desarrollar el arsenal de un pitcher para que funcione mejor en Harp Helú** | Tiene mucha información física/estadística, pero convertirla en un ajuste concreto no es inmediato | Saber **qué funciona, qué está fallando y qué podría modificar** |
| **Director deportivo / scouting** | Reclutamiento, prospectos, agentes libres y construcción de roster | **Identificar pitchers y perfiles de arsenal que tengan buen fit con CDMX** | Un pitcher que funciona bien en otro entorno puede no trasladar igual a la altitud | Saber **a quién buscar y por qué su perfil encaja con Harp Helú** |
| **Catcher / decisión táctica** | Preparación del juego y decisiones situacionales | **Evaluar rápidamente qué partes del arsenal y qué ubicaciones presentan mejores condiciones en una situación específica** | No puede procesar demasiadas métricas durante una decisión | Recibir una **lectura rápida, contextual y accionable** |

### Usuario 1 — Coach de pitcheo

**Contexto:** desarrollo, preparación y estrategia del pitcher.

**Job to Be Done:**
> Entender cómo usar y desarrollar el arsenal de un pitcher para que funcione mejor en Harp Helú.

**Dolores principales:**
- Mucha información física y estadística.
- Difícil convertir métricas en ajustes concretos.
- Los scores aislados no explican qué hacer.
- Necesidad de entender qué cambia con la altitud.
- Necesidad de probar escenarios antes de llevarlos al campo.

**Ganancias buscadas:**
- Saber qué funciona.
- Saber qué está fallando.
- Entender por qué.
- Identificar qué podría modificarse.
- Simular escenarios.
- Guardar experimentos.
- Dar seguimiento histórico.

---

### Usuario 2 — Director deportivo / scouting

**Contexto:** reclutamiento, prospectos, agentes libres, construcción de roster.

**Job to Be Done:**
> Identificar pitchers y perfiles de arsenal que tengan buen fit con CDMX, no solo buenos resultados en otros entornos.

**Dolores principales:**
- Un buen pitcher en otro estadio puede no trasladar igual a Harp Helú.
- Un Stuff+ alto no necesariamente responde la pregunta de adaptación.
- Comparar candidatos con muchas variables es lento.
- Necesita distinguir desempeño observado de características transferibles.

**Ganancias buscadas:**
- Saber qué perfiles buscar.
- Comparar candidatos.
- Identificar pitchers potencialmente infravalorados.
- Ver estabilidad histórica.
- Entender riesgos de adaptación.
- Construir shortlists.

---

### Usuario 3 — Catcher / decisión táctica

**Contexto:** preparación y decisiones rápidas de juego.

**Job to Be Done:**
> Evaluar rápidamente qué partes del arsenal y qué ubicaciones presentan mejores condiciones en una situación específica.

**Dolores principales:**
- No puede leer múltiples dashboards durante una decisión.
- Demasiadas métricas ralentizan la acción.
- Necesita información contextual y muy resumida.

**Ganancias buscadas:**
- Pocas opciones claras.
- Heatmaps simples.
- Lectura rápida de calidad y riesgo.
- Explicación solo si la solicita.

**Restricción metodológica:**
Mientras no exista evidencia suficiente para recomendar causalmente una política óptima pitch-by-pitch, el sistema debe mostrar **calidad esperada, probabilidades y contexto**, no afirmar de forma absoluta “este es el mejor pitch”.

---

## 5. Jobs, Pains y Gains globales

### Customer Jobs

1. **Entender**  
   ¿Qué tiene este pitcher?

2. **Diagnosticar**  
   ¿Qué le ayuda y qué lo limita?

3. **Comparar**  
   ¿Cómo cambia en Harp Helú frente a otros entornos?

4. **Modificar**  
   ¿Qué ocurre si ajusto velocidad, movimiento, ubicación o uso?

5. **Decidir**  
   ¿Qué debería hacer con esta información?

6. **Buscar**  
   ¿Qué otros pitchers tienen el perfil que quiero?

---

### Customer Pains

1. **Data overload**  
   Demasiadas variables y gráficas.

2. **Scores sin explicación**  
   Un “Stuff+ = 106” no dice qué hacer.

3. **Recomendaciones sin nivel de confianza**  
   No todas las señales tienen la misma robustez.

4. **Análisis estático**  
   Describir al pitcher no basta; hay que experimentar.

5. **Ignorar el entorno**  
   Una misma evaluación no necesariamente se traslada igual entre estadios.

6. **Repetir cargas y análisis manuales**  
   Tener que volver a subir perfiles o recrear estudios vuelve el sistema lento e ineficiente.

---

### Customer Gains

- Claridad.
- Explicabilidad.
- Acción.
- Experimentación.
- Contexto Harp Helú.
- Comparación Home vs Away.
- Persistencia e historial.
- Confianza explícita.
- Capacidad de scouting.
- Seguimiento de desarrollo.

---

## 6. Propuesta conceptual del producto

No se plantea como un simple **dashboard de Stuff+**.

Se plantea como un:

# Pitching Decision System para Harp Helú

Flujo conceptual:

**Datos → Modelo → Diagnóstico → Simulación → Recomendación → Decisión**

El dashboard será la interfaz de ese sistema.

---

## 7. Lluvia de ideas — 25 propuestas

1. **Simulador de juego**  
   Seleccionar pitch, ubicación y contexto para mostrar probabilidades, calidad esperada y mapa de ubicación.

2. **Perfil ideal**  
   Ingresar métricas de un prospecto o jugador y medir qué tan parecido es al perfil ideal.

3. **Simulador de altitud**  
   Comparación side-by-side del mismo pitch bajo distintas condiciones de altitud.

4. **Semáforo de arsenal**  
   Verde / amarillo / rojo para mostrar qué mantener, ajustar o reducir.

5. **Recomendador de pitch mix**  
   Sugerir porcentajes de uso del arsenal en Harp Helú.

6. **Comparador Home vs Away**  
   Mostrar cómo debería interpretarse el arsenal en casa vs fuera.

7. **Player Card para coaches**  
   Una página con fortalezas, riesgos y recomendaciones.

8. **Mapa Stuff+ vs Command**  
   Ubicar pitchers según calidad física y capacidad de ubicación.

9. **What-if Simulator**  
   Probar escenarios como +10% sliders, +1 mph, cambios de IVB/HB o ubicación.

10. **Detector de pitchers infravalorados**  
    Identificar jugadores con resultados normales pero características potencialmente valiosas en Harp Helú.

11. **Recomendador de secuencias**  
    Sugerir combinaciones de pitcheos.

12. **Heatmaps simplificados de ubicación**  
    Mostrar dónde atacar y qué zonas evitar.

13. **Alertas de mala adaptación**  
    Detectar uso excesivo de un pitch que sufre bajo ciertas condiciones.

14. **Decision Tree para roster**  
    Clasificar pitchers en categorías operativas.

15. **Comparador de pitchers**  
    Comparar candidatos en stuff, command, arsenal y adaptación.

16. **Historial del jugador**  
    Guardar evaluaciones por temporada / fecha.

17. **Harp Helú Fit**  
    Vista específica de adaptación del pitcher al entorno de CDMX.

18. **Development Targets**  
    Mostrar ajustes potenciales para acercarse a un perfil deseable.

19. **Arsenal Fingerprint**  
    Visualización de la “huella” del repertorio completo.

20. **Home/Away Strategy Card**  
    Tarjeta rápida con fortalezas, riesgos y diferencias casa / fuera.

21. **Confidence Layer**  
    Etiquetas: Robusta / Condicional / Incierta.

22. **Why? / Explain Recommendation**  
    Expandir cualquier recomendación para ver sus razones principales.

23. **Pitch Library / Saved Experiments**  
    Guardar escenarios simulados y compararlos después.

24. **Archetype Explorer**  
    Explorar grupos de pitchers o pitches con perfiles similares.

25. **Recommendation Inbox**  
    Resumen del roster con alertas, oportunidades y perfiles a revisar.

---

## 8. Criterios de priorización

Cada idea se filtra con cinco lentes:

### Deseabilidad
¿El usuario realmente la utilizaría?

### Factibilidad
¿Los datos y modelos actuales permiten construirla?

### Viabilidad
¿Puede cambiar una decisión real?

### Responsabilidad
¿Podemos mostrarla sin exagerar certeza o causalidad?

### Diferenciación
¿Nos separa de un dashboard estándar de métricas?

---

## 9. Priorización de funcionalidades


### Matriz de priorización

| Idea | Deseabilidad | Factibilidad | Viabilidad | Responsabilidad | Diferenciación | Decisión |
|---|---|---|---|---|---|---|
| Simulador de juego | Alta | Media | Alta | Media | Alta | **V2 / condicional** |
| Perfil ideal | Alta | Alta | Alta | Alta | Alta | **CORE** |
| Simulador de altitud | Alta | Alta | Alta | Alta | Muy alta | **CORE** |
| Semáforo de arsenal | Alta | Alta | Alta | Alta | Media | **CORE** |
| Recomendador de pitch mix | Alta | Baja-Media | Alta | Baja-Media | Alta | **V2** |
| Comparador Home vs Away | Alta | Alta | Alta | Alta | Muy alta | **CORE** |
| Player Card para coaches | Muy alta | Alta | Alta | Alta | Media | **CORE** |
| Mapa Stuff+ vs Command | Alta | Alta | Media-Alta | Alta | Media | **CORE secundario** |
| What-if Simulator | Muy alta | Media-Alta | Muy alta | Alta si se limita | Muy alta | **CORE** |
| Detector de pitchers infravalorados | Alta | Media | Alta | Media-Alta | Muy alta | **V1 avanzada / scouting** |
| Recomendador de secuencias | Alta | Baja | Alta | Baja | Muy alta | **V2 / V3** |
| Heatmaps simplificados de ubicación | Alta | Alta | Alta | Alta | Media | **CORE secundario** |
| Alertas de mala adaptación | Alta | Alta | Alta | Alta | Alta | **CORE** |
| Decision Tree para roster | Media-Alta | Media | Alta | Media | Alta | **Secundario / V2** |
| Comparador de pitchers | Muy alta | Alta | Alta | Alta | Alta | **CORE** |
| Historial del jugador | Muy alta | Alta | Alta | Alta | Media | **INFRAESTRUCTURA CORE** |
| Harp Helú Fit | Muy alta | Alta | Muy alta | Alta | Muy alta | **CORE** |
| Development Targets | Muy alta | Media-Alta | Muy alta | Media-Alta | Muy alta | **CORE** |
| Arsenal Fingerprint | Alta | Alta | Alta | Alta | Alta | **CORE secundario** |
| Home/Away Strategy Card | Alta | Alta | Alta | Alta | Alta | **CORE** |
| Confidence Layer | Muy alta | Alta | Muy alta | Muy alta | Alta | **CORE transversal** |
| Why? / Explain Recommendation | Muy alta | Alta | Alta | Muy alta | Alta | **CORE transversal** |
| Saved Experiments | Alta | Alta | Alta | Alta | Alta | **INFRAESTRUCTURA CORE** |
| Archetype Explorer | Media-Alta | Media | Media-Alta | Alta | Alta | **V2** |
| Recommendation Inbox | Alta | Alta | Alta | Alta | Alta | **CORE para director deportivo** |

### CORE absoluto

- Perfil ideal.
- Simulador de altitud.
- Semáforo de arsenal.
- Comparador Home vs Away.
- Player Card.
- What-if Simulator.
- Alertas de adaptación.
- Comparador de pitchers.
- Historial del jugador.
- Harp Helú Fit.
- Development Targets.
- Confidence Layer.
- Why? / Explain Recommendation.
- Saved Experiments.
- Recommendation Inbox.

### CORE secundario

- Stuff+ vs Command.
- Heatmaps.
- Arsenal Fingerprint.
- Home/Away Strategy Card.

### V2 / Condicional

- Simulador de juego.
- Detector de pitchers infravalorados.
- Decision Tree para roster.
- Archetype Explorer.

### No incluir en V1 salvo evidencia adicional

- Recomendador de pitch mix.
- Recomendador de secuencias.

La razón es que estos dos últimos requieren ir más allá de evaluar calidad y entrar en una política de uso óptima o recomendación táctica causal.

---

## 10. Persistencia e históricos

La persistencia se considera **infraestructura core**, no una comodidad.

El producto no debe funcionar como:

> subir archivo → analizar → cerrar → perder contexto

Debe funcionar como:

> roster / player library → perfil guardado → temporadas → arsenal → scores → simulaciones → recomendaciones → evolución

Ejemplo:

| Temporada | Stuff+ | Harp Helú Fit | Slider | Four-Seam |
|---|---:|---|---:|---:|
| 2024 | 103 | Medio | 101 | 102 |
| 2025 | 106 | Alto | 105 | 98 |
| 2026 | 109 | Alto | 108 | 96 |

Esto permite estudiar **desarrollo**, no solo evaluación puntual.

También deben guardarse:

- Perfiles de jugadores.
- Métricas cargadas.
- Resultados de modelos.
- Simulaciones.
- Experimentos.
- Recomendaciones.
- Shortlists.
- Historial temporal.

---

## 11. Confidence Layer

Toda recomendación debe mostrar su nivel de evidencia.

### Robusta
Consistente entre ventanas, chequeos y sensibilidades.

### Condicional
Existe señal, pero depende de supuestos, ponderaciones o contexto.

### Incierta
Muestra limitada, intervalos amplios o signo inestable.

La incertidumbre no debe esconderse: debe ser parte de la experiencia del producto.

---

## 12. User Journey — Coach de pitcheo

**Objetivo:** entender qué está pasando con un pitcher, cómo afecta Harp Helú y qué ajustes explorar.

### Flujo

1. Entra al roster.
2. Ve Stuff+, Location+, Harp Helú Fit, alertas y evolución.
3. Selecciona pitcher.
4. Abre Player Card.
5. Ve diagnóstico rápido:
   - fortaleza principal;
   - riesgo principal;
   - mejor pitch;
   - pitch con mayor riesgo;
   - confianza.
6. Explora arsenal.
7. Revisa Arsenal Fingerprint.
8. Revisa semáforo por pitch.
9. Compara Home vs Away.
10. Revisa componentes, no solo score neto.
11. Lee una recomendación.
12. Pulsa “¿Por qué?”.
13. Abre What-if Simulator.
14. Modifica variables.
15. Compara actual vs simulado.
16. Guarda experimento.
17. Consulta historial.
18. Sale con una decisión de desarrollo o uso.

### Patrón del journey

**Diagnóstico → explicación → simulación → decisión**

---

## 13. User Journey — Director deportivo / scouting

**Objetivo:** encontrar pitchers que encajen con Harp Helú y comparar candidatos.

### Flujo

1. Entra a Scouting.
2. Ve Perfil Ideal.
3. Revisa rangos deseables por pitch type y arsenal.
4. Busca candidato.
5. Puede:
   - cargar prospecto;
   - seleccionar jugador guardado;
   - filtrar roster.
6. Calcula Harp Helú Fit.
7. Ve desglose del fit:
   - físico;
   - arsenal;
   - adaptación;
   - riesgos.
8. Compara dos o más candidatos.
9. Revisa Home vs Away.
10. Explora pitchers potencialmente infravalorados.
11. Abre “Why?”.
12. Consulta historial.
13. Clasifica candidato.
14. Guarda shortlist.

### Patrón del journey

**Buscar → evaluar → comparar → explicar → priorizar**

---

## 14. User Journey — Catcher / decisión táctica

**Objetivo:** tomar una decisión rápida con poca carga cognitiva.

### Flujo

1. Selecciona pitcher.
2. Selecciona lado del bateador.
3. Selecciona count / contexto disponible.
4. Ve arsenal.
5. Ve opciones resumidas por pitch.
6. Observa:
   - calidad física;
   - Location+;
   - probabilidades relevantes;
   - confianza.
7. Abre heatmap.
8. Ve zonas favorables / riesgosas.
9. Consulta recomendación resumida.
10. Expande “Why?” solo si lo necesita.

### Regla de UX

Este usuario no debe ver por defecto:

- SHAP.
- demasiadas métricas.
- historial completo.
- simuladores complejos.
- documentación técnica.

### Patrón del journey

**Contexto → pocas opciones → ubicación → decisión rápida**

---

## 15. Arquitectura preliminar del producto

### Navegación principal

**Roster | Player | Simulator | Scouting | Game**

---

### Roster

- Recommendation Inbox.
- Alertas.
- Lista de pitchers.
- Harp Helú Fit.
- Tendencias.
- Comparaciones rápidas.

---

### Player

#### Overview
- Player Card.
- Stuff+.
- Location+.
- Pitching+.
- Harp Helú Fit.
- fortalezas y riesgos.

#### Arsenal
- Semáforo.
- Arsenal Fingerprint.
- Stuff+ por pitch.
- Location+ por pitch.
- uso.
- diferenciales vs fastball.

#### Harp Helú Fit
- Home vs Away.
- adaptación.
- componentes.
- confianza.

#### History
- tendencias.
- cambios de arsenal.
- evolución de scores.
- recomendaciones pasadas.

---

### Simulator

#### Altitude
- mismo pitch en distintos entornos.
- comparación side-by-side.

#### What-if
- velocidad.
- IVB.
- HB.
- ubicación.
- uso, si está metodológicamente permitido.

#### Saved Experiments
- escenarios guardados.
- comparación de versiones.

---

### Scouting

#### Ideal Profile
- perfil físico / arsenal deseable.

#### Compare
- comparación de candidatos.

#### Shortlist
- jugadores guardados.

#### Undervalued
- candidatos potencialmente infravalorados.

---

### Game

- pitcher.
- lado del bateador.
- count.
- arsenal.
- quality / expected outcomes.
- heatmap.
- recomendación resumida.
- nivel de confianza.

---

## 16. Principios de UX

1. **Conclusión antes que métrica.**
2. **Explicar solo bajo demanda.**
3. **No esconder incertidumbre.**
4. **No convertir correlación en causalidad.**
5. **No mostrar un score aislado sin contexto.**
6. **El usuario debe poder comparar Home vs Away.**
7. **El coach debe poder experimentar.**
8. **El director debe poder comparar perfiles.**
9. **El catcher debe poder decidir rápido.**
10. **Persistir historial, perfiles y experimentos.**
11. **Reducir carga cognitiva.**
12. **Separar análisis técnico de acción operativa.**

---

## 17. Propuesta de valor preliminar

> **Nuestro Pitching Decision System ayuda al cuerpo técnico y a la dirección deportiva de Diablos Rojos a convertir Stuff+, ubicación, arsenal y efecto de altitud en decisiones claras de desarrollo, uso y selección de pitchers, mediante perfiles explicables, simulaciones Home vs Away y recomendaciones con nivel de confianza.**

---

## 18. Qué NO queremos construir

- Un dashboard lleno de gráficas sin decisión asociada.
- Un visor estático de Stuff+.
- Un sistema que obligue a cargar el mismo jugador cada vez.
- Recomendaciones categóricas sin nivel de evidencia.
- Un “mejor pitch” absoluto sin soporte causal.
- Una interfaz diseñada para impresionar técnicamente pero difícil de usar.
- Un producto que ignore la diferencia entre casa y fuera.
- Un score único que esconda trade-offs importantes entre componentes.

---

## 19. Estado actual del proceso de Design Thinking

### Completado

- Definición del problema.
- Identificación de 3 usuarios.
- Jobs to Be Done.
- Pains.
- Gains.
- How Might We.
- Lluvia de 25 ideas.
- Priorización.
- User journeys.
- Arquitectura preliminar.

### Siguiente paso

**Diseñar el flujo pantalla por pantalla**, comenzando por el **Coach de pitcheo**, ya que es el usuario con el journey más completo y probablemente define la mayoría de los componentes reutilizables del sistema.

Después:

1. Wireframe de Coach.
2. Reutilización para Director / Scouting.
3. Simplificación para Game Mode.
4. Diseño visual.
5. Prototipo.
6. Validación con usuarios / pruebas de uso.
