# Panorama del proyecto — Stuff+ Diablos Rojos

> Una explicación sencilla de qué estamos construyendo, qué hemos hecho hasta ahora y qué sigue.

---

## 1. ¿Qué estamos tratando de construir?

La idea central del hackathon es construir un sistema que pueda **evaluar la calidad de cada lanzamiento**.

No queremos simplemente decir si un pitcher es “bueno” o “malo”. Queremos bajar al nivel de cada pitch y responder preguntas como:

- ¿Qué tan bueno es físicamente este lanzamiento?
- ¿Qué tan bien fue ubicado?
- ¿Qué tan bien funciona dentro del arsenal completo del pitcher?
- ¿Cómo cambia su valor cuando se lanza en la altitud de CDMX?

El reto plantea tres capas principales:

### Stuff+ físico

Responde:

> **¿Qué tan bueno es este lanzamiento por sus características físicas?**

Aquí importan cosas como:

- velocidad;
- movimiento vertical;
- movimiento horizontal;
- spin;
- punto de release;
- extensión;
- geometría del lanzamiento.

En esta capa **no queremos usar todavía la ubicación final ni el resultado del lanzamiento**.

---

### Location+

Responde:

> **Dado este tipo de lanzamiento, ¿qué tan bien fue colocado?**

Aquí ya entran cosas como:

- ubicación en el plato;
- zona;
- lado del bateador;
- posiblemente el conteo.

---

### Pitching+ / modelo combinado

Responde:

> **Juntando físico, ubicación, arsenal, contexto y altitud, ¿qué tan valioso esperamos que sea este pitch?**

Esta será la capa más cercana a una herramienta realmente útil para coaches o scouting.

---

## 2. ¿Por qué la altitud importa tanto?

El Harp Helú está a gran altitud.

A mayor altitud hay menor densidad del aire, así que la pelota interactúa diferente con el ambiente. En particular, puede cambiar cuánto se mueve un lanzamiento.

Por eso una de nuestras preguntas más importantes es:

> **Si mantuviéramos exactamente el mismo pitch, ¿cómo cambiaría su valoración si se lanzara en CDMX en vez de en un ambiente de menor altitud?**

Ese es uno de los elementos que hace diferente este proyecto frente a un Stuff+ genérico.

---

## 3. ¿Qué hicimos primero?

Antes de modelar, tuvimos que conocer muy bien los datos.

El dataset contiene aproximadamente:

- **635 mil lanzamientos**;
- tres temporadas;
- más de mil pitchers;
- múltiples categorías de altitud;
- 84 variables originales.

También vimos que hay muchos pitchers que aparecen en más de un entorno de altitud.

Eso es importante porque nos permite hacer comparaciones más limpias.

En vez de comparar:

> pitcher A en baja altitud vs pitcher B en alta altitud,

podemos comparar:

> **el mismo pitcher, el mismo tipo de pitch y la misma temporada en ambientes diferentes.**

Eso reduce muchísimo el riesgo de atribuir a la altitud algo que en realidad se debe a que estamos comparando jugadores distintos.

---

## 4. ¿Qué tipos de pitch estamos usando?

Para la primera versión decidimos concentrarnos en:

- Four-Seam;
- Sinker;
- Slider;
- Changeup;
- Curveball;
- Cutter;
- Splitter.

Otros tipos, como Sweeper, tienen muy pocos datos o una cobertura muy particular, así que por ahora se dejan fuera del modelado principal.

---

## 5. ¿Encontramos evidencia de que la altitud sí afecta los lanzamientos?

Sí.

Antes de construir el modelo final hicimos análisis descriptivos para comprobar si el fenómeno que motiva el hackathon realmente aparecía en los datos.

Y sí aparece.

Cuando comparamos:

- mismo pitcher;
- mismo tipo de pitch;
- misma temporada;
- diferente altitud,

vemos que varios tipos de lanzamiento **pierden movimiento en Extreme Altitude**, aunque su velocidad y su spin cambien muy poco.

### Ejemplo: Four-Seam

Aproximadamente:

- velocidad: casi igual;
- spin: casi igual;
- IVB: **-3.83 pulgadas**;
- movimiento horizontal: **-1.94 pulgadas**;
- whiff: también disminuye.

La interpretación sencilla es:

> **El pitcher puede lanzar de forma muy parecida, pero la pelota no se comporta igual en altitud.**

Esto es una señal fuerte de que sí tiene sentido construir un modelo específico para el entorno de CDMX.

---

## 6. ¿Por qué empezamos a hablar del “arsenal”?

Porque un pitch no existe aislado.

Por ejemplo, un changeup de 84 mph puede ser muy distinto dependiendo de la recta del pitcher.

No es lo mismo:

- changeup de 84 mph + fastball de 87 mph;

que:

- changeup de 84 mph + fastball de 96 mph.

En el segundo caso hay mucho mayor separación de velocidad, lo cual puede hacer que el cambio funcione de forma diferente.

Entonces empezamos a crear variables que describen cada lanzamiento **en relación con la fastball principal del pitcher**.

---

## 7. ¿Qué variables nuevas de arsenal construimos?

Creamos, entre otras:

### `velo_diff_vs_fb`

Diferencia de velocidad entre el pitch actual y la fastball principal.

---

### `ivb_diff_vs_fb`

Diferencia de movimiento vertical respecto a la fastball.

---

### `hb_diff_vs_fb`

Diferencia de movimiento horizontal respecto a la fastball.

---

### `spin_diff_vs_fb`

Diferencia de spin respecto a la fastball.

---

### `release_distance_vs_fb`

Qué tan lejos está el punto de release de este pitch respecto al release típico de la fastball.

Es un **proxy** inicial de similitud de release, pero no lo llamamos tunneling completo.

---

### `extension_diff_vs_fb`

Diferencia de extensión respecto a la fastball.

---

### `pitch_usage_pct`

Qué porcentaje del arsenal del pitcher representa ese tipo de lanzamiento.

---

## 8. Para hacer eso tuvimos que definir la fastball principal

Esto parece un detalle pequeño, pero en realidad afecta muchas variables del proyecto.

Decidimos que la **primary fastball** puede ser:

- Four-Seam;
- o Sinker.

Se elige la que tenga más uso dentro del periodo de referencia.

Pero descubrimos un problema.

Había pitcher-years donde la supuesta fastball principal estaba definida con apenas:

- 1 lanzamiento;
- 6 lanzamientos;
- 8 lanzamientos.

Eso era demasiado inestable.

Por eso fijamos una regla:

> **La fastball principal necesita al menos 20 lanzamientos para considerarse una referencia válida.**

Con ese umbral todavía conservamos cerca del **97% del volumen útil**.

---

## 9. ¿Y qué pasa si Four-Seam y Sinker están empatadas?

También cerramos esa regla.

Si ambas tienen exactamente el mismo número de lanzamientos:

1. se elige la que tenga mayor velocidad media;
2. si también empatan en velocidad, se elige Four-Seam;
3. siempre se exige un mínimo de 20 lanzamientos.

Además probamos que la selección ya no depende del orden de las filas del dataset.

Eso hace que el pipeline sea reproducible.

---

## 10. Ya construimos un primer modelo

Sí.

Todavía no es el Stuff+ final, pero ya tenemos un primer modelo útil.

La primera pregunta que elegimos fue:

> **¿Qué tan probable es que este lanzamiento genere un whiff?**

Whiff significa que el bateador hace swing y no toca la pelota.

Para hacerlo paso a paso comparamos tres niveles:

### Modelo 1 — Promedio global

Todos los pitches reciben prácticamente la misma probabilidad.

AUC:

**0.5000**

---

### Modelo 2 — Solo tipo de pitch

El modelo solo sabe si el lanzamiento es:

- Slider;
- Four-Seam;
- Changeup;
- etc.

AUC:

**0.5929**

---

### Modelo 3 — Propiedades físicas

Además del tipo de pitch, usa características como:

- velocidad;
- movimiento;
- spin;
- release;
- geometría.

AUC:

**0.6136**

También mejora Log Loss y Brier Score.

La conclusión importante no es que 0.6136 sea “el número final”.

La conclusión es:

> **Las propiedades físicas del lanzamiento sí contienen información adicional sobre su capacidad de generar whiffs.**

Eso valida la idea de construir un Stuff+ físico.

---

## 11. ¿Qué significa eso de “leakage” que aparece tanto?

Leakage significa que el modelo recibe información que **no debería conocer al momento de hacer una predicción**.

Ejemplo:

Supongamos que queremos predecir 2026.

Sería incorrecto calcular el arsenal del pitcher usando datos de 2026 y luego usar ese arsenal para “predecir” 2026.

Sería como estudiar para un examen usando las respuestas del propio examen.

Por eso construimos un pipeline especial para evitarlo.

---

## 12. ¿Qué hace `04_fold_safe_arsenal.py`?

Ese script asegura que toda la información del arsenal venga únicamente del pasado permitido.

Por ejemplo:

### Para evaluar 2025

Se construyen las referencias del pitcher usando solo:

**2024**

---

### Para evaluar 2026

Se construyen usando:

**2024 + 2025**

---

Nunca se utiliza el año que estamos evaluando para construir:

- primary fastball;
- perfil de fastball;
- pitch usage;
- referencias de arsenal.

Esto hace que el experimento sea mucho más realista.

---

## 13. ¿Funcionó esta estrategia?

Sí.

Entre pitchers que ya tienen historial previo:

- para 2025 tenemos referencia válida de fastball en aproximadamente **96.2%** de las filas elegibles;
- para 2026 llegamos aproximadamente a **98.0%**.

Eso significa que podemos usar información histórica del arsenal en una gran parte de los casos reales.

También vimos que la primary fastball es bastante estable de año a año, pero no completamente fija.

Eso justifica actualizarla con el tiempo en vez de definir una única fastball para toda la carrera del pitcher.

---

## 14. Entonces, ¿dónde estamos ahora?

La forma más fácil de verlo es:

### Ya terminamos la cimentación.

Todavía no tenemos el Stuff+ final.

Pero ya sabemos:

- qué problema estamos resolviendo;
- qué significa cada capa;
- qué variables pertenecen a Stuff+, Location+ y Pitching+;
- qué pitch types vamos a usar;
- que la altitud sí muestra un efecto en los datos;
- cómo representar el arsenal;
- cómo definir una fastball principal estable;
- cómo evitar leakage;
- cómo validar con pitchers no vistos;
- cómo validar temporalmente;
- que el Stuff físico sí contiene señal predictiva.

---

## 15. ¿Cuál es el siguiente paso?

> **Cierre ratificado:** `05_target_definitions.py` está cerrado. V1 incluye whiff por pitch, whiff dado swing, called strike desde PitchCall, chase condicionado a fuera de zona y groundball en BIP con tipo conocido y sin bunts. Weak/hard contact, out-within-PA y xRunValue quedan fuera de V1. Las inconsistencias se documentan sin corregir etiquetas; `is_hit_by_pitch` no se usa como fuente. La evidencia está en `05_target_definitions_audit.txt`. El experimento incremental de arsenal es el siguiente paso pendiente, ahora numerado 06.

El siguiente experimento es muy concreto.

Ya sabemos que el modelo físico tiene señal.

Ahora queremos responder:

> **¿Agregar contexto de arsenal mejora todavía más el modelo?**

En otras palabras, comparar algo parecido a:

### Modelo físico

Sabe:

> “Este pitch va a X mph, se mueve Y pulgadas, tiene este spin y sale desde aquí.”

vs.

### Modelo físico + arsenal

También sabe:

> “Este pitch es 8 mph más lento que la fastball del pitcher, tiene cierto diferencial vertical, cierto diferencial horizontal, sale desde casi el mismo punto y el pitcher lo usa 20% del tiempo.”

El siguiente script es:

`06_arsenal_incremental_value.py`

La pregunta será:

> **¿Las features de arsenal aportan información predictiva adicional sobre whiff por encima del Stuff físico?**

No estamos suponiendo que sí.

Lo vamos a medir.

---

## 16. ¿Y después qué sigue?

El roadmap general es:

```text
Datos
  ↓
Exploración y limpieza
  ↓
Stuff físico
  ↓
Arsenal
  ↓
Location+
  ↓
Altitud
  ↓
Pitching+ combinado
  ↓
Normalización a escala Stuff+
  ↓
Interpretabilidad
  ↓
Agregaciones por pitcher / arsenal
  ↓
Dashboard
```

---

## 17. ¿Cómo se vería el producto final?

La meta es poder tomar un lanzamiento y producir algo conceptualmente parecido a:

> Este slider tiene Stuff+ 112.  
> Físicamente es un lanzamiento por encima del promedio.  
> Además funciona bien dentro del arsenal porque tiene buena separación de velocidad respecto a la fastball.  
> Su ubicación añade valor.  
> En baja altitud esperaríamos un score mayor, pero en CDMX pierde parte de su efectividad por la reducción de movimiento.

Todavía no estamos produciendo exactamente ese output.

Pero **todas las piezas que hemos construido hasta ahora existen para llegar a eso de forma correcta y defendible**.

---

## Resumen en una frase

> **No hemos construido todavía el Stuff+ final; hemos construido y validado las piezas necesarias para que, cuando lo construyamos, tenga sentido, sea reproducible y no esté haciendo trampa con los datos.**
