# Presentación final (charla de 5 minutos, en castellano)

| archivo | contenido |
|---|---|
| `presentacion.html` | La charla, autocontenida: 10 diapositivas principales + 3 extra. Teclas: ← → para moverse, N notas del orador, O índice. Las ecuaciones usan KaTeX desde un CDN (requiere internet). |
| `respuestas.txt` | Respuestas a las preguntas hechas al pedir esta versión. |
| `fuente/deck_source.html` | Fuente editable de la charla. |
| `fuente/build_presentation.py` | Incrusta las figuras de `figuras/` y `pol_data.json` y escribe `presentacion.html`. |
| `fuente/figuras_es.py` | Regenera `figuras/r_limite_bh_es.png` (reflectividad, en castellano) desde `Full material/First round/results/scattering.json`. |
| `figuras/` | Figuras usadas (copias recortadas de `Full material/Final round/figures` más la figura en castellano). |

Editar: cambiar `fuente/deck_source.html` y correr `python -B fuente/build_presentation.py`.

Simulaciones (todas corren en el navegador): colapso de la pared de dominio frente al cascarón fluido; las dos
paridades llegando al cascarón (la deformación del cascarón par es ilustrativa); paquete de ondas con ecos;
cascarón frente a estrella de densidad uniforme (paridad impar, cálculo real); explorador de polarización (extra).
