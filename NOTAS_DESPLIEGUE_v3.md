# Notas de Despliegue - main.py v3

## Qué ha cambiado

Se mantiene la arquitectura rápida basada en pocas llamadas a IA, pero se mejora la recuperación de filas para evitar falsos negativos como `semillas` -> ID 371 (`Semillas germinadas listas para el consumo`).

### Cambios principales

- Los prompts dejan de depender de Google Docs y quedan incluidos y versionados dentro de `main.py`.
- Se elimina la lógica y las cachés de los Google Docs de prompts que ya no son necesarias.
- La base de datos sigue siendo la Google Sheet de Smel y mantiene una caché de 5 minutos.
- Se mantiene la clasificación semántica por categorías.
- Se añade una segunda vía determinística de búsqueda sobre `Categoría + Producto + Observaciones`.
- Los candidatos obtenidos por ambas vías se unen y se eliminan duplicados antes de la evaluación final.
- La búsqueda ya no termina sin resultados solo porque la IA no haya seleccionado una categoría: la vía textual puede rescatar filas válidas.
- El Paso 2 prioriza no omitir peligros, pero excluye productos claramente incompatibles con las especificaciones del usuario.
- El orquestador conserva especificaciones relevantes como `crudo`, `cocido`, `congelado`, `germinado`, etc.
- Se añade un log de consola con categorías, filas recuperadas por cada vía, candidatos finales, resultados y tiempo de proceso.

## Dependencias

No se añaden nuevas dependencias.

La arquitectura operativa queda simplificada a:

- `main.py`: lógica y prompts.
- `.env`: claves y configuración sensible. No borrar ni sustituir.
- Google Sheet: base de datos editable de peligros.

## Cómo desplegar

Seguir el procedimiento habitual de actualización del chatbot:

1. Hacer copia/backup del contenedor o versión actual.
2. Sustituir únicamente `main.py` por la nueva versión.
3. Mantener intacto el fichero `.env`.
4. Ejecutar el script habitual de actualización del contenedor:

   `sh actualizar_chatbot_smel.sh`

5. Comprobar que el nuevo contenedor queda operativo y conservar el anterior parado hasta validar las pruebas.

## Cómo revertir

Si aparece un problema:

1. Parar el contenedor nuevo.
2. Volver a arrancar/restaurar el contenedor anterior conservado durante el despliegue.

## Pruebas mínimas después del despliegue

Comprobar, como mínimo:

- `semillas`: debe incluir ID 371.
- `semillas germinadas`: debe incluir ID 371.
- `productos germinados`: debe incluir ID 371.
- `frutas germinadas`: no debe incluir ID 371.
- `frutas troceadas`: debe alcanzar IDs 372 y 405.
- `moluscos cocidos`: debe alcanzar IDs 369, 403 y 404.
- `moluscos vivos`: debe alcanzar IDs 370 y 377.
- `helados`: debe alcanzar IDs 366 y 388.
- `berberechos`: comprobar que la búsqueda semántica sigue funcionando aunque el término no aparezca literalmente en la base de datos.
- `frutas` y `aditivos`: mantener las comprobaciones históricas para detectar regresiones.

Además de revisar las filas, comprobar el tiempo de respuesta y el log de consola. El objetivo es mantener una respuesta de pocos segundos sin sacrificar precisión.

## Base de datos

La Google Sheet sigue siendo la fuente operativa. No es necesario cambiar su estructura para desplegar esta versión.

Antes de eliminar registros duplicados de la base, revisar que las IDs no estén referenciadas desde otros procesos o datos históricos.
