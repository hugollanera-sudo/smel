# Notas de Despliegue - main.py v2
# ====================================

## Qué ha cambiado

Se ha sustituido el sistema de evaluación fila a fila (616 llamadas API por consulta)
por una arquitectura de 2 pasos (2 llamadas API por consulta).

### Eliminado:
- `evaluar_fila()` - hacía 1 llamada API por cada fila del Excel
- `evaluar_relevancia_df_async()` - lanzaba las 616 llamadas en paralelo
- `class IsRelevant` - modelo Pydantic para la respuesta fila a fila

### Añadido:
- `clasificar_categorias()` - Paso 1: 1 llamada a gpt-4o para identificar categorías relevantes
- `filtrar_filas_por_categorias()` - Filtrado determinístico (sin IA)
- `evaluar_filas_filtradas()` - Paso 2: 1 llamada a gpt-4o para evaluar filas en batch
- `get_sheet_data()` - Caché de 5 min para la Google Sheet (antes se descargaba en cada consulta)
- `_normalizar()` - Helper para comparación insensible a tildes
- `_detectar_columna_categoria()` - Detección automática de columna "Categoría"/"Categoria"
- Constantes `MODELO_CLASIFICACION`, `MODELO_EVALUACION`, `MODELO_ORQUESTADOR` al inicio del archivo

### Sin cambios:
- Orquestador (`agente_principal`)
- WebSocket endpoint
- Formato de tabla Markdown (idéntico, compatible con Odoo)
- Sesiones de chat
- Funciones de caché de prompts (Google Docs)
- `limpiar_texto()`, `construir_tabla_markdown()`

## Cómo desplegar

1. Hacer backup del main.py actual:
   cp main.py main_BACKUP.py

2. Sustituir main.py por el nuevo archivo

3. No se necesitan nuevas dependencias (requirements.txt no cambia)

4. Reiniciar el servicio:
   docker restart smel-chatbots
   (o el comando equivalente según vuestra configuración)

## Cómo revertir

1. Restaurar el backup:
   cp main_BACKUP.py main.py

2. Reiniciar el servicio

## Resultado esperado

- Tiempo de respuesta: de 30-60 seg → 3-8 seg
- Llamadas API por consulta: de 616 → 2
- Modelo: gpt-4o (antes gpt-4o-mini)
- Precisión: mayor (el modelo ve categorías completas en vez de filas aisladas)
