import os
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pydantic import BaseModel
import json
import pandas as pd
import openai
import requests
import time
import re
import unicodedata
from typing import List, Dict, Any, Optional

# =========================
# CARGA CONFIG / ENTORNO
# =========================
load_dotenv(override=True)
PELIGROS_OPENAI_API_KEY = os.getenv("PELIGROS_OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODELO A UTILIZAR
# =========================
MODELO_CLASIFICACION = "gpt-4o"    # Paso 1: clasificar categorías
MODELO_EVALUACION = "gpt-4o"       # Paso 2: evaluar filas filtradas
MODELO_ORQUESTADOR = "gpt-4o-mini" # Orquestador (decidir table vs answer)

# =========================
# CACHÉS PROMPTS
# =========================
_cached_prompt_peligros = None
_cached_ai_context = None
_cached_prompt_orquestador = None
_cached_orquestador_timestamp = 0
_cached_prompt_timestamp = 0
_cache_timestamp = 0

CACHE_TTL = 60 * 10  # 10 min

def get_prompt_orquestador():
    global _cached_prompt_orquestador, _cached_orquestador_timestamp
    ahora = time.time()
    if _cached_prompt_orquestador and (ahora - _cached_orquestador_timestamp < CACHE_TTL):
        return _cached_prompt_orquestador
    file_id = "1FszjxanxJpQMxTQ2DdBeB_NBZS8eKYqHRfzD0S7Kr-I"
    url = f"https://docs.google.com/document/d/{file_id}/export?format=txt"
    response = requests.get(url)
    if response.status_code == 200:
        _cached_prompt_orquestador = str(response.text)
        _cached_orquestador_timestamp = ahora
        return _cached_prompt_orquestador
    else:
        return (
            "Eres un clasificador. Devuelve JSON {\"is_relevant\": false} "
            "si no puedes evaluar."
        )

def get_ai_peligros_prompt():
    global _cached_prompt_peligros, _cached_prompt_timestamp
    ahora = time.time()
    if _cached_prompt_peligros and (ahora - _cached_prompt_timestamp < CACHE_TTL):
        return _cached_prompt_peligros
    file_id = "1akk0bIrF8UBixgNrdf8yyiFg8umPr_uBNH2I2izgznI"
    url = f"https://docs.google.com/document/d/{file_id}/export?format=txt"
    response = requests.get(url)
    if response.status_code == 200:
        _cached_prompt_peligros = str(response.text)
        _cached_prompt_timestamp = ahora
        return _cached_prompt_peligros
    else:
        return (
            "Eres un clasificador. Devuelve JSON {\"is_relevant\": false} "
            "si no puedes evaluar."
        )

def get_ai_context_info():
    global _cached_ai_context, _cache_timestamp
    ahora = time.time()
    if _cached_ai_context and (ahora - _cache_timestamp < CACHE_TTL):
        return _cached_ai_context
    file_id = "1UI9tBBU4tJz2fot2BeAIZKyu3B7gyobaRJ4Yu7pgLpc"
    url = f"https://docs.google.com/document/d/{file_id}/export?format=txt"
    response = requests.get(url)
    if response.status_code == 200:
        _cached_ai_context = str(response.text)
        _cache_timestamp = ahora
        return _cached_ai_context
    else:
        return 'Responde que ha habido un problema en la extracción del prompt'

# =========================
# CACHÉ DE DATOS (Google Sheet)
# =========================
_cached_sheet_data = None
_cached_sheet_timestamp = 0
_cached_categorias = None
SHEET_CACHE_TTL = 60 * 5  # 5 minutos

SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/1CBzzbdIxGCyVbSaTv4tQwY8rCUj4z_Yxm3A2X5caOeM/"
    "export?format=csv&gid=374981202"
)

def _normalizar(texto):
    """Quita tildes y pasa a minúsculas para comparaciones."""
    if pd.isna(texto):
        return ""
    return unicodedata.normalize('NFD', str(texto).lower()).encode('ascii', 'ignore').decode().strip()

def _detectar_columna_categoria(df):
    """Detecta la columna de categoría con o sin tilde."""
    for col in df.columns:
        if _normalizar(col) == "categoria":
            return col
    return None

def get_sheet_data():
    """Lee y cachea la Google Sheet. Devuelve (df, nombre_col_categoria, lista_categorias)."""
    global _cached_sheet_data, _cached_sheet_timestamp, _cached_categorias

    ahora = time.time()
    if _cached_sheet_data is not None and (ahora - _cached_sheet_timestamp < SHEET_CACHE_TTL):
        return _cached_sheet_data, _cached_categorias

    raw_data = pd.read_csv(SHEET_URL)

    # Limpia filas totalmente vacías
    raw_data = raw_data[
        ~raw_data.apply(
            lambda row: all(
                str(cell).strip() in ["", "NaN", "nan", "-"] or pd.isna(cell)
                for cell in row
            ),
            axis=1,
        )
    ].reset_index(drop=True)

    # Detectar columna de categoría (con o sin tilde)
    col_cat = _detectar_columna_categoria(raw_data)
    if col_cat and col_cat != "Categoría":
        raw_data = raw_data.rename(columns={col_cat: "Categoría"})

    # Extraer categorías únicas
    categorias = sorted(raw_data["Categoría"].dropna().unique().tolist())

    _cached_sheet_data = raw_data
    _cached_categorias = categorias
    _cached_sheet_timestamp = ahora

    return raw_data, categorias


# =========================
# SESIONES DE CHAT
# =========================
class ChatSession:
    _next_conversation_id = 1

    def __init__(self):
        self.conversation_id = ChatSession._next_conversation_id
        ChatSession._next_conversation_id += 1
        self.messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": get_prompt_orquestador(),
            }
        ]

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def clone_messages(self) -> List[Dict[str, str]]:
        return list(self.messages)


# =========================
# CONNECTION MANAGER WS
# =========================
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, ChatSession] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = ChatSession()

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]

    def get_session(self, websocket: WebSocket) -> Optional[ChatSession]:
        return self.active_connections.get(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()


# =========================
# HELPERS
# =========================
def limpiar_texto(valor):
    if pd.isna(valor):
        return ""
    texto = str(valor).strip()
    if texto.lower() == "nan":
        return ""
    texto = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060\uFEFF\xa0]', '', texto)
    texto = texto.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    texto = texto.replace("|", "\\|")
    return texto


# ==========================================================
# PASO 1: CLASIFICACIÓN SEMÁNTICA DE CATEGORÍAS
# ==========================================================
PROMPT_PASO1 = """Eres un experto en seguridad alimentaria y clasificación de alimentos según la legislación europea.

TAREA: Dado un alimento que introduce el usuario, identifica TODAS las categorías de la lista proporcionada que podrían contener peligros relevantes para ese alimento.

INSTRUCCIONES:
- Analiza el alimento e identifica TODAS las categorías aplicables, tanto específicas como genéricas.
- Ejemplo: "cacahuete" podría estar en "Cacahuetes", "Cacahuetes y habas de soja", "Leguminosas", "Frutos secos", "Semillas oleaginosas", "Semillas y frutos oleaginosos".
- Si el alimento es un término genérico (ej: "marisco"), incluye todas las subcategorías relevantes (moluscos, crustáceos, cefalópodos, productos de la pesca, etc.).
- La comparación es insensible a tildes, mayúsculas y minúsculas.
- Ante la duda, INCLUYE la categoría. Es mucho mejor incluir una categoría de más que omitir una relevante.
- Devuelve los nombres EXACTOS tal como aparecen en la lista.

FORMATO DE RESPUESTA - JSON estricto, sin texto adicional, sin backticks:
{"categorias": ["Nombre exacto 1", "Nombre exacto 2", ...]}

Si el alimento no encaja en ninguna categoría:
{"categorias": []}"""


async def clasificar_categorias(alimento: str, categorias: list) -> list:
    """Paso 1: identifica qué categorías son relevantes para el alimento."""
    client = openai.AsyncClient(api_key=PELIGROS_OPENAI_API_KEY)

    user_msg = f"Alimento consultado: {alimento}\n\nCATEGORÍAS DISPONIBLES:\n"
    user_msg += "\n".join(f"- {c}" for c in categorias)

    try:
        completion = await client.chat.completions.create(
            model=MODELO_CLASIFICACION,
            temperature=0,
            messages=[
                {"role": "system", "content": PROMPT_PASO1},
                {"role": "user", "content": user_msg}
            ]
        )
        raw = completion.choices[0].message.content
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        return data.get("categorias", [])
    except Exception as e:
        print(f"[Paso 1] Error clasificando categorías: {e}")
        return []


# ==========================================================
# FILTRADO DETERMINÍSTICO (sin IA)
# ==========================================================
def filtrar_filas_por_categorias(df: pd.DataFrame, categorias_relevantes: list) -> pd.DataFrame:
    """Filtra las filas cuya Categoría coincide con las identificadas en el Paso 1."""
    cats_norm = set(_normalizar(c) for c in categorias_relevantes)

    def coincide(valor):
        val_norm = _normalizar(valor)
        # Coincidencia exacta o parcial (para categorías con subcadenas)
        return any(val_norm == cn or cn in val_norm or val_norm in cn for cn in cats_norm)

    mask = df["Categoría"].apply(coincide)
    return df[mask].reset_index(drop=True)


# ==========================================================
# PASO 2: EVALUACIÓN DE FILAS FILTRADAS
# ==========================================================
PROMPT_PASO2 = """Eres un experto en peligros alimentarios.

TAREA: Se te proporcionan filas pre-filtradas por categoría alimentaria. Para cada fila, determina si el peligro aplica al alimento concreto que consulta el usuario.

REGLAS:
1. INCLUYE la fila si el alimento pertenece claramente al grupo/producto/categoría descrito en esa fila.
2. EXCLUYE la fila SOLO si el campo Producto u Observaciones contiene una condición que excluye EXPLÍCITAMENTE el alimento consultado (ej: "excepto higos secos" para una consulta de "higo seco").
3. Ante la duda, INCLUYE. Es preferible incluir un peligro que podría no aplicar que omitir uno que sí aplica.
4. Evalúa todas las filas, no te saltes ninguna.

FORMATO DE RESPUESTA - JSON estricto, sin texto adicional, sin backticks:
{"indices": [0, 1, 3, 5]}

Donde los números son los índices [N] de las filas que SÍ son relevantes."""


async def evaluar_filas_filtradas(filas: list, alimento: str) -> list:
    """Paso 2: evalúa en batch las filas filtradas y devuelve los índices relevantes."""
    if not filas:
        return []

    client = openai.AsyncClient(api_key=PELIGROS_OPENAI_API_KEY)

    # Construir representación compacta de cada fila
    filas_texto = []
    for i, fila in enumerate(filas):
        partes = [
            f"Categoría: {fila.get('Categoría', '')}",
            f"Producto: {fila.get('Producto', '')}",
            f"Peligro: {fila.get('Peligro', '')}",
            f"Tipo: {fila.get('Tipo de peligro', '')}",
        ]
        obs = fila.get('Observaciones', '')
        if obs and str(obs).strip() and str(obs).strip().lower() != 'nan':
            partes.append(f"Observaciones: {obs}")
        filas_texto.append(f"[{i}] {' | '.join(partes)}")

    user_msg = f"Alimento consultado: {alimento}\n\nFILAS A EVALUAR ({len(filas)} filas):\n"
    user_msg += "\n".join(filas_texto)

    try:
        completion = await client.chat.completions.create(
            model=MODELO_EVALUACION,
            temperature=0,
            messages=[
                {"role": "system", "content": PROMPT_PASO2},
                {"role": "user", "content": user_msg}
            ]
        )
        raw = completion.choices[0].message.content
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        indices = data.get("indices", [])
        # Validar que los índices son válidos
        return [i for i in indices if isinstance(i, int) and 0 <= i < len(filas)]
    except Exception as e:
        print(f"[Paso 2] Error evaluando filas: {e}")
        # Fallback: incluir todas las filas filtradas (mejor de más que de menos)
        return list(range(len(filas)))


# ==========================================================
# CONSTRUIR TABLA MARKDOWN (sin cambios)
# ==========================================================
def construir_tabla_markdown(filas, alimento) -> str:
    if not filas:
        return "**No se han encontrado peligros alimentarios registrados para dichas especificaciones.**"

    tabla = f"**Peligros encontrados para: {alimento}**\n\n"
    tabla += "| Id | Producto | Categoría | Peligro | Tipo de peligro | Causa | Límite legal + n + c | Texto Legal | Observaciones |\n"
    tabla += "|----|----------|-----------|---------|-----------------|-------|---------------------|-------------|----------------|\n"

    for fila in filas:
        producto = limpiar_texto(fila.get("Producto"))
        categoria = limpiar_texto(fila.get("Categoría"))
        peligro = limpiar_texto(fila.get("Peligro"))
        tipo = limpiar_texto(fila.get("Tipo de peligro"))
        causa = limpiar_texto(fila.get("Causa"))
        limite = f"{limpiar_texto(fila.get('Limite legal'))} + {limpiar_texto(fila.get('n'))} + {limpiar_texto(fila.get('c'))}"
        texto_legal = limpiar_texto(fila.get("Texto legal"))
        url_texto_legal = str(fila.get("URL Texto legal") or "").strip()
        observaciones = limpiar_texto(fila.get("Observaciones"))
        _id = limpiar_texto(fila.get("ID"))

        if url_texto_legal and url_texto_legal.lower() != "none":
            texto_legal = f"[{texto_legal}]({url_texto_legal})"

        tabla += (
            f"| {_id} | {producto} | {categoria} | {peligro} | {tipo} | {causa} | "
            f"{limite} | {texto_legal} | {observaciones} |\n"
        )
    return tabla


# ==========================================================
# FUNCIÓN PRINCIPAL: COMPROBAR PELIGROS (2 PASOS)
# ==========================================================
async def comprobar_peligros_alimento(alimento: str) -> str:
    """
    Busca peligros alimentarios en 2 pasos:
    1. Clasificación semántica: identifica categorías relevantes (1 llamada LLM)
    2. Filtrado determinístico + evaluación de filas (1 llamada LLM)

    Sustituye el sistema anterior de 616 llamadas individuales.
    """
    # 1. Leer datos (con caché)
    df, categorias = get_sheet_data()

    # 2. Paso 1: clasificar categorías relevantes
    cats_relevantes = await clasificar_categorias(alimento, categorias)

    if not cats_relevantes:
        return "**No se han encontrado peligros alimentarios registrados para dichas especificaciones.**"

    # 3. Filtrado determinístico por categorías
    df_filtrado = filtrar_filas_por_categorias(df, cats_relevantes)

    if df_filtrado.empty:
        return "**No se han encontrado peligros alimentarios registrados para dichas especificaciones.**"

    filas_filtradas = df_filtrado.to_dict('records')

    # 4. Paso 2: evaluar filas filtradas en batch
    indices_relevantes = await evaluar_filas_filtradas(filas_filtradas, alimento)

    # 5. Construir tabla con las filas finales
    filas_finales = [filas_filtradas[i] for i in indices_relevantes]
    tabla = construir_tabla_markdown(filas_finales, alimento)
    return tabla


# =========================
# AGENTE PRINCIPAL / ORQUESTADOR (sin cambios)
# =========================
async def agente_principal(session: ChatSession, mensaje_usuario: str) -> dict:
    client = openai.AsyncClient(api_key=PELIGROS_OPENAI_API_KEY)
    session.add_user_message(mensaje_usuario)
    mensajes_llamada = session.clone_messages()

    completion = await client.chat.completions.create(
        model=MODELO_ORQUESTADOR,
        temperature=0,
        messages=mensajes_llamada
    )

    raw = completion.choices[0].message.content
    session.add_assistant_message(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "type": "answer",
            "content": "No he podido interpretar tu pregunta. ¿Puedes repetirla de otra forma sencilla?"
        }

    return data


# =========================
# WEBSOCKET (sin cambios)
# =========================
@app.websocket("/chat/peligros")
async def peligros_websocket(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            mensaje_usuario = await websocket.receive_text()

            session = manager.get_session(websocket)
            if session is None:
                session = ChatSession()
                manager.active_connections[websocket] = session

            decision = await agente_principal(session, mensaje_usuario)
            tipo = decision.get("type")

            if tipo == "table":
                alimento = decision.get("alimento", mensaje_usuario)
                tabla_markdown = await comprobar_peligros_alimento(alimento)
                await manager.send_message(tabla_markdown, websocket)

            elif tipo == "answer":
                contenido = decision.get("content", "No tengo respuesta ahora mismo.")
                await manager.send_message(contenido, websocket)

            else:
                await manager.send_message(
                    "No he entendido tu petición. Intenta preguntarme por un alimento concreto o una duda concreta.",
                    websocket
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await manager.send_message(
            f"Ha ocurrido un error procesando tu solicitud: {str(e)}",
            websocket
        )
        manager.disconnect(websocket)
