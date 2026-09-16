import os
import json
import re
import time
import unicodedata
from typing import List, Dict, Optional

import openai
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


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
# MODELOS A UTILIZAR
# =========================
MODELO_CLASIFICACION = "gpt-4o"    # Paso 1: clasificar categorias
MODELO_EVALUACION = "gpt-4o"       # Paso 2: evaluar filas candidatas
MODELO_ORQUESTADOR = "gpt-4o-mini" # Decidir table vs answer y extraer alimento


# =========================
# PROMPTS (VERSIONADOS CON EL CODIGO)
# =========================
PROMPT_ORQUESTADOR = """Eres Smel, asistente experto en seguridad alimentaria. Tu mision es ayudar al usuario de forma segura y prudente. Responde SIEMPRE en JSON valido y NADA MAS.

Formatos posibles:
1. {"type": "answer", "content": "..."}
2. {"type": "table", "alimento": "..."}

CUANDO usar type="table":
- Cuando el usuario quiera conocer peligros, riesgos, contaminantes o limites legales de un alimento o grupo de alimentos.
- Si el usuario escribe solo un nombre de alimento o categoria (por ejemplo: "harina", "pollo", "atun", "fruta", "cereales", "aditivos", "lacteos", "marisco"), se entiende que quiere los peligros. Usa type="table".
- Los terminos genericos o de grupo son consultas validas. No pidas al usuario que concrete mas.

COMO rellenar el campo "alimento":
- Extrae la expresion que identifica el alimento consultado y elimina solamente palabras conversacionales o de intencion como "peligros de", "riesgos de", "que peligros tiene" o equivalentes.
- CONSERVA todas las especificaciones que describan el alimento, porque pueden cambiar los registros aplicables: tipo, variedad, especie, grupo, estado, conservacion, preparacion o tratamiento.
- NO elimines palabras como fresco, congelado, crudo, cocido, en conserva, germinado, seco, ahumado, etc.
- Ejemplos:
  - "berberechos frescos de la ria" -> "berberechos frescos de la ria"
  - "pollo crudo entero" -> "pollo crudo entero"
  - "harina de trigo integral" -> "harina de trigo integral"
  - "atun rojo fresco" -> "atun rojo fresco"
  - "espinacas congeladas" -> "espinacas congeladas"
  - "leche cruda de cabra" -> "leche cruda de cabra"
  - "semillas germinadas" -> "semillas germinadas"
  - "frutas germinadas" -> "frutas germinadas"

- SOLO cuando el usuario mencione DOS O MAS alimentos DISTINTOS separados por "y", coma o similar (por ejemplo: "pollo y merluza", "harina, leche y huevos"), responde con type="answer" indicando que consulte los alimentos de uno en uno para obtener resultados precisos.

CUANDO usar type="answer":
- Recomendaciones de consumo seguro, embarazo, conservacion, temperatura interna, etc.
- Preguntas generales sobre seguridad alimentaria.
- SOLO cuando el usuario enumera dos o mas alimentos distintos en una misma consulta.

NO des mas claves aparte de las definidas. NO metas saltos de linea fuera de JSON."""


PROMPT_PASO1 = """Eres un experto en seguridad alimentaria y clasificacion de alimentos segun la legislacion europea.

TAREA: Dado un alimento que introduce el usuario, identifica TODAS las categorias de la lista proporcionada que podrian contener peligros relevantes para ese alimento.

INSTRUCCIONES:
- Analiza el alimento e identifica TODAS las categorias aplicables, tanto especificas como genericas.
- Ejemplo: "cacahuete" podria estar en "Cacahuetes", "Cacahuetes y habas de soja", "Leguminosas", "Frutos secos", "Semillas oleaginosas", "Semillas y frutos oleaginosos".
- Si el alimento es un termino generico (ej: "marisco"), incluye todas las subcategorias relevantes (moluscos, crustaceos, cefalopodos, productos de la pesca, etc.).
- Las especificaciones indicadas por el usuario (por ejemplo: crudo, cocido, fresco, congelado, germinado) forman parte de la consulta y no deben ignorarse.
- La comparacion es insensible a tildes, mayusculas y minusculas.
- Ante la duda, INCLUYE la categoria. Es mucho mejor incluir una categoria de mas que omitir una relevante.
- Devuelve los nombres EXACTOS tal como aparecen en la lista.

FORMATO DE RESPUESTA - JSON estricto, sin texto adicional, sin backticks:
{"categorias": ["Nombre exacto 1", "Nombre exacto 2", ...]}

Si el alimento no encaja en ninguna categoria:
{"categorias": []}"""


PROMPT_PASO2 = """Eres un experto en peligros alimentarios.

TAREA: Se te proporcionan filas candidatas de una base de datos. Para cada fila, determina si el peligro puede aplicar al alimento concreto que consulta el usuario.

OBJETIVO: priorizar el RECALL. Es preferible devolver alguna fila de mas, porque los resultados seran validados posteriormente uno a uno, pero no debes incluir filas claramente incompatibles con lo que el usuario ha especificado.

REGLAS:
1. Si la consulta es generica, INCLUYE los subtipos concretos que pertenecen claramente al grupo consultado.
   - "semillas" incluye "semillas germinadas listas para el consumo".
   - "moluscos" puede incluir tanto moluscos cocidos como moluscos vivos.
2. Una expresion generica puede abarcar un producto concreto aunque no repita el mismo sustantivo si la relacion es clara.
   - "productos germinados" incluye "semillas germinadas listas para el consumo".
3. Si el usuario anade una especificacion que hace que la fila sea claramente incompatible, EXCLUYE la fila.
   - "frutas germinadas" NO incluye "semillas germinadas".
   - "moluscos cocidos" NO incluye una fila aplicable exclusivamente a "moluscos vivos".
4. EXCLUYE tambien cuando Producto u Observaciones contengan una exclusion explicita que afecte al alimento consultado (por ejemplo, "excepto higos secos" para una consulta de "higos secos").
5. No descartes una fila solo porque sea mas especifica que la consulta. Si ese producto especifico es un subtipo razonable del termino consultado, INCLUYE.
6. Ante una duda real, INCLUYE. Solo excluye por incompatibilidad suficientemente clara.
7. Evalua TODAS las filas y no te saltes ninguna.

FORMATO DE RESPUESTA - JSON estricto, sin texto adicional, sin backticks:
{"indices": [0, 1, 3, 5]}

Donde los numeros son los indices [N] de las filas que SI son relevantes."""


# =========================
# CACHE DE DATOS (Google Sheet)
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
    """Quita tildes, pasa a minusculas y recorta extremos."""
    if pd.isna(texto):
        return ""
    return (
        unicodedata.normalize("NFD", str(texto).lower())
        .encode("ascii", "ignore")
        .decode()
        .strip()
    )


def _detectar_columna_categoria(df):
    """Detecta la columna de categoria con o sin tilde."""
    for col in df.columns:
        if _normalizar(col) == "categoria":
            return col
    return None


def get_sheet_data():
    """Lee y cachea la Google Sheet. Devuelve (df, lista_categorias)."""
    global _cached_sheet_data, _cached_sheet_timestamp, _cached_categorias

    ahora = time.time()
    if _cached_sheet_data is not None and (ahora - _cached_sheet_timestamp < SHEET_CACHE_TTL):
        return _cached_sheet_data, _cached_categorias

    raw_data = pd.read_csv(SHEET_URL)

    # Limpia filas totalmente vacias
    raw_data = raw_data[
        ~raw_data.apply(
            lambda row: all(
                str(cell).strip() in ["", "NaN", "nan", "-"] or pd.isna(cell)
                for cell in row
            ),
            axis=1,
        )
    ].reset_index(drop=True)

    # Detectar columna de categoria (con o sin tilde)
    col_cat = _detectar_columna_categoria(raw_data)
    if col_cat and col_cat != "Categoría":
        raw_data = raw_data.rename(columns={col_cat: "Categoría"})

    # Extraer categorias unicas
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
                "content": PROMPT_ORQUESTADOR,
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
# HELPERS DE SALIDA
# =========================
def limpiar_texto(valor):
    if pd.isna(valor):
        return ""
    texto = str(valor).strip()
    if texto.lower() == "nan":
        return ""
    texto = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060\uFEFF\xa0]", "", texto)
    texto = texto.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    texto = texto.replace("|", "\\|")
    return texto


# ==========================================================
# PASO 1: CLASIFICACION SEMANTICA DE CATEGORIAS
# ==========================================================
async def clasificar_categorias(alimento: str, categorias: list) -> list:
    """Identifica categorias potencialmente relevantes para el alimento."""
    client = openai.AsyncClient(api_key=PELIGROS_OPENAI_API_KEY)

    user_msg = f"Alimento consultado: {alimento}\n\nCATEGORÍAS DISPONIBLES:\n"
    user_msg += "\n".join(f"- {c}" for c in categorias)

    try:
        completion = await client.chat.completions.create(
            model=MODELO_CLASIFICACION,
            temperature=0,
            messages=[
                {"role": "system", "content": PROMPT_PASO1},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = completion.choices[0].message.content
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        return data.get("categorias", [])
    except Exception as e:
        print(f"[Paso 1] Error clasificando categorias: {e}")
        return []


# ==========================================================
# GENERACION DE CANDIDATOS DETERMINISTICA
# ==========================================================
_STOPWORDS_BUSQUEDA = {
    "a", "al", "de", "del", "el", "la", "los", "las", "un", "una", "unos", "unas",
    "y", "e", "o", "u", "en", "con", "sin", "para", "por", "que", "como",
    "tipo", "tipos", "alimento", "alimentos", "producto", "productos",
    "materia", "materias", "prima", "primas", "ingrediente", "ingredientes",
    "peligro", "peligros", "riesgo", "riesgos", "limite", "limites", "legal", "legales",
}


def _raiz_token_busqueda(token: str) -> str:
    """Normalizacion morfologica ligera sin dependencias externas.

    Igualamos plurales habituales y participios con genero distinto:
    semillas -> semilla; moluscos -> molusco; germinadas/germinados -> germinad.
    No pretende ser un stemmer general: solo ampliar recall de la busqueda textual.
    """
    if len(token) > 5 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("s"):
        token = token[:-1]

    if len(token) > 5 and token.endswith(("ado", "ada", "ido", "ida")):
        token = token[:-1]

    return token


def _tokenizar_busqueda(texto: str) -> list:
    normalizado = _normalizar(texto)
    tokens = re.findall(r"[a-z0-9]+", normalizado)
    resultado = []
    for token in tokens:
        if token in _STOPWORDS_BUSQUEDA or len(token) <= 2:
            continue
        resultado.append(_raiz_token_busqueda(token))
    return resultado


def filtrar_filas_por_categorias(df: pd.DataFrame, categorias_relevantes: list) -> pd.DataFrame:
    """Filtra filas cuya categoria coincide con las identificadas en el Paso 1."""
    if not categorias_relevantes:
        return df.iloc[0:0].copy()

    cats_norm = {_normalizar(c) for c in categorias_relevantes}

    def coincide(valor):
        val_norm = _normalizar(valor)
        return any(val_norm == cn or cn in val_norm or val_norm in cn for cn in cats_norm)

    mask = df["Categoría"].apply(coincide)
    return df[mask].copy()


def filtrar_filas_por_texto(df: pd.DataFrame, alimento: str) -> pd.DataFrame:
    """Via de rescate deterministica por contenido de la propia base de datos.

    Todos los terminos informativos de la consulta deben aparecer en el conjunto
    Categoria + Producto + Observaciones. Esta via complementa a la clasificacion
    por categorias para no perder filas como:
    semillas -> Vegetales -> Semillas germinadas listas para el consumo.
    """
    tokens_consulta = _tokenizar_busqueda(alimento)
    if not tokens_consulta:
        return df.iloc[0:0].copy()

    campos = ["Categoría", "Producto", "Observaciones"]

    def coincide_fila(fila):
        texto = " ".join(str(fila.get(campo, "") or "") for campo in campos)
        tokens_fila = set(_tokenizar_busqueda(texto))
        return all(token in tokens_fila for token in tokens_consulta)

    mask = df.apply(coincide_fila, axis=1)
    return df[mask].copy()


def combinar_candidatos(df_categorias: pd.DataFrame, df_texto: pd.DataFrame) -> pd.DataFrame:
    """Une ambas vias y elimina duplicados conservando el orden."""
    candidatos = pd.concat([df_categorias, df_texto], ignore_index=True)
    if candidatos.empty:
        return candidatos

    if "ID" in candidatos.columns:
        candidatos = candidatos.drop_duplicates(subset=["ID"], keep="first")
    else:
        candidatos = candidatos.drop_duplicates(keep="first")

    return candidatos.reset_index(drop=True)


# ==========================================================
# PASO 2: EVALUACION DE FILAS CANDIDATAS
# ==========================================================
async def evaluar_filas_filtradas(filas: list, alimento: str) -> list:
    """Evalua en batch las filas candidatas y devuelve los indices relevantes."""
    if not filas:
        return []

    client = openai.AsyncClient(api_key=PELIGROS_OPENAI_API_KEY)

    filas_texto = []
    for i, fila in enumerate(filas):
        partes = [
            f"Categoría: {fila.get('Categoría', '')}",
            f"Producto: {fila.get('Producto', '')}",
            f"Peligro: {fila.get('Peligro', '')}",
            f"Tipo: {fila.get('Tipo de peligro', '')}",
        ]
        obs = fila.get("Observaciones", "")
        if obs and str(obs).strip() and str(obs).strip().lower() != "nan":
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
                {"role": "user", "content": user_msg},
            ],
        )
        raw = completion.choices[0].message.content
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        indices = data.get("indices", [])
        return [i for i in indices if isinstance(i, int) and 0 <= i < len(filas)]
    except Exception as e:
        print(f"[Paso 2] Error evaluando filas: {e}")
        # Fallback conservador: ante un fallo del modelo, devolver los candidatos.
        return list(range(len(filas)))


# ==========================================================
# CONSTRUIR TABLA MARKDOWN
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
        limite = (
            f"{limpiar_texto(fila.get('Limite legal'))} + "
            f"{limpiar_texto(fila.get('n'))} + {limpiar_texto(fila.get('c'))}"
        )
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
# FUNCION PRINCIPAL: COMPROBAR PELIGROS
# ==========================================================
async def comprobar_peligros_alimento(alimento: str) -> str:
    """Busca peligros mediante dos vias de candidatos y una evaluacion final.

    1. Clasificacion semantica de categorias (1 llamada LLM).
    2. En paralelo conceptual, rescate textual deterministico sobre la base de datos.
    3. Union de candidatos y evaluacion final en batch (1 llamada LLM).

    La via textual evita que una mala clasificacion de categoria haga desaparecer
    una fila cuyo Producto contiene claramente lo buscado.
    """
    inicio = time.perf_counter()

    # 1. Leer datos (con cache)
    df, categorias = get_sheet_data()

    # 2. Via textual local (practicamente instantanea)
    df_texto = filtrar_filas_por_texto(df, alimento)

    # 3. Via semantica por categorias
    cats_relevantes = await clasificar_categorias(alimento, categorias)
    df_categorias = filtrar_filas_por_categorias(df, cats_relevantes)

    # 4. Union de candidatos. Ya no se aborta si Paso 1 no devuelve categorias:
    #    la via textual puede haber encontrado filas validas.
    df_filtrado = combinar_candidatos(df_categorias, df_texto)

    if df_filtrado.empty:
        print(
            f"[Peligros] alimento={alimento!r} categorias={len(cats_relevantes)} "
            f"filas_categoria={len(df_categorias)} filas_texto={len(df_texto)} candidatos=0"
        )
        return "**No se han encontrado peligros alimentarios registrados para dichas especificaciones.**"

    filas_filtradas = df_filtrado.to_dict("records")

    # 5. Evaluacion final en batch
    indices_relevantes = await evaluar_filas_filtradas(filas_filtradas, alimento)
    filas_finales = [filas_filtradas[i] for i in indices_relevantes]

    duracion = time.perf_counter() - inicio
    print(
        f"[Peligros] alimento={alimento!r} categorias={len(cats_relevantes)} "
        f"filas_categoria={len(df_categorias)} filas_texto={len(df_texto)} "
        f"candidatos={len(filas_filtradas)} finales={len(filas_finales)} "
        f"tiempo={duracion:.2f}s"
    )

    return construir_tabla_markdown(filas_finales, alimento)


# =========================
# AGENTE PRINCIPAL / ORQUESTADOR
# =========================
async def agente_principal(session: ChatSession, mensaje_usuario: str) -> dict:
    client = openai.AsyncClient(api_key=PELIGROS_OPENAI_API_KEY)
    session.add_user_message(mensaje_usuario)
    mensajes_llamada = session.clone_messages()

    completion = await client.chat.completions.create(
        model=MODELO_ORQUESTADOR,
        temperature=0,
        messages=mensajes_llamada,
    )

    raw = completion.choices[0].message.content
    session.add_assistant_message(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "type": "answer",
            "content": "No he podido interpretar tu pregunta. ¿Puedes repetirla de otra forma sencilla?",
        }

    return data


# =========================
# WEBSOCKET
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
                    "No he entendido tu peticion. Intenta preguntarme por un alimento concreto o una duda concreta.",
                    websocket,
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await manager.send_message(
            f"Ha ocurrido un error procesando tu solicitud: {str(e)}",
            websocket,
        )
        manager.disconnect(websocket)
