import streamlit as st
import os
import hashlib
import chromadb
import google.generativeai as genai
import pandas as pd
import io

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(page_title="Super Data Assistant", layout="wide")

# Carga variables de entorno
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Modelo de embeddings (solo se usa para PDF)
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Inicializa ChromaDB
client = chromadb.Client()

# ============================================================
# 2. SESSION STATE (Memoria de la app)
# ============================================================
if "collection" not in st.session_state:
    st.session_state.collection = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

# ============================================================
# 3. FUNCIONES PARA DATOS (CSV / EXCEL) 📊
# ============================================================
def load_data(file, separator):
    """Carga CSV usando el separador elegido o Excel automáticamente"""
    filename = file.name.lower()
    
    try:
        if filename.endswith('.csv'):
            # seek(0) es vital para asegurar que leemos desde el principio
            file.seek(0)
            return pd.read_csv(file, sep=separator)
            
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            return pd.read_excel(file)
            
    except Exception as e:
        st.error(f"Error cargando archivo: {e}")
        return None

def analyze_outliers(df):
    """Detecta outliers usando el método IQR (Rango Intercuartil)"""
    numeric_cols = df.select_dtypes(include=['number'])
    outlier_summary = {}
    
    for col in numeric_cols.columns:
        series = df[col].dropna()
        if len(series) > 0:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            if len(outliers) > 0:
                outlier_summary[col] = len(outliers)
            
    return outlier_summary

def get_gemini_data_insight(df_info_str, stats_str, outliers_str):
    """Pide a Gemini que interprete los datos estadísticos"""
    model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
    prompt = f"""
    Actúa como un Científico de Datos Senior realizando una auditoría.
    Analiza estas métricas del dataset cargado:

    1. ESTRUCTURA DEL DATASET:
    {df_info_str}

    2. ESTADÍSTICAS DESCRIPTIVAS:
    {stats_str}
    
    3. OUTLIERS (Valores Atípicos):
    {outliers_str}

    TU MISIÓN:
    - Escribe un resumen ejecutivo sobre la calidad de los datos.
    - Menciona si hay problemas graves (muchos nulos, outliers sospechosos).
    - Sugiere 2 pasos de limpieza recomendados.
    - Usa formato Markdown limpio y profesional.
    """
    response = model.generate_content(prompt)
    return response.text

# ============================================================
# 4. FUNCIONES PARA PDF (RAG + CHAT) 📄
# ============================================================
def hash_file(file) -> str:
    return hashlib.sha256(file.getvalue()).hexdigest()

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for i, page in enumerate(reader.pages):
        content = page.extract_text()
        if content:
            text += f"\n[Página {i+1}]\n{content}"
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        chunk_text = text[start:start + chunk_size]
        chunks.append({
            "id": f"chunk_{chunk_id}",
            "content": chunk_text,
            "start_index": start,
            "size": len(chunk_text)
        })
        chunk_id += 1
        start += chunk_size - overlap
    return chunks

def create_chroma_collection(chunks):
    try:
        client.delete_collection("pdf_rag")
    except:
        pass
    collection = client.create_collection(name="pdf_rag")
    texts = [c["content"] for c in chunks]
    embeddings = EMBEDDING_MODEL.encode(texts)
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=[c["id"] for c in chunks],
        metadatas=[{"chunk_index": i, "start_index": c["start_index"], "chunk_size": c["size"]} for i, c in enumerate(chunks)]
    )
    return collection

def retrieve_context(collection, query, k=4):
    query_embedding = EMBEDDING_MODEL.encode([query])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=k)
    return results

# ============================================================
# 5. INTERFAZ GRÁFICA (FRONTEND)
# ============================================================

st.title("🤖 Asistente de Datos Pro (PDF + Excel + CSV)")
st.markdown("Sube un archivo para comenzar: **PDF** (Chat) o **Excel/CSV** (Auditoría Automática).")

uploaded_file = st.file_uploader("Cargar archivo aquí", type=["pdf", "csv", "xlsx"])

if uploaded_file:
    # Detectar extensión
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    # Resetear estado si cambia el archivo
    current_hash = hash_file(uploaded_file)
    if st.session_state.file_hash != current_hash:
        st.session_state.file_hash = current_hash
        st.session_state.pdf_processed = False
        st.session_state.collection = None

    # =======================================================
    # 🅰️ MODO ANALISTA: CSV O EXCEL
    # =======================================================
    if file_ext in ['csv', 'xlsx']:
        
        # --- CONFIGURACIÓN DE SEPARADOR (Solo CSV) ---
        sep = ',' # Default
        if file_ext == 'csv':
            col_sep, col_info = st.columns([1, 3])
            with col_sep:
                option = st.selectbox(
                    "🛠️ Separador de columnas:",
                    ("Coma (,)", "Punto y coma (;)", "Tabulación (Tab)", "Barra (|)"),
                    index=0
                )
                if "Coma" in option: sep = ','
                elif "Punto y coma" in option: sep = ';'
                elif "Tab" in option: sep = '\t'
                elif "Barra" in option: sep = '|'
        
        # --- CARGA DE DATOS ---
        df = load_data(uploaded_file, sep)
        
        if df is not None:
            # Validación rápida de carga
            st.success(f"✅ Archivo cargado correctamente: {df.shape[0]} filas, {df.shape[1]} columnas.")
            
            # Advertencia si se ve mal (1 sola columna suele ser error de separador)
            if df.shape[1] == 1 and file_ext == 'csv':
                st.warning("⚠️ ¡Atención! Se detectó solo 1 columna. ¿Quizás el separador es incorrecto? Intenta cambiarlo arriba.")

            # --- VISTA PREVIA ---
            with st.expander("👀 Ver Datos (Primeras 5 filas)", expanded=True):
                st.dataframe(df.head())

            st.divider()

            # --- DASHBOARD DE SALUD ---
            st.subheader("1. 🏥 Salud de los Datos")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Filas Totales", df.shape[0])
            col2.metric("Variables", df.shape[1])
            col3.metric("Filas Duplicadas", df.duplicated().sum())
            total_nulos = df.isnull().sum().sum()
            col4.metric("Datos Faltantes (Nulos)", total_nulos)

            if total_nulos > 0:
                st.caption("Gráfica de nulos por columna:")
                st.bar_chart(df.isnull().sum())

            # --- ESTADÍSTICAS AVANZADAS ---
            st.subheader("2. 📈 Perfilamiento Estadístico")
            tab1, tab2 = st.tabs(["Estadística Descriptiva", "Detectar Outliers"])
            
            with tab1:
                st.dataframe(df.describe().T) # Transpuesta para leer mejor
            
            with tab2:
                outliers = analyze_outliers(df)
                if outliers:
                    st.error(f"🚨 Se detectaron valores atípicos en {len(outliers)} columnas.")
                    st.write("Conteo de valores fuera del rango normal (Método IQR):")
                    st.json(outliers)
                else:
                    st.success("✅ No se detectaron outliers estadísticos evidentes.")

            # --- GEMINI INSIGHTS ---
            st.divider()
            st.subheader("3. 🧠 Diagnóstico Inteligente (IA)")
            st.markdown("Deja que Gemini analice los números y te dé una opinión experta.")
            
            if st.button("✨ Generar Reporte con Gemini"):
                with st.spinner("Analizando patrones en los datos..."):
                    # Preparamos la info técnica para enviarla al prompt
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    info_str = buffer.getvalue()
                    
                    insight = get_gemini_data_insight(
                        df_info_str=info_str,
                        stats_str=df.describe().to_string(),
                        outliers_str=str(outliers)
                    )
                    st.markdown(insight)

    # =======================================================
    # 🅱️ MODO CHAT: PDF
    # =======================================================
    elif file_ext == 'pdf':
        st.info("📂 Archivo PDF detectado. Modo Chat RAG activado.")
        
        # Botón para procesar (Indexar)
        if not st.session_state.pdf_processed:
            if st.button("📥 Procesar Documento"):
                with st.spinner("Leyendo y vectorizando el PDF..."):
                    text = extract_text_from_pdf(uploaded_file)
                    chunks = chunk_text(text)
                    st.session_state.collection = create_chroma_collection(chunks)
                    st.session_state.pdf_processed = True
                st.success("¡Documento listo! Ya puedes chatear con él.")

        # Área de Chat
        if st.session_state.pdf_processed:
            question = st.chat_input("Pregunta algo sobre el documento...")
            
            if question:
                # 1. Mostrar pregunta usuario
                with st.chat_message("user"):
                    st.write(question)

                # 2. Generar respuesta
                with st.spinner("Buscando en el documento..."):
                    results = retrieve_context(st.session_state.collection, question)
                    context_text = "\n\n".join(results["documents"][0])
                    
                    # Llamada a Gemini
                    model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
                    prompt_rag = f"""
                    Responde a la pregunta basándote ÚNICAMENTE en el siguiente contexto extraído del PDF.
                    Si la respuesta no está en el texto, indícalo claramente.
                    
                    CONTEXTO:
                    {context_text}
                    
                    PREGUNTA: {question}
                    """
                    response = model.generate_content(prompt_rag)
                
                