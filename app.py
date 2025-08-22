import os
import io
from typing import List, Dict

import streamlit as st
from PIL import Image, ImageDraw
import easyocr
from groq import Groq


# ==============================
# Configuración UI
# ==============================
st.set_page_config(page_title="OCR → LLM (Groq)", page_icon="🔎", layout="centered")
st.title("🔎 OCR → 💬 Explicación con LLM (Groq)")
st.caption("Sube una imagen con texto. Usamos EasyOCR para extraer el texto y Groq LLM para explicarlo.")


# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("⚙️ Ajustes")

    # API Key Groq
    api_key = st.text_input("GROQ_API_KEY", type="password")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    model_name = st.selectbox(
        "Modelo Groq",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-8b-8192"],
        index=0
    )

    # Idiomas OCR soportados por EasyOCR
    ocr_langs = st.multiselect(
        "Idiomas OCR (EasyOCR)",
        ["en", "es", "pt", "fr", "de"],
        default=["es"]
    )

    explain_lang = st.selectbox("Idioma de explicación", ["auto", "es", "en", "pt"], index=0)


# ==============================
# OCR con EasyOCR
# ==============================
@st.cache_resource
def load_ocr_reader(langs: List[str]):
    return easyocr.Reader(langs, gpu=False)


def run_easyocr(pil_img: Image.Image, langs: List[str]) -> (str, List[Dict]):
    reader = load_ocr_reader(langs)
    results = reader.readtext(np.array(pil_img))

    text = "\n".join([res[1] for res in results])
    boxes = [{"text": res[1], "bbox": res[0]} for res in results]
    return text, boxes


def draw_bboxes(original: Image.Image, boxes: List[Dict]) -> Image.Image:
    img = original.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for b in boxes:
        poly = [tuple(p) for p in b["bbox"]]
        draw.polygon(poly, outline=(0, 200, 0), width=2)
    return img


# ==============================
# Groq LLM
# ==============================
def explain_with_groq(text: str, model: str, lang: str = "auto") -> str:
    if not os.environ.get("GROQ_API_KEY"):
        return "⚠️ Falta GROQ_API_KEY. Configúralo en la barra lateral o en Secrets."

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    lang_instr = {
        "auto": "Usa el idioma más apropiado según el texto.",
        "es": "Responde en español.",
        "en": "Respond in English.",
        "pt": "Responda em português."
    }[lang]

    system_msg = (
        "Eres un asistente que explica texto detectado por OCR. "
        "Ofrece: resumen, propósito probable, público objetivo, contexto y acciones recomendadas."
    )

    user_msg = f"""{lang_instr}

TEXTO DETECTADO (OCR)
---
{text}
---

Responde con:
1) 📄 Resumen (2–4 líneas)
2) 🎯 Propósito probable
3) 👥 Público objetivo
4) 🧭 Contexto o dominio
5) ✅ Acciones recomendadas (bullets)
"""

    chat = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
    )
    return chat.choices[0].message.content


# ==============================
# Main
# ==============================
uploaded = st.file_uploader("Sube una imagen (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])

if uploaded:
    import numpy as np

    pil_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(pil_img, caption="Imagen cargada", use_column_width=True)

    with st.spinner("Ejecutando OCR (EasyOCR)…"):
        ocr_text, word_boxes = run_easyocr(pil_img, ocr_langs or ["es"])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📝 Texto extraído (OCR)")
        st.code(ocr_text or "(vacío)", language="markdown")
    with c2:
        st.subheader("🖼️ Palabras detectadas")
        st.image(draw_bboxes(pil_img, word_boxes), use_column_width=True)

    st.divider()
    with st.spinner("Generando explicación con Groq…"):
        explanation = explain_with_groq(ocr_text, model=model_name, lang=explain_lang)
    st.subheader("💬 Explicación del LLM")
    st.write(explanation)

    st.download_button(
        "⬇️ Descargar resultado (txt)",
        data=f"=== OCR TEXT ===\n{ocr_text}\n\n=== LLM EXPLANATION ===\n{explanation}",
        file_name="ocr_llm_result.txt",
        mime="text/plain"
    )
else:
    st.info("👉 Sube una imagen para comenzar.")
