import os
import io
from typing import Tuple, List, Dict

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import pytesseract

from groq import Groq


# ----------------------------
# UI CONFIG
# ----------------------------
st.set_page_config(
    page_title="OCR → LLM Explainer (Groq)",
    page_icon="🔎",
    layout="centered"
)

st.title("🔎 OCR → 💬 LLM Explainer (Groq)")
st.write(
    "Sube una imagen con texto. El sistema hará OCR (Tesseract) y un LLM de Groq explicará el contenido."
)


# ----------------------------
# SIDEBAR: Settings
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    # GROQ API KEY
    api_key_input = st.text_input(
        "GROQ_API_KEY",
        type="password",
        help="Pon tu API key de Groq aquí o configúrala como variable de entorno."
    )
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input

    # Model selector (safe defaults that exist on Groq)
    model_name = st.selectbox(
        "Groq model",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-8b-8192",   # legacy but often still available
        ],
        index=0,
        help="Modelos Groq recomendados para buen balance entre calidad y latencia."
    )

    # OCR language
    ocr_lang = st.selectbox(
        "OCR language (Tesseract)",
        options=["spa", "eng", "por", "fra", "deu"],
        index=0,
        help="Asegúrate de tener instalado el paquete de idioma Tesseract correspondiente."
    )

    # Preprocessing
    do_denoise = st.checkbox("Denoise (mediana)", value=True)
    do_threshold = st.checkbox("Adaptive threshold", value=True)

    # LLM output language
    explain_lang = st.selectbox(
        "Explicación en:",
        options=["auto", "es", "en", "pt"],
        index=0
    )


# ----------------------------
# OCR Pipeline
# ----------------------------
def preprocess_for_ocr(pil_img: Image.Image,
                       denoise: bool = True,
                       threshold: bool = True) -> np.ndarray:
    """Return OpenCV BGR image optimized for OCR."""
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if denoise:
        gray = cv2.medianBlur(gray, 3)
    if threshold:
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 15
        )
    return gray


def run_tesseract_ocr(pil_img: Image.Image, lang: str = "spa") -> Tuple[str, List[Dict]]:
    """
    Returns: (full_text, boxes)
    - full_text: concatenated extracted text
    - boxes: list of dicts with bbox + word + conf
    """
    pre = preprocess_for_ocr(pil_img, denoise=do_denoise, threshold=do_threshold)

    # text
    cfg = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(pre, lang=lang, config=cfg)

    # boxes (word-level)
    data = pytesseract.image_to_data(pre, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)

    boxes = []
    for i in range(len(data["text"])):
        if int(data["conf"][i]) > 0:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            boxes.append({
                "text": data["text"][i],
                "conf": float(data["conf"][i]),
                "bbox": (x, y, x + w, y + h)
            })

    return text.strip(), boxes


def draw_bboxes(original: Image.Image, boxes: List[Dict]) -> Image.Image:
    """Draw green boxes around detected words."""
    img = original.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for b in boxes:
        draw.rectangle(b["bbox"], outline=(0, 200, 0), width=2)
    return img


# ----------------------------
# LLM (Groq) Call
# ----------------------------
def explain_with_groq(extracted_text: str,
                      model: str = "llama-3.3-70b-versatile",
                      lang: str = "auto") -> str:
    """
    Ask a Groq LLM to explain the extracted text clearly.
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    lang_instr = {
        "auto": "Usa el idioma que creas más apropiado según el texto, preferiblemente español si hay señales.",
        "es": "Responde en español.",
        "en": "Respond in English.",
        "pt": "Responda em português."
    }[lang]

    system_msg = (
        "Eres un asistente que explica texto detectado por OCR. "
        "Da contexto, propósito probable, público objetivo, tono, y acciones recomendadas. "
        "Si el texto está incompleto, señala lagunas. Sé claro y conciso."
    )

    user_msg = f"""{lang_instr}

TEXTO DETECTADO (OCR):
---
{extracted_text}
---

Entrega tu respuesta con esta estructura:
1) 📄 Resumen (2-4 líneas)
2) 🎯 Propósito probable
3) 👥 Público objetivo
4) 🧭 Contexto o dominio
5) ✅ Acciones recomendadas (bullets)
6) ⚠️ Señales de baja calidad OCR (si aplica)
"""

    try:
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        return chat.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error al llamar al LLM: {e}"


# ----------------------------
# MAIN APP
# ----------------------------
uploaded = st.file_uploader(
    "Sube una imagen (PNG/JPG/JPEG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded is not None:
    try:
        pil_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        st.image(pil_img, caption="Imagen cargada", use_column_width=True)

        with st.spinner("Ejecutando OCR..."):
            text, boxes = run_tesseract_ocr(pil_img, lang=ocr_lang)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📝 Texto extraído (OCR)")
            st.code(text or "(vacío)", language="markdown")
        with col2:
            st.subheader("🖼️ Detecciones")
            st.caption("Cajas de palabras detectadas por Tesseract")
            st.image(draw_bboxes(pil_img, boxes), use_column_width=True)

        st.divider()

        if os.environ.get("GROQ_API_KEY") is None:
            st.warning("Define GROQ_API_KEY (en la barra lateral o como variable de entorno) para usar el LLM.")
        else:
            with st.spinner("Pidiendo explicación al LLM (Groq)…"):
                explanation = explain_with_groq(text, model=model_name, lang=explain_lang)
            st.subheader("💬 Explicación del LLM")
            st.write(explanation)

            # Download JSON-ish (simple)
            st.download_button(
                "⬇️ Descargar resultado (txt)",
                data=f"=== OCR TEXT ===\n{text}\n\n=== LLM EXPLANATION ===\n{explanation}",
                file_name="ocr_llm_result.txt",
                mime="text/plain"
            )

    except Exception as e:
        st.error(f"Ocurrió un error procesando la imagen: {e}")
else:
    st.info("👉 Sube una imagen para comenzar.")
