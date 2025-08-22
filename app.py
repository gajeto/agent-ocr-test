import os
import io
import shutil
from typing import Tuple, List, Dict

import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np
import pytesseract
from groq import Groq


# ==============================
# Configuración de página / UI
# ==============================
st.set_page_config(page_title="OCR → LLM (Groq)", page_icon="🔎", layout="centered")
st.title("🔎 OCR → 💬 Explicación con LLM (Groq)")
st.caption("Sube una imagen con texto. Hacemos OCR y luego un LLM de Groq te sugiere una explicación clara y accionable.")


# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("⚙️ Ajustes")

    # API key (puedes pegarla aquí o usar Secrets/ENV)
    api_key = st.text_input("GROQ_API_KEY", type="password",
                            help="Pon aquí tu API key o configúrala como variable de entorno/Secret.")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    model_name = st.selectbox(
        "Modelo Groq",
        [
            "llama-3.3-70b-versatile",   # alta calidad
            "llama-3.1-8b-instant",     # más rápido
            "llama3-8b-8192",           # legado (suele estar)
        ],
        index=0
    )

    ocr_lang = st.selectbox(
        "Idioma OCR (Tesseract)",
        ["spa", "eng", "por", "fra", "deu"],
        index=0,
        help="Asegúrate de tener instalado el paquete de idioma correspondiente."
    )

    explain_lang = st.selectbox(
        "Idioma de explicación",
        ["auto", "es", "en", "pt"],
        index=0
    )

    st.markdown("---")
    st.caption("Preprocesamiento (mantenemos la lógica de OCR simple: gris + filtro + umbral).")
    use_denoise = st.checkbox("Denoise (Mediana 3x3)", value=True)
    use_threshold = st.checkbox("Threshold global", value=True)
    thr_method = st.selectbox("Método de umbral", ["mean", "percentile_80"], index=0)

    # Ruta específica Tesseract (Windows local)
    tesseract_cmd = st.text_input("Ruta Tesseract (Windows, opcional)")
    if tesseract_cmd.strip():
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd.strip()


# =========================================
# Comprobaciones de entorno (útil en Cloud)
# =========================================
with st.expander("🔧 Comprobaciones (diagnóstico rápido)"):
    st.write("pytesseract versión:", getattr(pytesseract, "__version__", "desconocida"))
    tess_path = shutil.which("tesseract")
    if not tess_path:
        st.error(
            "No se encontró el ejecutable de **Tesseract**. "
            "En Streamlit Cloud debes incluir `packages.txt` con:\n"
            "`tesseract-ocr`, `tesseract-ocr-eng`, `tesseract-ocr-spa` (y los que necesites)."
        )
    else:
        st.success(f"Tesseract encontrado en: {tess_path}")


# ==============================
# OCR helpers (misma lógica base)
# ==============================
def _global_threshold(arr: np.ndarray, method: str = "mean") -> np.ndarray:
    if method == "percentile_80":
        thr = np.percentile(arr, 80)
    else:
        thr = arr.mean()
    return (arr > thr).astype(np.uint8) * 255


def preprocess_for_ocr(pil_img: Image.Image,
                       denoise: bool = True,
                       threshold: bool = True,
                       method: str = "mean") -> Image.Image:
    """
    Mantiene una lógica sencilla de pre-procesamiento:
    1) Gris
    2) (Opcional) Filtro mediana
    3) (Opcional) Umbral global (mean/percentile_80)
    4) Autocontraste ligero
    """
    gray = pil_img.convert("L")
    if denoise:
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
    if threshold:
        arr = np.array(gray)
        bin_arr = _global_threshold(arr, method=method)
        gray = Image.fromarray(bin_arr)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    return gray


def run_tesseract_ocr(pil_img: Image.Image, lang: str = "spa") -> Tuple[str, List[Dict]]:
    """
    Devuelve: (texto, boxes)
    boxes: [{text, conf, bbox=(x1,y1,x2,y2)}]
    """
    pre = preprocess_for_ocr(
        pil_img,
        denoise=use_denoise,
        threshold=use_threshold,
        method=thr_method
    )

    # Config "estándar" para bloques de texto
    cfg = "--oem 3 --psm 6"

    # Texto
    text = pytesseract.image_to_string(pre, lang=lang, config=cfg)

    # Boxes por palabra
    data = pytesseract.image_to_data(pre, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
    boxes = []
    for i in range(len(data["text"])):
        try:
            conf_i = float(data["conf"][i])
        except ValueError:
            conf_i = -1.0
        if conf_i > 0:
            x, y = data["left"][i], data["top"][i]
            w, h = data["width"][i], data["height"][i]
            boxes.append({"text": data["text"][i], "conf": conf_i, "bbox": (x, y, x + w, y + h)})
    return (text or "").strip(), boxes


def draw_bboxes(original: Image.Image, boxes: List[Dict]) -> Image.Image:
    img = original.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for b in boxes:
        draw.rectangle(b["bbox"], outline=(0, 200, 0), width=2)
    return img


# ==============================
# Groq LLM
# ==============================
def explain_with_groq(text: str, model: str, lang: str = "auto") -> str:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    if not os.environ.get("GROQ_API_KEY"):
        return "⚠️ Falta GROQ_API_KEY. Configura la clave en la barra lateral o en los Secrets."

    lang_instr = {
        "auto": "Usa el idioma más apropiado según el texto; prioriza español si hay señales.",
        "es": "Responde en español.",
        "en": "Respond in English.",
        "pt": "Responda em português."
    }[lang]

    system_msg = (
        "Eres un asistente que explica texto detectado por OCR. "
        "Ofrece: resumen, propósito probable, público objetivo, contexto, acciones recomendadas. "
        "Si el texto está incompleto o ruidoso, indícalo."
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
6) ⚠️ Observaciones sobre calidad del OCR (si aplica)
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
    try:
        pil_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        st.image(pil_img, caption="Imagen cargada", use_column_width=True)

        with st.spinner("Ejecutando OCR…"):
            ocr_text, word_boxes = run_tesseract_ocr(pil_img, lang=ocr_lang)

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
    except Exception as e:
        st.error(f"Ocurrió un error procesando la imagen: {e}")
else:
    st.info("👉 Sube una imagen para comenzar.")
