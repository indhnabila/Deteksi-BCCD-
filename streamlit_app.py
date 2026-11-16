from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageOps
import torchvision.transforms.functional as TF
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead

# ============================================================
# KONFIGURASI
# ============================================================
st.set_page_config(
    page_title="Blood Cell Detection - SSD300-VGG16",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ssd_bccd_best.pth"
SUMMARY_PATH = BASE_DIR / "ssd_bccd_summary.json"  # opsional, hanya untuk nilai threshold awal

# Sesuai kelas pada eksperimen BCCD, bukan limfoblas/ALL.
CLASS_NAMES = ["Platelets", "RBC", "WBC"]
NUM_CLASSES = len(CLASS_NAMES) + 1  # background + 3 kelas
IMAGE_SIZE = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# STYLE: tampilan sederhana seperti contoh yang dikirim
# ============================================================
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 1.5rem; max-width: 1450px;}
      .app-header {
        background: #17365d; color: white; padding: 18px 14px;
        border-radius: 10px; text-align: center; margin-bottom: 18px;
      }
      .app-header h1 {font-size: 29px; margin: 0; font-weight: 750;}
      .app-header p {margin: 5px 0 0 0; color: #dbeafe; font-size: 14px;}
      .metric-card {
        background: #151515; border-radius: 12px; padding: 13px 16px;
        min-height: 86px; margin-bottom: 8px;
      }
      .metric-label {font-size: 14px; color: #d1d5db; margin-bottom: 4px;}
      .metric-value {font-size: 27px; font-weight: 700; color: white;}
      .notice {
        background: #3b2500; color: #ffcf66; padding: 12px 15px;
        border-radius: 10px; margin-top: 14px; font-size: 14px;
      }
      div[data-testid="stFileUploader"] {border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MEMUAT MODEL
# ============================================================
@st.cache_resource(show_spinner="Memuat model SSD300-VGG16...")
def load_ssd_model(model_path: str, device_name: str):
    device = torch.device(device_name)
    model = ssd300_vgg16(weights=None, weights_backbone=None)

    in_channels = [
        module.in_channels
        for module in model.head.classification_head.module_list
    ]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=NUM_CLASSES,
    )

    try:
        checkpoint = torch.load(
            model_path, map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_default_threshold() -> float:
    """Ambil threshold dari JSON bila ada; jika tidak, gunakan 0.50."""
    import json
    if SUMMARY_PATH.exists():
        try:
            with SUMMARY_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            value = float(data.get("confidence_threshold", 0.50))
            return min(0.95, max(0.05, value))
        except Exception:
            pass
    return 0.50


def run_detection(image: Image.Image, threshold: float):
    model = load_ssd_model(str(MODEL_PATH), str(DEVICE))

    # Samakan ukuran input dengan model SSD300 yang digunakan saat training.
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_tensor = TF.to_tensor(resized).to(DEVICE)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        prediction = model([image_tensor])[0]
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Gambar bounding box pada ukuran citra asli.
    result_image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result_image)
    width, height = result_image.size
    scale_x, scale_y = width / IMAGE_SIZE, height / IMAGE_SIZE

    colors = {
        "Platelets": (235, 105, 75),
        "RBC": (40, 150, 230),
        "WBC": (25, 180, 105),
    }
    counts = {name: 0 for name in CLASS_NAMES}
    rows = []

    boxes = prediction["boxes"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    scores = prediction["scores"].detach().cpu()

    for box, label, score in zip(boxes, labels, scores):
        score_value = float(score.item())
        class_id = int(label.item())
        if score_value < threshold or not (1 <= class_id <= len(CLASS_NAMES)):
            continue

        class_name = CLASS_NAMES[class_id - 1]
        counts[class_name] += 1
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        x1 = max(0, min(width - 1, int(x1 * scale_x)))
        y1 = max(0, min(height - 1, int(y1 * scale_y)))
        x2 = max(0, min(width - 1, int(x2 * scale_x)))
        y2 = max(0, min(height - 1, int(y2 * scale_y)))

        color = colors[class_name]
        line_width = max(2, min(5, width // 350))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label_text = f"{class_name} {score_value:.2f}"
        text_y = max(0, y1 - 19)
        text_bbox = draw.textbbox((x1, text_y), label_text)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, text_y), label_text, fill="white")

        rows.append({
            "Kelas": class_name,
            "Confidence": round(score_value, 4),
            "x_min": x1,
            "y_min": y1,
            "x_max": x2,
            "y_max": y2,
        })

    return result_image, counts, pd.DataFrame(rows), elapsed


def metric_card(label: str, value: str):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="app-header"><h1>🩸 BLOOD CELL DETECTION</h1>'
    '<p>SSD300-VGG16 | BCCD Dataset</p></div>',
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR: HANYA PENGATURAN DETEKSI
# ============================================================
with st.sidebar:
    st.markdown("## Pengaturan")
    uploaded_file = st.file_uploader(
        "Unggah citra sel darah",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Gunakan citra sel darah yang sesuai dengan kelas pada dataset BCCD.",
    )
    confidence = st.slider(
        "Ambang skor keyakinan",
        min_value=0.05,
        max_value=0.95,
        value=get_default_threshold(),
        step=0.05,
        format="%.2f",
    )
    detect_button = st.button("🔍 Deteksi", type="primary", use_container_width=True)
    st.caption(f"Perangkat: {DEVICE}")

# ============================================================
# KONTEN UTAMA: DETEKSI SAJA, TANPA TAB EVALUASI
# ============================================================
st.subheader("Deteksi dan Penghitungan Sel Darah")

if not MODEL_PATH.exists():
    st.error(
        f"File model `{MODEL_PATH.name}` tidak ditemukan. "
        "Letakkan ssd_bccd_best.pth di folder yang sama dengan streamlit_detection_only.py."
    )
    st.stop()

if uploaded_file is None:
    st.info("Unggah citra sel darah melalui panel Pengaturan di sebelah kiri untuk memulai.")
else:
    try:
        original = Image.open(uploaded_file)
        original = ImageOps.exif_transpose(original).convert("RGB")

        input_col, result_col = st.columns(2, gap="large")
        with input_col:
            st.markdown("**Citra masukan**")
            st.image(original, use_container_width=True)

        if detect_button:
            try:
                with st.spinner("Model sedang mendeteksi sel darah..."):
                    annotated, counts, detections_df, elapsed = run_detection(
                        original, confidence
                    )
                st.session_state["ssd_result"] = {
                    "image": annotated,
                    "counts": counts,
                    "detections": detections_df,
                    "elapsed": elapsed,
                    "threshold": confidence,
                }
            except Exception as exc:
                st.error(f"Deteksi gagal: {exc}")
                st.stop()

        result = st.session_state.get("ssd_result")
        if result is not None:
            with result_col:
                st.markdown("**Citra hasil**")
                st.image(result["image"], use_container_width=True)

            total = sum(result["counts"].values())
            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Total sel terdeteksi", str(total))
            with c2:
                metric_card("Ambang skor", f'{result["threshold"]:.2f}')
            with c3:
                metric_card("Waktu proses", f'{result["elapsed"]:.2f} detik')

            class_cols = st.columns(len(CLASS_NAMES))
            for col, class_name in zip(class_cols, CLASS_NAMES):
                with col:
                    st.metric(f"Jumlah {class_name}", result["counts"][class_name])

            image_buffer = io.BytesIO()
            result["image"].save(image_buffer, format="PNG")
            csv_data = result["detections"].to_csv(index=False).encode("utf-8-sig")

            b1, b2 = st.columns(2)
            with b1:
                st.download_button(
                    "⬇️ Simpan citra hasil",
                    data=image_buffer.getvalue(),
                    file_name="hasil_deteksi_sel_darah.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with b2:
                st.download_button(
                    "📄 Simpan ringkasan (CSV)",
                    data=csv_data,
                    file_name="ringkasan_deteksi_sel_darah.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with st.expander("Lihat detail objek yang terdeteksi"):
                if result["detections"].empty:
                    st.write("Tidak ada objek yang melewati ambang skor.")
                else:
                    st.dataframe(result["detections"], use_container_width=True, hide_index=True)
        else:
            with result_col:
                st.markdown("**Citra hasil**")
                st.info("Tekan tombol Deteksi untuk menampilkan hasil.")

    except Exception as exc:
        st.error(f"Citra tidak dapat dibuka: {exc}")

st.markdown(
    '<div class="notice">ⓘ Hasil deteksi merupakan alat bantu analisis citra, '
    'bukan diagnosis medis.</div>',
    unsafe_allow_html=True,
)

st.caption("Blood Cell Detection | SSD300-VGG16 | BCCD Dataset")
