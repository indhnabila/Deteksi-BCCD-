from __future__ import annotations

import io
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageOps
import torchvision.transforms.functional as TF
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Blood Cell Detection - SSD300-VGG16",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ssd_bccd_best.pth"
DB_PATH = BASE_DIR / "detection_history.db"
SETTINGS_PATH = BASE_DIR / "app_settings.json"
HISTORY_IMAGE_DIR = BASE_DIR / "history_images"
HISTORY_IMAGE_DIR.mkdir(exist_ok=True)

# Match the actual trained BCCD model. Do not rename these to lymphoblast/ALL.
CLASS_NAMES = ["Platelets", "RBC", "WBC"]
NUM_CLASSES = len(CLASS_NAMES) + 1  # background + 3 object classes
IMAGE_SIZE = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLOR_OPTIONS = {
    "Hijau": (25, 180, 105),
    "Merah": (235, 75, 75),
    "Biru": (40, 125, 230),
    "Kuning": (230, 175, 30),
}

DEFAULT_SETTINGS = {
    "confidence_threshold": 0.60,
    "box_color": "Hijau",
    "show_score": True,
    "auto_save_history": True,
}

# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.15rem; padding-bottom: 1.5rem; max-width: 1450px;}
      .app-header {background:#17365d;color:white;padding:18px 14px;border-radius:10px;text-align:center;margin-bottom:18px;}
      .app-header h1 {font-size:29px;margin:0;font-weight:750;}
      .app-header p {margin:5px 0 0 0;color:#dbeafe;font-size:14px;}
      .section-note {color:#64748b;font-size:0.9rem;}
      .warning-note {background:#3b2500;color:#ffcf66;padding:12px 15px;border-radius:10px;margin-top:14px;font-size:14px;}
      div[data-testid="stFileUploader"] {border-radius:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SETTINGS STORAGE
# ============================================================
def load_settings() -> dict:
    settings = DEFAULT_SETTINGS.copy()
    if SETTINGS_PATH.exists():
        try:
            stored = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                settings.update(stored)
        except (OSError, json.JSONDecodeError):
            pass
    if settings.get("box_color") not in COLOR_OPTIONS:
        settings["box_color"] = DEFAULT_SETTINGS["box_color"]
    try:
        settings["confidence_threshold"] = min(
            0.95, max(0.05, float(settings["confidence_threshold"]))
        )
    except (TypeError, ValueError):
        settings["confidence_threshold"] = DEFAULT_SETTINGS["confidence_threshold"]
    settings["show_score"] = bool(settings.get("show_score", True))
    settings["auto_save_history"] = bool(settings.get("auto_save_history", True))
    return settings


def save_settings(settings: dict) -> None:
    SETTINGS_PATH.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

# ============================================================
# HISTORY DATABASE (local SQLite for local prototype)
# ============================================================
def db_connect():
    return sqlite3.connect(DB_PATH)


def init_history_db() -> None:
    with db_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT NOT NULL,
                image_name TEXT NOT NULL,
                detected_at TEXT NOT NULL,
                total_objects INTEGER NOT NULL,
                platelets_count INTEGER NOT NULL,
                rbc_count INTEGER NOT NULL,
                wbc_count INTEGER NOT NULL,
                confidence_threshold REAL NOT NULL,
                processing_seconds REAL NOT NULL,
                result_image TEXT,
                detections_json TEXT
            )
            """
        )
        conn.commit()


def add_history(record: dict) -> int:
    with db_connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO detection_history (
                sample_id, image_name, detected_at, total_objects,
                platelets_count, rbc_count, wbc_count,
                confidence_threshold, processing_seconds,
                result_image, detections_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["sample_id"], record["image_name"], record["detected_at"],
                record["total_objects"], record["platelets_count"],
                record["rbc_count"], record["wbc_count"],
                record["confidence_threshold"], record["processing_seconds"],
                record.get("result_image"), record.get("detections_json", "[]"),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_history(search_text: str = "") -> pd.DataFrame:
    query = "SELECT * FROM detection_history"
    params: tuple = ()
    if search_text.strip():
        query += " WHERE sample_id LIKE ? OR image_name LIKE ?"
        like = f"%{search_text.strip()}%"
        params = (like, like)
    query += " ORDER BY id DESC"
    with db_connect() as conn:
        return pd.read_sql_query(query, conn, params=params)


def delete_history_record(record_id: int) -> None:
    image_path = None
    with db_connect() as conn:
        row = conn.execute(
            "SELECT result_image FROM detection_history WHERE id = ?",
            (record_id,),
        ).fetchone()
        if row:
            image_path = row[0]
        conn.execute("DELETE FROM detection_history WHERE id = ?", (record_id,))
        conn.commit()
    if image_path:
        try:
            Path(image_path).unlink(missing_ok=True)
        except OSError:
            pass


def clear_all_history() -> None:
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT result_image FROM detection_history"
        ).fetchall()
        conn.execute("DELETE FROM detection_history")
        conn.commit()
    for row in rows:
        if row and row[0]:
            try:
                Path(row[0]).unlink(missing_ok=True)
            except OSError:
                pass


init_history_db()

# ============================================================
# MODEL LOADER
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


def run_detection(
    image: Image.Image,
    threshold: float,
    box_color: str,
    show_score: bool,
):
    model = load_ssd_model(str(MODEL_PATH), str(DEVICE))
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

    result_image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result_image)
    width, height = result_image.size
    scale_x, scale_y = width / IMAGE_SIZE, height / IMAGE_SIZE
    rgb = COLOR_OPTIONS.get(box_color, COLOR_OPTIONS["Hijau"])
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
        line_width = max(2, min(5, width // 350))
        draw.rectangle((x1, y1, x2, y2), outline=rgb, width=line_width)
        label_text = f"{class_name} {score_value:.2f}" if show_score else class_name
        text_y = max(0, y1 - 19)
        text_bbox = draw.textbbox((x1, text_y), label_text)
        draw.rectangle(text_bbox, fill=rgb)
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

# ============================================================
# SMALL UI HELPERS
# ============================================================
def metric_card(label: str, value: str):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def show_header():
    st.markdown(
        '<div class="app-header"><h1>🩸 BLOOD CELL DETECTION</h1>'
        '<p>SSD300-VGG16 | BCCD Dataset</p></div>',
        unsafe_allow_html=True,
    )


def show_detection_page(settings: dict):
    st.subheader("Deteksi dan Penghitungan Sel Darah")
    st.caption("Kelas model: Platelets, RBC, dan WBC. Unggah satu citra untuk memulai.")

    if not MODEL_PATH.exists():
        st.error(
            f"File model `{MODEL_PATH.name}` tidak ditemukan. "
            "Letakkan checkpoint di folder yang sama dengan streamlit_app.py."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Unggah citra sel darah", type=["jpg", "jpeg", "png", "bmp"],
        help="Gunakan citra yang sesuai dengan kelas dataset BCCD.",
    )
    sample_id_input = st.text_input(
        "ID sampel (opsional)",
        placeholder="Contoh: BCCD-001 (kosongkan untuk ID otomatis)",
        help="Gunakan kode sampel, bukan nama atau data pribadi pasien.",
    )

    if uploaded_file is None:
        st.info("Unggah citra melalui area di atas untuk melihat citra masukan dan hasil deteksi.")
        return

    try:
        image_bytes = uploaded_file.getvalue()
        original = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    except Exception as exc:
        st.error(f"Citra tidak dapat dibuka: {exc}")
        return

    input_col, result_col = st.columns(2, gap="large")
    with input_col:
        st.markdown("**Citra masukan**")
        st.image(original, use_container_width=True)

    with result_col:
        st.markdown("**Citra hasil**")
        current_result = st.session_state.get("latest_detection")
        if current_result and current_result.get("image_name") == uploaded_file.name:
            st.image(current_result["annotated_image"], use_container_width=True)
        else:
            st.info("Tekan tombol Deteksi untuk menampilkan hasil.")

    if st.button("🔍 Deteksi", type="primary", use_container_width=True):
        try:
            with st.spinner("Model sedang mendeteksi sel darah..."):
                annotated, counts, detections_df, elapsed = run_detection(
                    original,
                    float(settings["confidence_threshold"]),
                    settings["box_color"],
                    bool(settings["show_score"]),
                )

            sample_id = sample_id_input.strip() or datetime.now().strftime("SMP-%Y%m%d-%H%M%S")
            detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            total = int(sum(counts.values()))
            saved_image_path = None

            if settings["auto_save_history"]:
                image_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
                saved_image_path = HISTORY_IMAGE_DIR / image_filename
                annotated.save(saved_image_path, format="PNG")

                record = {
                    "sample_id": sample_id,
                    "image_name": uploaded_file.name,
                    "detected_at": detected_at,
                    "total_objects": total,
                    "platelets_count": counts["Platelets"],
                    "rbc_count": counts["RBC"],
                    "wbc_count": counts["WBC"],
                    "confidence_threshold": float(settings["confidence_threshold"]),
                    "processing_seconds": float(elapsed),
                    "result_image": str(saved_image_path),
                    "detections_json": detections_df.to_json(orient="records"),
                }
                add_history(record)

            st.session_state["latest_detection"] = {
                "image_name": uploaded_file.name,
                "annotated_image": annotated,
                "counts": counts,
                "detections": detections_df,
                "elapsed": elapsed,
                "threshold": float(settings["confidence_threshold"]),
                "sample_id": sample_id,
            }

            # Re-render so both columns show the just-created result.
            st.rerun()

        except Exception as exc:
            st.error(f"Deteksi gagal: {exc}")

    result = st.session_state.get("latest_detection")
    if result is not None and result.get("image_name") == uploaded_file.name:
        # The result is also displayed under the original/result images after rerun.
        st.markdown("### Ringkasan hasil")
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Jumlah sel terdeteksi", str(sum(result["counts"].values())))
        with c2:
            metric_card("Ambang skor", f'{result["threshold"]:.2f}')
        with c3:
            metric_card("Waktu proses", f'{result["elapsed"]:.2f} detik')

        class_cols = st.columns(3)
        for col, name in zip(class_cols, CLASS_NAMES):
            with col:
                st.metric(f"Jumlah {name}", result["counts"][name])

        image_buffer = io.BytesIO()
        result["annotated_image"].save(image_buffer, format="PNG")
        csv_data = result["detections"].to_csv(index=False).encode("utf-8-sig")
        b1, b2 = st.columns(2)
        with b1:
            st.download_button(
                "⬇️ Simpan citra hasil", data=image_buffer.getvalue(),
                file_name="hasil_deteksi_sel_darah.png", mime="image/png",
                use_container_width=True,
            )
        with b2:
            st.download_button(
                "📄 Simpan ringkasan (CSV)", data=csv_data,
                file_name="ringkasan_deteksi_sel_darah.csv", mime="text/csv",
                use_container_width=True,
            )
        with st.expander("Lihat detail objek yang terdeteksi"):
            if result["detections"].empty:
                st.write("Tidak ada objek yang melewati ambang skor.")
            else:
                st.dataframe(result["detections"], use_container_width=True, hide_index=True)

    if not settings["auto_save_history"]:
        st.caption("Penyimpanan riwayat otomatis sedang nonaktif di Pengaturan.")

    st.markdown(
        '<div class="warning-note">ⓘ Hasil deteksi merupakan alat bantu analisis citra, bukan diagnosis medis.</div>',
        unsafe_allow_html=True,
    )


def show_history_page():
    st.subheader("Riwayat Deteksi Sel Darah")
    st.caption("Riwayat tersimpan berisi ID sampel, nama file, waktu, jumlah sel, dan parameter deteksi.")

    search_text = st.text_input("Cari ID sampel atau nama file", placeholder="Ketik kata kunci...")
    history = get_history(search_text)

    if history.empty:
        st.info("Belum ada riwayat deteksi. Jalankan deteksi terlebih dahulu dengan penyimpanan riwayat aktif.")
        return

    display_df = history[[
        "id", "sample_id", "image_name", "detected_at", "total_objects",
        "platelets_count", "rbc_count", "wbc_count", "confidence_threshold", "processing_seconds"
    ]].copy()
    display_df.columns = [
        "ID", "ID Sampel", "Nama File", "Tanggal", "Total Sel",
        "Platelets", "RBC", "WBC", "Ambang Skor", "Waktu (detik)"
    ]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    record_options = {
        f'ID {int(row["id"])} — {row["sample_id"]} — {row["image_name"]}': int(row["id"])
        for _, row in history.iterrows()
    }
    selected_label = st.selectbox("Pilih riwayat untuk melihat detail", list(record_options.keys()))
    selected_id = record_options[selected_label]
    selected = history.loc[history["id"] == selected_id].iloc[0]

    with st.expander("Detail riwayat terpilih", expanded=True):
        a, b, c = st.columns(3)
        a.metric("Total sel", int(selected["total_objects"]))
        b.metric("Ambang skor", f'{float(selected["confidence_threshold"]):.2f}')
        c.metric("Waktu proses", f'{float(selected["processing_seconds"]):.2f} detik')
        st.write(f'**ID sampel:** {selected["sample_id"]}')
        st.write(f'**Nama file:** {selected["image_name"]}')
        st.write(f'**Tanggal:** {selected["detected_at"]}')
        st.write(
            f'**Jumlah kelas:** Platelets {int(selected["platelets_count"])} · '
            f'RBC {int(selected["rbc_count"])} · WBC {int(selected["wbc_count"])}'
        )
        image_path_value = selected["result_image"]
        if isinstance(image_path_value, str) and image_path_value and Path(image_path_value).exists():
            st.image(str(image_path_value), caption="Citra hasil deteksi tersimpan", use_container_width=True)
        if st.button("Hapus riwayat terpilih", key="delete_selected_history"):
            delete_history_record(selected_id)
            st.success("Riwayat berhasil dihapus.")
            st.rerun()

    st.download_button(
        "⬇️ Unduh riwayat (CSV)",
        data=history.to_csv(index=False).encode("utf-8-sig"),
        file_name="riwayat_deteksi_sel_darah.csv",
        mime="text/csv",
    )


def show_settings_page(settings: dict):
    st.subheader("Pengaturan Sistem")

    with st.form("settings_form"):
        st.markdown("**A. Model AI SSD**")
        st.text_input("Model aktif", value="SSD300-VGG16", disabled=True)
        st.caption("Model mengikuti checkpoint yang sudah dilatih pada dataset BCCD.")

        st.markdown("**B. Pengaturan Tampilan**")
        new_threshold = st.slider(
            "Ambang skor default",
            min_value=0.05,
            max_value=0.95,
            value=float(settings["confidence_threshold"]),
            step=0.05,
        )
        color_names = list(COLOR_OPTIONS.keys())
        new_color = st.selectbox(
            "Warna bounding box",
            options=color_names,
            index=color_names.index(settings["box_color"]),
        )
        new_show_score = st.checkbox(
            "Tampilkan confidence score pada bounding box",
            value=bool(settings["show_score"]),
        )

        st.markdown("**C. Manajemen Data**")
        new_auto_save = st.checkbox(
            "Simpan riwayat deteksi secara otomatis",
            value=bool(settings["auto_save_history"]),
        )
        st.caption(f"Penyimpanan lokal aplikasi: {DB_PATH.name} dan folder history_images.")

        save_clicked = st.form_submit_button("💾 Simpan Perubahan", type="primary")

    if save_clicked:
        updated = {
            "confidence_threshold": float(new_threshold),
            "box_color": new_color,
            "show_score": bool(new_show_score),
            "auto_save_history": bool(new_auto_save),
        }
        save_settings(updated)
        st.session_state["app_settings"] = updated
        st.success("Pengaturan berhasil disimpan.")
        st.rerun()

    st.divider()
    st.markdown("**Hapus Riwayat Deteksi**")
    confirm_clear = st.checkbox("Saya yakin ingin menghapus seluruh riwayat deteksi.")
    if st.button("🗑️ Bersihkan Semua Riwayat", disabled=not confirm_clear):
        clear_all_history()
        st.success("Seluruh riwayat deteksi berhasil dihapus.")
        st.rerun()

    st.info(
        "Catatan: gunakan ID sampel, bukan nama, email, atau nomor identitas pasien. "
        "Aplikasi ini ditujukan untuk eksperimen citra dataset BCCD, bukan untuk menyimpan rekam medis."
    )

# ============================================================
# MAIN NAVIGATION
# ============================================================
if "app_settings" not in st.session_state:
    st.session_state["app_settings"] = load_settings()

settings = st.session_state["app_settings"]
show_header()

with st.sidebar:
    st.markdown("## Navigasi")
    page = st.radio(
        "Pilih halaman",
        ["Beranda", "Riwayat", "Pengaturan"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Model: SSD300-VGG16")
    st.caption(f"Perangkat: {DEVICE}")

if page == "Beranda":
    show_detection_page(settings)
elif page == "Riwayat":
    show_history_page()
elif page == "Pengaturan":
    show_settings_page(settings)

st.caption("Blood Cell Detection | SSD300-VGG16 | BCCD Dataset")
