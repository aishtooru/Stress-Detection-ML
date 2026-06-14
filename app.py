"""
Prediksi Stress Mahasiswa
Aplikasi berbasis Gradio untuk mendeteksi indikasi stress pada mahasiswa
menggunakan model Random Forests dan Naive Bayes.
"""

import numpy as np
import pandas as pd
import gradio as gr
import kagglehub
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# ─────────────────────────────────────────────
# 1. LOAD & TRAIN MODEL
# ─────────────────────────────────────────────

MODEL_CACHE = "models.pkl"

def prepare_data():
    """Download dataset dari Kaggle dan kembalikan X_train, X_test, y_train, y_test."""
    path = kagglehub.dataset_download(
        "sridevilavanyacse/student-lifestyle-and-stress-prediction-dataset"
    )
    file_name = "student-lifestyle-and-stress-dataset.csv"
    full_path = os.path.join(path, file_name)
    data = pd.read_csv(full_path)
    df = data.copy()

    numerical_cols = [
        "Sleep_Hours", "Study_Hours", "Social_Media_Hours",
        "Attendance", "Exam_Pressure", "Family_Support", "Month",
    ]

    # Hapus duplikat
    df.drop_duplicates(inplace=True)

    # Imputasi missing values pada Student_Type sebelum filter
    if df["Student_Type"].isnull().any():
        df["Student_Type"].fillna(df["Student_Type"].mode()[0], inplace=True)

    # Drop student type = 'school'
    df = df[df["Student_Type"] != "school"].copy()

    # Imputasi missing values numerik SETELAH filter agar median lebih representatif
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # paksa ke numerik
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Pembulatan — fillna 0 sebagai safety net untuk nilai inf/nan tersisa
    for col in ["Sleep_Hours", "Study_Hours", "Social_Media_Hours", "Exam_Pressure", "Family_Support"]:
        df[col] = df[col].round(0).fillna(0).astype(int)

    # Label encoding
    le = LabelEncoder()
    df["Student_Type_Encoded"] = le.fit_transform(df["Student_Type"])

    X = df[["Student_Type_Encoded", "Sleep_Hours", "Study_Hours",
             "Social_Media_Hours", "Exam_Pressure", "Family_Support"]]
    y = df["Stress_Level"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def load_or_train_models():
    """Load model dari cache atau latih ulang jika belum ada."""
    if os.path.exists(MODEL_CACHE):
        with open(MODEL_CACHE, "rb") as f:
            return pickle.load(f)

    print("⏳ Mengunduh dataset dan melatih model, mohon tunggu...")
    X_train, X_test, y_train, y_test = prepare_data()

    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)

    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)

    payload = {"rf": model_rf, "nb": model_nb}
    with open(MODEL_CACHE, "wb") as f:
        pickle.dump(payload, f)

    print("✅ Model berhasil dilatih!")
    return payload


# Load models sekali saat startup
models = load_or_train_models()
model_rf = models["rf"]
model_nb = models["nb"]

# ─────────────────────────────────────────────
# 2. FUNGSI PREDIKSI
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "Student_Type_Encoded", "Sleep_Hours", "Study_Hours",
    "Social_Media_Hours", "Exam_Pressure", "Family_Support",
]

STUDENT_TYPE_MAP = {"College": 0, "Working Student": 1}


def predict(student_type, sleep_hours, study_hours,
            social_media_hours, exam_pressure, family_support,
            model_choice):
    """
    Menerima input pengguna dan mengembalikan prediksi beserta probabilitas.
    """
    student_type_enc = STUDENT_TYPE_MAP[student_type]

    X_input = pd.DataFrame(
        [[student_type_enc, sleep_hours, study_hours,
          social_media_hours, exam_pressure, family_support]],
        columns=FEATURE_COLS,
    )

    if model_choice == "Random Forests":
        model = model_rf
    else:
        model = model_nb

    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]

    label = "😰 Terindikasi Stress" if pred == 1 else "😊 Tidak Terindikasi Stress"
    conf_stress = f"{proba[1] * 100:.1f}%"
    conf_no_stress = f"{proba[0] * 100:.1f}%"

    # Buat tabel probabilitas
    prob_table = pd.DataFrame({
        "Kelas": ["Tidak Stress", "Stress"],
        "Probabilitas": [f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
    })

    # Saran berdasarkan faktor risiko
    tips = []
    if sleep_hours < 6:
        tips.append("💤 Tidur kamu kurang dari 6 jam. Usahakan tidur 7–8 jam per malam.")
    if study_hours > 10:
        tips.append("📚 Jam belajar sangat tinggi (>10 jam). Pastikan ada waktu istirahat.")
    if social_media_hours > 4:
        tips.append("📱 Screen time media sosial >4 jam. Coba batasi penggunaannya.")
    if exam_pressure >= 4:
        tips.append("📝 Tekanan ujian tinggi. Pertimbangkan teknik belajar seperti Pomodoro.")
    if family_support <= 2:
        tips.append("👨‍👩‍👧 Dukungan keluarga rendah. Jangan ragu mencari dukungan dari konselor.")

    tips_text = "\n".join(tips) if tips else "✅ Pola hidupmu terlihat cukup sehat!"

    result_text = (
        f"### Hasil Prediksi: {label}\n\n"
        f"**Model yang digunakan:** {model_choice}\n\n"
        f"**Tingkat Keyakinan:**\n"
        f"- Tidak Stress: {conf_no_stress}\n"
        f"- Stress: {conf_stress}\n\n"
        f"---\n\n"
        f"### 💡 Saran Berdasarkan Input Kamu:\n{tips_text}"
    )

    return result_text, prob_table


# ─────────────────────────────────────────────
# 3. UI GRADIO
# ─────────────────────────────────────────────

DESCRIPTION = """
# 🎓 Prediksi Stress Mahasiswa

Aplikasi ini mendeteksi **indikasi stress** pada mahasiswa berdasarkan pola hidup dan faktor akademis
menggunakan algoritma **Random Forests** dan **Gaussian Naive Bayes**.

> Dataset: [Student Lifestyle and Stress Prediction Dataset](https://www.kaggle.com/datasets/sridevilavanyacse/student-lifestyle-and-stress-prediction-dataset)
"""

FOOTER = """
---
**Catatan:** Prediksi ini hanya bersifat indikatif dan tidak menggantikan konsultasi profesional.  
Jika kamu merasa tertekan, jangan ragu untuk menghubungi layanan konseling kampus. 💙
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Prediksi Stress Mahasiswa") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        # ── INPUT PANEL ──
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Masukkan Data Kamu")

            student_type = gr.Radio(
                choices=["College", "Working Student"],
                value="College",
                label="Tipe Mahasiswa",
            )
            sleep_hours = gr.Slider(
                minimum=1, maximum=12, step=1, value=7,
                label="⏰ Jam Tidur per Hari",
            )
            study_hours = gr.Slider(
                minimum=0, maximum=20, step=1, value=5,
                label="📖 Jam Belajar per Hari",
            )
            social_media_hours = gr.Slider(
                minimum=0, maximum=12, step=1, value=2,
                label="📱 Jam Media Sosial per Hari",
            )
            exam_pressure = gr.Slider(
                minimum=1, maximum=5, step=1, value=3,
                label="📝 Tekanan Ujian (1 = Rendah, 5 = Sangat Tinggi)",
            )
            family_support = gr.Slider(
                minimum=1, maximum=5, step=1, value=3,
                label="👨‍👩‍👧 Dukungan Keluarga (1 = Rendah, 5 = Sangat Tinggi)",
            )
            model_choice = gr.Dropdown(
                choices=["Random Forests", "Gaussian Naive Bayes"],
                value="Gaussian Naive Bayes",
                label="🤖 Pilih Model",
            )

            predict_btn = gr.Button("🔍 Prediksi Sekarang", variant="primary", size="lg")

        # ── OUTPUT PANEL ──
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Hasil Prediksi")
            result_md = gr.Markdown(value="*Hasil prediksi akan muncul di sini...*")
            prob_df = gr.Dataframe(
                headers=["Kelas", "Probabilitas"],
                label="Tabel Probabilitas",
                interactive=False,
            )

    gr.Markdown(FOOTER)

    # ── CONTOH INPUT ──
    gr.Examples(
        examples=[
            ["College",         5, 12, 6, 5, 2, "Gaussian Naive Bayes"],
            ["Working Student", 7,  5, 2, 2, 4, "Random Forests"],
            ["College",         8,  4, 1, 1, 5, "Gaussian Naive Bayes"],
        ],
        inputs=[
            student_type, sleep_hours, study_hours,
            social_media_hours, exam_pressure, family_support, model_choice,
        ],
        label="💡 Contoh Input",
    )

    predict_btn.click(
        fn=predict,
        inputs=[
            student_type, sleep_hours, study_hours,
            social_media_hours, exam_pressure, family_support, model_choice,
        ],
        outputs=[result_md, prob_df],
    )


if __name__ == "__main__":
    demo.launch()
