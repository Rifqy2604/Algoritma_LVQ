import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="LVQ Web App", layout="wide")
st.title("🔬 LVQ (Learning Vector Quantization) Web App")

# Upload file
uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("📄 Dataset Asli")
    st.dataframe(df)

    fitur_cols = st.multiselect("Pilih kolom fitur (X):", df.columns)
    target_col = st.selectbox("Pilih kolom target (Y):", df.columns)

    if fitur_cols and target_col:
        X = df[fitur_cols]
        y = df[target_col]

        # Normalisasi
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=fitur_cols)
        X_scaled_df[target_col] = y.values

        st.subheader("📊 Data Setelah Normalisasi")
        st.dataframe(X_scaled_df)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Gabungkan untuk tampil
        st.subheader("📚 Data Latih & Data Uji")
        st.write(f"Jumlah Data Latih: {len(X_train)}")
        st.write(f"Jumlah Data Uji: {len(X_test)}")
        st.dataframe(pd.DataFrame(X_train, columns=fitur_cols).assign(Label=y_train.values).reset_index(drop=True))

        # Inisialisasi bobot awal dari satu data tiap kelas
        unique_classes = np.unique(y_train)
        bobot = {}
        for c in unique_classes:
            idx = np.where(y_train == c)[0][0]
            bobot[c] = X_train[idx]

        st.subheader("🧠 Bobot Awal")
        for k in sorted(bobot):
            st.text(f"Bobot W{k} (kelas {k}): {np.round(bobot[k], 4)}")

        # Parameter
        learning_rate = 0.05
        epoch = 2
        st.subheader("⚙️ Parameter")
        st.write(f"Learning Rate: {learning_rate}")
        st.write(f"Epoch: {epoch}")

        # Proses training LVQ
        X_train_arr = np.array(X_train)
        y_train_arr = np.array(y_train)

        for ep in range(epoch):
            for i in range(len(X_train_arr)):
                x = X_train_arr[i]
                y_true = y_train_arr[i]

                # Hitung jarak ke semua bobot
                jarak = {k: np.linalg.norm(x - bobot[k]) for k in bobot}
                label_terdekat = min(jarak, key=jarak.get)

                # Update bobot
                if label_terdekat == y_true:
                    bobot[label_terdekat] += learning_rate * (x - bobot[label_terdekat])
                else:
                    bobot[label_terdekat] -= learning_rate * (x - bobot[label_terdekat])

        st.success("✅ Proses pelatihan selesai!")

        st.subheader("🧠 Bobot Akhir")
        for k in sorted(bobot):
            st.write(f"Bobot Akhir W{k}: {bobot[k]}")

        # Prediksi data uji
        X_test_arr = np.array(X_test)
        y_test_arr = np.array(y_test)
        prediksi = []

        for x in X_test_arr:
            jarak = {k: np.linalg.norm(x - bobot[k]) for k in bobot}
            label_pred = min(jarak, key=jarak.get)
            prediksi.append(label_pred)

        # Evaluasi
        hasil_uji = pd.DataFrame(X_test_arr, columns=fitur_cols)
        hasil_uji["Label Asli"] = y_test_arr
        hasil_uji["Prediksi"] = prediksi

        st.subheader("🧪 Hasil Prediksi Data Uji")
        st.dataframe(hasil_uji)

        akurasi = np.mean(y_test_arr == prediksi)
        st.metric("🎯 Akurasi", f"{akurasi*100:.2f}%")
