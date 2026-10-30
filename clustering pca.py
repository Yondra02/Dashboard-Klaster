import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import io

# ==========================================
# 1. PENGATURAN HALAMAN & SESSION STATE
# ==========================================
st.set_page_config(page_title="Dashboard Analisis Klaster Hierarki", layout="wide", initial_sidebar_state="collapsed")

if 'mulai_proses' not in st.session_state:
    st.session_state.mulai_proses = False

def reset_proses():
    st.session_state.mulai_proses = False

# ==========================================
# 2. CSS STYLING: PUTIH, 3D, BERTEKSTUR & RESPONSIF HP
# ==========================================
st.markdown("""
    <style>
        /* 1. Latar Belakang Utama (Putih Bersih) */
        .stApp { 
            background: #f8fafc; 
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: #1e293b;
        }
        .block-container { max-width: 95%; padding-top: 1.5rem; padding-bottom: 5rem; }

        /* 2. Hero Banner: Zamrud dengan Pola */
        .hero-banner {
            background: linear-gradient(135deg, #064e3b 0%, #10b981 100%);
            padding: 40px 50px; border-radius: 20px; color: white;
            box-shadow: 0 15px 35px rgba(16, 185, 129, 0.25);
            margin-bottom: 30px; position: relative; overflow: hidden;
        }
        .hero-banner::before {
            content: ''; position: absolute; top: -50%; right: -5%;
            width: 300px; height: 300px; background: rgba(255,255,255,0.1);
            border-radius: 50%; filter: blur(30px);
        }
        .hero-banner h1 { font-size: 35px; font-weight: 900; margin-bottom: 12px; color: #ffffff !important; position: relative; z-index: 2;}
        .hero-banner p { font-size: 16px; color: #ecfdf5; max-width: 900px; line-height: 1.6; position: relative; z-index: 2;}

        /* 3. Panel Kontainer (Putih Murni dengan 3D Shadow + Aksen Atas) */
        div[data-testid="stVerticalBlock"] > div[style*="border"] {
            background: #ffffff !important;
            border: none !important;
            border-top: 4px solid #10b981 !important;
            border-radius: 16px !important; padding: 28px !important;
            box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0,0,0,0.05) !important;
            transition: all 0.3s ease !important;
        }
        div[data-testid="stVerticalBlock"] > div[style*="border"]:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 40px -10px rgba(16, 185, 129, 0.15), 0 1px 3px rgba(0,0,0,0.05) !important;
        }

        /* 4. Tombol Proses (Elegan & Proporsional) */
        .stButton>button { 
            width: 100%; border-radius: 10px; 
            background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
            color: white; font-weight: 700; padding: 12px 15px; font-size: 15.5px; 
            border: none; box-shadow: 0 6px 15px rgba(16, 185, 129, 0.25);
            transition: all 0.2s ease; margin-top: 15px; letter-spacing: 0.5px;
        }
        .stButton>button:hover { 
            transform: translateY(-2px); box-shadow: 0 8px 20px rgba(16, 185, 129, 0.35); 
            background: linear-gradient(135deg, #059669 0%, #047857 100%); 
        }

        /* 5. Subheader Keren dengan Ikon Bulat */
        h3 { 
            color: #0f172a !important; font-size: 20px !important; font-weight: 800 !important; 
            margin-bottom: 25px !important; display: flex; align-items: center; gap: 10px;
        }
        
        /* 6. Kotak Keterangan Metode */
        .metode-info {
            background: linear-gradient(135deg, #f0fdf4 0%, #d1fae5 100%); 
            border-left: 5px solid #10b981; padding: 18px 22px;
            border-radius: 12px; font-size: 14px; color: #064e3b; margin-top: 15px;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.1); line-height: 1.7;
        }
        
        /* 7. Kotak PCA */
        .pca-box {
            background: linear-gradient(135deg, #064e3b 0%, #022c22 100%); color: white; 
            padding: 22px 25px; border-radius: 14px; font-size: 14.5px; 
            box-shadow: 0 10px 25px rgba(6, 78, 59, 0.25); margin-top: 10px; line-height: 1.7;
            position: relative; overflow: hidden; border: 1px solid #34d399;
        }
        .pca-box b { color: #6ee7b7; font-size: 15.5px; }

        /* 8. Kotak Anggota Klaster */
        .cluster-box {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); color: #334155;
            border-radius: 14px; padding: 20px; margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.04); font-size: 14.5px; line-height: 1.7;
            border: 1px solid #e2e8f0; border-left: 6px solid #0ea5e9;
        }
        .cluster-box b { 
            color: #0369a1; font-size: 16px; display: block; margin-bottom: 10px; 
            border-bottom: 2px dashed #bae6fd; padding-bottom: 8px;
        }

        /* 9. Gaya Interpretasi per Variabel */
        .var-card {
            background: #ffffff; border-radius: 12px; padding: 20px; margin-bottom: 18px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.06); border: 1px solid #f1f5f9;
            position: relative; overflow: hidden;
        }
        .var-card::before {
            content: ''; position: absolute; top: 0; left: 0; width: 6px; height: 100%;
            background: linear-gradient(to bottom, #10b981, #059669);
        }
        .var-title {
            background: #ecfdf5; color: #047857; padding: 6px 14px; border-radius: 6px;
            font-weight: 800; font-size: 14.5px; margin-bottom: 12px; display: inline-block;
        }
        .var-desc { font-size: 14.5px; color: #475569; line-height: 1.7; margin:0;}
        .var-desc b { color: #0f172a; font-weight: 800; }

        /* Responsive Mobile CSS */
        @media (max-width: 768px) {
            .hero-banner { padding: 25px 20px; }
            .hero-banner h1 { font-size: 24px; }
            .block-container { padding-top: 1rem; padding-bottom: 3rem; }
        }
        
        #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. HEADER & BANNER
# ==========================================
st.markdown("""
    <div class="hero-banner">
        <h1>Dashboard Analisis Klaster Hierarki</h1>
        <p>Unggah dataset Anda dan lakukan analisis klaster hierarki dengan empat metode hierarki, jarak Euclidean, visualisasi korelasi, opsi PCA, serta evaluasi Silhouette Score untuk menentukan jumlah klaster dan melihat distribusi hasil pengelompokan.</p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 4. PANDUAN FORMAT DATA (DENGAN CONTOH X1, X2, X3)
# ==========================================
with st.expander("📖 Panduan & Contoh Format Dataset (Klik untuk Buka/Tutup)", expanded=False):
    st.write("Pastikan file **.CSV** atau **.XLSX** Anda disusun dengan ketentuan berikut:")
    st.markdown("""
    - **Kolom Pertama:** Berisi label identitas wilayah/objek (contoh: Nama Provinsi atau Kabupaten).
    - **Kolom Selanjutnya:** Berisi variabel/indikator numerik (misal disimbolkan $X_1, X_2, X_3$, dst. atau menggunakan nama asli indikator Anda) untuk proses perhitungan klaster.
    """)
    st.markdown("**Contoh Struktur Tabel Data yang Valid:**")
    df_contoh = pd.DataFrame({
        "Wilayah": ["Aceh", "Sumatera Utara", "Sumatera Barat"],
        "X1": [305, 408, 196],
        "X2": [616, 232, 225],
        "X3": [4702, 26424, 6136]
    })
    st.table(df_contoh)

# ==========================================
# 5. KONTROL & UPLOAD DATASET
# ==========================================
uploaded_file = st.file_uploader("📂 Unggah Dataset (.CSV / .XLSX)", type=['csv', 'xlsx'], on_change=reset_proses)

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
        
    kolom_kiri, kolom_kanan = st.columns([1.2, 2.5], gap="large")
    
    with kolom_kiri:
        with st.container(border=True):
            st.subheader("⚙️ 1. Panel Kendali")
            kolom_numerik_tersedia = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            fitur_terpilih = st.multiselect(
                "Variabel observasi:", 
                options=kolom_numerik_tersedia, 
                default=kolom_numerik_tersedia,
                on_change=reset_proses
            )
            
            pilihan_metode = {
                "single": "Single Linkage (Jarak Terdekat)",
                "complete": "Complete Linkage (Jarak Terjauh)",
                "average": "Average Linkage (Jarak Rata-rata)",
                "ward": "Ward's Method (Varians Minimum)"
            }
            metode_linkage = st.selectbox("Algoritma pautan (Linkage):", options=list(pilihan_metode.keys()), format_func=lambda x: pilihan_metode[x], on_change=reset_proses)
            
            keterangan = {
                "single": "Mendefinisikan jarak klaster berdasarkan titik observasi terdekat. Rentan terhadap fenomena efek rantai <i>(chaining effect)</i> memanjang.",
                "complete": "Menghitung kedekatan berdasarkan titik observasi terjauh. Cenderung menghasilkan struktur klaster yang cembung dan sangat padat.",
                "average": "Mengevaluasi rata-rata jarak seluruh pasangan titik. Lebih tangguh <i>(robust)</i> dan tahan terhadap pengaruh nilai ekstrem/outlier.",
                "ward": "Metode optimal berbasis analisis varians (ANOVA). Menggabungkan klaster dengan tujuan meminimalkan peningkatan varians di dalamnya."
            }
            st.markdown(f'<div class="metode-info">💡 <b>Tinjauan Algoritma:</b><br>{keterangan[metode_linkage]}</div>', unsafe_allow_html=True)
            
            if st.button("🚀 PROSES DATA SEKARANG"):
                st.session_state.mulai_proses = True
        
    with kolom_kanan:
        with st.container(border=True):
            st.subheader("📋 Tinjauan Dataset Mentah")
            st.write("Sistem mendeteksi kolom non-numerik paling kiri sebagai label identitas.")
            st.dataframe(df, width='stretch', height=400)
        
    # ==========================================
    # 6. ALUR ANALISIS INTERAKTIF
    # ==========================================
    if st.session_state.mulai_proses:
        if len(fitur_terpilih) < 2:
            st.error("⚠️ Diperlukan minimal dua variabel numerik untuk proses komputasi multivariat.")
        else:
            X = df[fitur_terpilih].dropna()
            
            st.markdown("---")
            col_chart, col_zscore = st.columns([1.2, 1.5], gap="large")
            
            with col_chart:
                with st.container(border=True):
                    st.subheader("🗺️ 2. Matriks Korelasi Pearson")
                    st.write("Intensitas warna hijau merepresentasikan koefisien korelasi linear antar variabel.")
                    
                    fig, ax = plt.subplots(figsize=(6, 5)) 
                    sns.heatmap(X.corr(), annot=True, cmap="Greens", fmt=".2f", 
                                linewidths=1.5, cbar_kws={"shrink": 0.8}, 
                                annot_kws={"size": 11, "weight": "bold"}, ax=ax)
                    fig.patch.set_alpha(0.0) 
                    ax.set_facecolor('white')
                    st.pyplot(fig, use_container_width=True)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            df_scaled = pd.DataFrame(X_scaled, columns=fitur_terpilih)
            
            with col_zscore:
                with st.container(border=True):
                    st.subheader("🧮 3. Standardisasi Z-Score")
                    st.write("Transformasi metrik untuk mengeliminasi bias satuan pengamatan (μ=0, σ=1).")
                    st.dataframe(df_scaled, width='stretch', height=360)

            st.markdown("---")
            with st.container(border=True):
                st.subheader("🧬 4. Reduksi Dimensi (PCA)")
                st.write("Apabila matriks korelasi mengindikasikan gejala multikolinearitas (koefisien > 0.7), aktivasi PCA sangat direkomendasikan.")
                
                opsi_pca = st.toggle("Aktifkan Principal Component Analysis (PCA)?", value=False)
                
                X_clustering = X_scaled 
                
                if opsi_pca:
                    pca_check = PCA().fit(X_scaled)
                    eigenvalues = pca_check.explained_variance_
                    variance_ratios = pca_check.explained_variance_ratio_ * 100
                    
                    df_pca = pd.DataFrame({
                        'Komponen Utama (PC)': [f"PC-{i+1}" for i in range(len(eigenvalues))],
                        'Nilai Eigen': eigenvalues,
                        'Proporsi Varians (%)': variance_ratios,
                        'Varians Kumulatif (%)': np.cumsum(variance_ratios)
                    })
                    
                    pcs_terpilih = df_pca[df_pca['Nilai Eigen'] > 1]
                    n_pcs = len(pcs_terpilih) if len(pcs_terpilih) > 0 else 1
                    
                    pca_final = PCA(n_components=n_pcs)
                    X_clustering = pca_final.fit_transform(X_scaled)
                    total_var_terwakili = pcs_terpilih['Proporsi Varians (%)'].sum()
                    
                    col_pca_tabel, col_pca_teks = st.columns([1.5, 1])
                    with col_pca_tabel:
                        st.dataframe(df_pca.style.map(lambda x: 'background-color: #d1fae5; font-weight:bold;' if x > 1 else '', subset=['Nilai Eigen']), width='stretch')
                    
                    with col_pca_teks:
                        st.markdown(f"""
                        <div class="pca-box">
                            <b>📑 Ekstraksi Kaiser Criterion:</b><br><br>
                            Batas retensi komponen ditetapkan pada ambang batas Nilai Eigen > 1.<br><br>
                            Algoritma mengekstraksi <b>{n_pcs} Komponen Utama</b> yang secara kolektif merangkum <b>{total_var_terwakili:.2f}%</b> dari total varians observasi awal.
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown("---")
            with st.container(border=True):
                st.subheader("🎯 5. Validitas Klaster (Silhouette Index)")
                st.write("Indeks evaluasi spasial. Nilai yang mendekati +1 mengindikasikan konfigurasi klaster yang terpisah secara tegas.")
                
                skor_silhouette = {}
                col_metrik = st.columns(5)
                
                for idx, k in enumerate([2, 3, 4, 5, 6]):
                    model_eval = AgglomerativeClustering(n_clusters=k, linkage=metode_linkage)
                    label_eval = model_eval.fit_predict(X_clustering)
                    skor = silhouette_score(X_clustering, label_eval)
                    skor_silhouette[k] = skor
                    col_metrik[idx].metric(label=f"K = {k}", value=f"{skor:.4f}")

            st.markdown("---")
            with st.container(border=True):
                st.subheader("⚙️ 6. Distribusi Anggota Klaster")
                jumlah_klaster_user = st.slider("Tentukan parameter jumlah klaster (K) yang akan dibentuk:", min_value=2, max_value=6, value=3)
                
                model_final = AgglomerativeClustering(n_clusters=jumlah_klaster_user, linkage=metode_linkage)
                df['Klaster_Final'] = model_final.fit_predict(X_clustering)
                
                with st.container(border=True):
                    
                    kolom_teks = df.select_dtypes(include=['object', 'string']).columns.tolist()
                    kolom_label = kolom_teks[0] if kolom_teks else df.columns[0]
                    
                    kolom_klaster_ui = st.columns(2)
                    for i in range(jumlah_klaster_user):
                        anggota = df[df['Klaster_Final'] == i][kolom_label].tolist()
                        teks_anggota = ", ".join(str(x) for x in anggota)
                        kolom_klaster_ui[i % 2].markdown(f"""
                            <div class="cluster-box">
                                <b>🔵 KLASTER {i} (n = {len(anggota)})</b>
                                {teks_anggota}
                            </div>
                        """, unsafe_allow_html=True)

            st.markdown("---")
            col_tabel, col_interpretasi = st.columns([1, 1.2], gap="large")
            
            with col_tabel:
                with st.container(border=True):
                    st.subheader("📊 Profil Rata-Rata Klaster")
                    st.write(f"Distribusi nilai rata-rata observasi asli pada masing-masing entitas klaster.")
                    df_mean = df.groupby('Klaster_Final')[fitur_terpilih].mean().reset_index()
                    df_mean.rename(columns={'Klaster_Final': 'Klaster'}, inplace=True)
                    st.table(df_mean.style.format(precision=2))
                
            with col_interpretasi:
                with st.container(border=True):
                    st.subheader("💡 Interpretasi Karakteristik")
                    st.write("Berdasarkan perbandingan nilai rata-rata sentral:")
                    
                    interpretasi_html = ""
                    for col in fitur_terpilih:
                        max_idx = df_mean[col].idxmax()
                        min_idx = df_mean[col].idxmin()
                        
                        klaster_max = int(df_mean.loc[max_idx, 'Klaster'])
                        klaster_min = int(df_mean.loc[min_idx, 'Klaster'])
                        val_max = df_mean.loc[max_idx, col]
                        val_min = df_mean.loc[min_idx, col]
                        
                        interpretasi_html += f"""
                        <div class='var-card'>
                            <div class='var-title'>Indikator {col}</div>
                            <p class='var-desc'>👉 <b>Klaster {klaster_max}</b> menunjukkan konsentrasi observasi tertinggi (Rata-rata = <b>{val_max:.2f}</b>).</p>
                            <p class='var-desc'>👉 <b>Klaster {klaster_min}</b> memiliki tendensi observasi terendah (Rata-rata = <b>{val_min:.2f}</b>).</p>
                        </div>
                        """
                    st.markdown(interpretasi_html, unsafe_allow_html=True)

            st.markdown("---")
            with st.container(border=True):
                st.subheader("📂 Ekspor Laporan Analitik")
                st.dataframe(df, width='stretch')
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Data_Terklaster')
                excel_data = output.getvalue()
                
                st.download_button(
                    label=f"⬇️ UNDUH DATASET FINAL (XLSX)", 
                    data=excel_data, 
                    file_name=f"Dataset_Klaster_{jumlah_klaster_user}K.xlsx", 
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
