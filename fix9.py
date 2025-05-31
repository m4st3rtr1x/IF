import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import io
import matplotlib.cm as cm

st.set_page_config(page_title="Clustering CPL-PL", layout="wide")
st.title("Analisis Clustering CPL & PL berdasarkan Nilai A Mahasiswa")

uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    with st.expander("1. Dataset Awal", expanded=True):
        st.dataframe(df)

    df = df.rename(columns={
        'kodematakuliah': 'Kode Mata Kuliah',
        'NAMA MATAKULIAH': 'Nama Mata Kuliah',
        'CPL Yang dipenuhi': 'CPL',
        'PL yang dipenuhi': 'PL',
        'nilai': 'Nilai'
    })

    df = df.dropna()
    df = df.drop_duplicates()
    df = df[df['Nilai'] == 'A']

    excluded_courses = [
        'AGAMA', 'BAHASA INDONESIA', 'BAHASA INGGRIS',
        'KEWIRAUSAHAAN', 'HARDWARE KOMPUTER', 'ANIMASI DAN MULTIMEDIA'
    ]
    df = df[~df['Nama Mata Kuliah'].str.upper().isin(excluded_courses)]

    df = df[['Kode Mata Kuliah', 'Nama Mata Kuliah', 'CPL', 'PL']]

    with st.expander("2. Dataset Setelah Pra-pemrosesan", expanded=False):
        st.dataframe(df)

    le_nama_mk = LabelEncoder()
    le_cpl = LabelEncoder()
    le_pl = LabelEncoder()

    df_encoded = df.copy()
    df_encoded['Nama Mata Kuliah'] = le_nama_mk.fit_transform(df_encoded['Nama Mata Kuliah'])
    df_encoded['CPL'] = le_cpl.fit_transform(df_encoded['CPL'])
    df_encoded['PL'] = le_pl.fit_transform(df_encoded['PL'])

    df_encoded['Jumlah Mahasiswa'] = 1

    with st.expander("3. Dataset Setelah Encoding", expanded=False):
        st.dataframe(df_encoded)

    df_agg = df_encoded.groupby(
        ['Kode Mata Kuliah', 'Nama Mata Kuliah', 'CPL', 'PL']
    ).agg({'Jumlah Mahasiswa': 'sum'}).reset_index()

    with st.expander("4. Dataset Setelah Agregasi", expanded=False):
        st.dataframe(df_agg)

    features = ['Nama Mata Kuliah', 'CPL', 'PL', 'Jumlah Mahasiswa']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_agg[features])

    with st.expander("5. Dataset Setelah Normalisasi (fitur numerik)", expanded=False):
        df_normalized = pd.DataFrame(X_scaled, columns=features)
        st.dataframe(df_normalized)

    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    with st.expander("6. Data Mining : Elbow Method", expanded=False):
        fig, ax = plt.subplots()
        ax.plot(K, inertia, marker='o')
        ax.set_xlabel('Jumlah Cluster (k)')
        ax.set_ylabel('Inertia')
        ax.set_title('Menentukan k Optimal')
        st.pyplot(fig)

        k_opt = st.slider("Pilih jumlah cluster", min_value=2, max_value=10, value=4)

    kmeans = KMeans(n_clusters=k_opt, random_state=42)
    df_agg['Cluster'] = kmeans.fit_predict(X_scaled)

    df_agg['Nama Mata Kuliah'] = le_nama_mk.inverse_transform(df_agg['Nama Mata Kuliah'])
    df_agg['CPL'] = le_cpl.inverse_transform(df_agg['CPL'])
    df_agg['PL'] = le_pl.inverse_transform(df_agg['PL'])

    with st.expander("7. Clustering dengan K-Means (Hasil Per Record)", expanded=True):
        st.write(f"Hasil clustering dengan jumlah cluster = {k_opt}")
        st.dataframe(df_agg)

    with st.expander("8. Representasi Masing-masing Cluster", expanded=True):
        for i in range(k_opt):
            st.write(f"📊 **Cluster {i}**")
            df_cluster_i = df_agg[df_agg['Cluster'] == i]
            st.dataframe(df_cluster_i)

            towrite_cluster = io.BytesIO()
            df_cluster_i.to_excel(towrite_cluster, index=False, engine='openpyxl')
            towrite_cluster.seek(0)
            st.download_button(
                label=f"📥 Download Data Cluster {i}",
                data=towrite_cluster,
                file_name=f"cluster_{i}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    interpretasi_map = {
        0: 'Cluster ini menunjukkan mahasiswa dengan kemampuan praktik langsung yang cukup baik, banyak di antaranya berhasil menguasai mata kuliah berbasis implementasi.',
        1: 'Cluster ini menunjukkan mahasiswa memiliki kompetensi awal dalam pemrograman dan pengelolaan data, cocok untuk posisi entry-level developer.',
        2: 'Cluster ini berisi mahasiswa yang kuat secara sistemik dan manajerial, menunjukkan pemahaman konsep arsitektur dan infrastruktur TI.',
        3: 'Cluster paling unggul secara performa mahasiswa, dengan dominasi mata kuliah data science dan teknologi canggih.'
    }

    profesi_map = {
        0: 'Hardware Integrator, System Analyst, IT Consultant IoT',
        1: 'Junior Data Engineer, Frontend Dev, Entry-Level Developer',
        2: 'IT Infrastructure Consultant, SysAdmin, Technical Advisor',
        3: 'Data Scientist, BI Specialist, Big Data Analyst'
    }

    df_agg['Interpretasi'] = df_agg['Cluster'].map(interpretasi_map)
    df_agg['Rekomendasi Profesi'] = df_agg['Cluster'].map(profesi_map)

    cluster_summary = df_agg.groupby('Cluster').agg({
        'Jumlah Mahasiswa': 'mean',
        'Nama Mata Kuliah': lambda x: ', '.join(x.unique()[:3]),
        'CPL': lambda x: ', '.join(x.unique()[:3]),
        'PL': lambda x: ', '.join(x.unique()[:2]),
        'Interpretasi': 'first',
        'Rekomendasi Profesi': 'first'
    }).reset_index()

    cluster_summary.rename(columns={
        'Nama Mata Kuliah': 'Mata Kuliah',
        'CPL': 'CPL',
        'PL': 'PL Dominan',
        'Interpretasi': 'Interpretasi',
        'Rekomendasi Profesi': 'Rekomendasi Profesi',
        'Jumlah Mahasiswa': 'Rata-rata Mahasiswa'
    }, inplace=True)

    with st.expander("9. Ringkasan Cluster", expanded=False):
        st.dataframe(cluster_summary)
        towrite = io.BytesIO()
        cluster_summary.to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        st.download_button(
            label="Download Hasil Ringkasan",
            data=towrite,
            file_name="hasil_ringkasan_cluster.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with st.expander("10. Interpretasi dan Rekomendasi Per Cluster", expanded=True):
        for i in range(k_opt):
            st.markdown(f"### Cluster {i}")
            st.write(f"🧠 **Interpretasi:** {interpretasi_map.get(i, '-')}")
            st.write(f"💼 **Rekomendasi Profesi:** {profesi_map.get(i, '-')}")

    # with st.expander("11. Data Lengkap per Record", expanded=False):
    #     st.dataframe(df_agg)
    #     towrite2 = io.BytesIO()
    #     df_agg.to_excel(towrite2, index=False, engine='openpyxl')
    #     towrite2.seek(0)
    #     st.download_button(
    #         label="Download Hasil Per Record",
    #         data=towrite2,
    #         file_name="hasil_per_record_cluster.xlsx",
    #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    #     )

    with st.expander("11. Visualisasi Heatmap CPL vs PL per Mata Kuliah", expanded=False):
        pivot_heatmap = df_agg.pivot_table(
            index='CPL',
            columns='PL',
            values='Jumlah Mahasiswa',
            aggfunc='sum',
            fill_value=0
        )
        fig_hm, ax_hm = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_heatmap, annot=True, fmt='d', cmap='YlGnBu', ax=ax_hm)
        ax_hm.set_title("Jumlah Mahasiswa Berdasarkan CPL dan PL")
        st.pyplot(fig_hm)
