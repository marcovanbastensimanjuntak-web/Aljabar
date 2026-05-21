import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import io
import os

# Konfigurasi Halaman Web
st.set_page_config(page_title="Kompresi SVD", page_icon="🗜️", layout="wide")

st.title("🗜️ Kompresi Gambar Menggunakan SVD")
st.markdown("Aplikasi web untuk mengompresi gambar berwarna dan grayscale menggunakan metode **Singular Value Decomposition (SVD)**.")
st.markdown("---")

# ── Fungsi Inti SVD (Sama seperti Colab) ──
@st.cache_data
def compress_channel(channel: np.ndarray, k: int) -> np.ndarray:
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    k = min(k, len(S))
    result = (U[:, :k] * S[:k]) @ Vt[:k, :]
    return np.clip(result, 0, 255).astype(np.uint8)

@st.cache_data
def compress_rgb(img_array: np.ndarray, k: int) -> np.ndarray:
    channels = [compress_channel(img_array[:, :, c].astype(float), k) for c in range(3)]
    return np.stack(channels, axis=2)

@st.cache_data
def compress_gray(img_array: np.ndarray, k: int) -> np.ndarray:
    return compress_channel(img_array.astype(float), k)

def hitung_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    if mse == 0: return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))

def hitung_rasio(shape: tuple, k: int) -> tuple:
    M, N = shape[0], shape[1]
    asli  = M * N
    svd   = k * (M + 1 + N)
    rasio = asli / svd
    hemat = (1 - svd / asli) * 100
    return rasio, hemat

def max_k(shape: tuple) -> int:
    return min(shape[0], shape[1])

# ── UI Web Streamlit ──
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📂 Pilih foto (JPG / PNG / WEBP)", type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file is not None:
    # Proses Gambar Asli
    img_pil = Image.open(uploaded_file)
    img_rgb = np.array(img_pil.convert("RGB"))
    img_gray = np.array(img_pil.convert("L"))
    H, W = img_gray.shape
    k_max = max_k((H, W))

    with col2:
        st.info(f"📐 Ukuran gambar: **{W} × {H} pixel** | 🔢 Nilai *k* maks: **{k_max}**")
        k_input = st.number_input(f"Masukkan nilai k (1 - {k_max})", min_value=1, max_value=k_max, value=min(50, k_max), step=5)
        proses_btn = st.button("🚀 Mulai Kompresi", type="primary", use_container_width=True)

    if proses_btn:
        with st.spinner("⏳ Sedang memproses matriks SVD..."):
            hasil_color = compress_rgb(img_rgb, k_input)
            hasil_gray  = compress_gray(img_gray, k_input)

            rasio_c, hemat_c = hitung_rasio((H, W), k_input)
            rasio_g, hemat_g = hitung_rasio((H, W), k_input)
            psnr_c = hitung_psnr(img_rgb, hasil_color)
            psnr_g = hitung_psnr(img_gray, hasil_gray)

            filename = uploaded_file.name

            # ── Visualisasi Matplotlib (Sama persis dengan Colab) ──
            fig = plt.figure(figsize=(20, 9))
            fig.patch.set_facecolor("#0d0d16")
            fig.suptitle(f"Kompresi SVD  —  {filename}   |   k = {k_input}", fontsize=16, fontweight="bold", color="#e8e8f5", y=0.97)

            gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.06, left=0.02, right=0.98, top=0.90, bottom=0.06, width_ratios=[1, 1, 0.55])

            def buat_ax(subplot, gambar, judul, sub="", cmap_val=None, border=None):
                ax = fig.add_subplot(subplot)
                ax.imshow(gambar, cmap=cmap_val)
                ax.set_title(judul, fontsize=12, fontweight="bold", color="#ffffff", pad=7)
                ax.set_xlabel(sub, fontsize=9.5, color="#aaaacc", labelpad=5)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(bool(border))
                    if border:
                        spine.set_edgecolor(border)
                        spine.set_linewidth(3)
                return ax

            buat_ax(gs[0, 0], img_rgb, "📷 Asli — Berwarna", f"{W} × {H} px")
            buat_ax(gs[0, 1], hasil_color, f"🗜️  Kompres Berwarna  (k = {k_input})", f"Rasio {rasio_c:.1f}×  |  Hemat {hemat_c:.0f}%  |  PSNR {psnr_c:.1f} dB", border="#4fc3f7")
            buat_ax(gs[1, 0], img_gray, "📷 Asli — Grayscale", f"{W} × {H} px", cmap_val="gray")
            buat_ax(gs[1, 1], hasil_gray, f"🗜️  Kompres Grayscale  (k = {k_input})", f"Rasio {rasio_g:.1f}×  |  Hemat {hemat_g:.0f}%  |  PSNR {psnr_g:.1f} dB", cmap_val="gray", border="#81c784")

            # Bar chart PSNR
            ax_bar = fig.add_subplot(gs[0, 2])
            ax_bar.set_facecolor("#161626")
            labels = ["Berwarna", "Grayscale"]
            psnrs  = [psnr_c, psnr_g]
            clrs   = ["#4fc3f7", "#81c784"]
            bars   = ax_bar.bar(labels, psnrs, color=clrs, width=0.42, edgecolor="none")
            for bar, val in zip(bars, psnrs):
                ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25, f"{val:.1f} dB", ha="center", va="bottom", fontsize=11, color="white", fontweight="bold")
            ax_bar.set_title("Kualitas PSNR", color="white", fontsize=11, pad=8)
            ax_bar.tick_params(colors="white", labelsize=10)
            ax_bar.set_ylabel("dB", color="#aaaacc", fontsize=9)
            ax_bar.set_ylim(0, max(psnrs) * 1.2)
            for spine in ax_bar.spines.values(): spine.set_edgecolor("#2a2a4e")

            # Tabel ringkasan
            ax_tbl = fig.add_subplot(gs[1, 2])
            ax_tbl.axis("off")
            header = ["Metrik", "Berwarna", "Grayscale"]
            rows   = [
                ["Nilai k", str(k_input), str(k_input)],
                ["Rasio", f"{rasio_c:.2f}×", f"{rasio_g:.2f}×"],
                ["Hemat", f"{hemat_c:.1f}%", f"{hemat_g:.1f}%"],
                ["PSNR", f"{psnr_c:.2f} dB", f"{psnr_g:.2f} dB"],
            ]
            tbl = ax_tbl.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1.15, 2.1)
            for (r, c), cell in tbl.get_celld().items():
                if r == 0: cell.set_facecolor("#2a2a4e")
                elif c == 0: cell.set_facecolor("#1c1c38")
                else: cell.set_facecolor("#13132a")
                cell.set_text_props(color="white")
                cell.set_edgecolor("#2a2a5e")
            ax_tbl.set_title("Ringkasan", color="white", fontsize=11, pad=8)

            st.pyplot(fig) # Tampilkan grafik matplotlib di Web

            # ── Fitur Download ──
            st.markdown("### 💾 Download Hasil")
            col_dl1, col_dl2 = st.columns(2)
            
            # Convert RGB array to bytes for download
            buf_color = io.BytesIO()
            Image.fromarray(hasil_color).save(buf_color, format="JPEG")
            col_dl1.download_button(label="🟦 Download Berwarna", data=buf_color.getvalue(), file_name=f"kompres_{k_input}_warna.jpg", mime="image/jpeg", use_container_width=True)

            # Convert Gray array to bytes for download
            buf_gray = io.BytesIO()
            Image.fromarray(hasil_gray).save(buf_gray, format="JPEG")
            col_dl2.download_button(label="⬜ Download Grayscale", data=buf_gray.getvalue(), file_name=f"kompres_{k_input}_gray.jpg", mime="image/jpeg", use_container_width=True)
