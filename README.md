\# Neural IR Exercise dengan BERT Cross-Encoder



Repositori ini berisi implementasi sistem pencarian dua tahap (\*Two-stage Retrieval\*) dan ekstraksi jawaban otomatis untuk memenuhi Tugas Lab Mata Kuliah Advanced Information Retrieval.



\## 👥 Anggota Kelompok 1

\* Rizqia Fauziah Rachma – 14250002

\* Tungkot Siregar – 14250003

\* Intan Saesaria – 14250004



\## 📂 Struktur Proyek

\* 'src/judgement\_aggregation.py' - Pembersih \& agregasi data penilai FiRA.

\* 'src/bert\_cross\_encoder.py' - Model re-ranking menggunakan MiniLM Cross-Encoder.

\* 'src/extractive\_qa.py' - Modul ekstraksi jawaban menggunakan RoBERTa SQuAD2.

\* 'notebooks/Neural\_IR\_Exercise.ipynb` - Jupyter Notebook utama untuk eksekusi di Google Colab.

\* 'data/' - Direktori penyimpanan dataset mentah dan hasil agregasi (qrels).



\## 🚀 Cara Menjalankan di Google Colab

1. Unggah (\*upload\*) seluruh folder proyek kelompok ('data', 'src', 'notebooks') ke dalam Google Drive kalian.
2. Buka file notebook 'notebooks/Neural\_IR\_Exercise.ipynb' melalui Google Colab.
3. Jalankan \*\*Cell 1 (Environment Setup)\*\* untuk melakukan instalasi pustaka dari 'requirements.txt' secara otomatis serta menghubungkan (\*mount\*) Google Drive ke lingkungan Colab.
4. Jalankan seluruh cell berikutnya secara berurutan untuk melihat proses algoritma, ekstraksi jawaban teks QA, hingga grafik evaluasi metrik akhir (P@10, MRR@10, NDCG@10).

