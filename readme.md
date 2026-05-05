# 🔬 Hücre Segmentasyonu ve Morfolojik Analiz (Cell Segmentation)

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange?style=for-the-badge&logo=tensorflow)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV-blue?style=for-the-badge&logo=opencv)
![Python](https://img.shields.io/badge/Python-3.x-green?style=for-the-badge&logo=python)

Düşük kontrastlı biyolojik görüntülerde hücre tespiti ve sınırlarının belirlenmesi için özelleştirilmiş **U-Net** mimarisi kullanılarak geliştirilen derin öğrenme projesidir.

---

## 🚀 Proje Özeti
Bu çalışma, mikroskop altındaki hücrelerin otomatik olarak maskelenmesini amaçlar. Eğitim sürecinde karşılaşılan doku karmaşası (su damlacıkları, yapay kesikler vb.) ileri seviye optimizasyon teknikleriyle çözülmüştür.

*   **Model:** Custom U-Net Architecture
*   **Final Dice Skoru:** **0.8287** (TTA ve Post-Processing sonrası)
*   **Ham Model Skoru:** 0.7351

---

## 🛠️ Teknik Özellikler ve Başarı Faktörleri

### 1. Eğitim Dinamiği
*   **Öğrenme Oranı (LR) Yönetimi:** `ReduceLROnPlateau` ile 9. epokta LR **0.0002**'ye indirilerek hassas öğrenme sağlanmıştır.
*   **Stabilizasyon:** Eğitim kaybı (loss) **0.39** bandında sabitlenerek overfitting engellenmiştir.

### 2. Performans İyileştirme
*   **TTA (Test Time Augmentation):** Görüntü döndürme ve yansıtma teknikleriyle modelin geometrik tutarlılığı artırılmış, skor %12.7 iyileştirilmiştir.
*   **Eşik Optimizasyonu:** Matematiksel taramalar sonucunda en verimli eşik değeri **0.5** olarak belirlenmiştir.

---

## 📊 Hata Analizi (Error Analysis)
Modelin ilk versiyonlarında, yeşil elma veya hücre dokusu üzerindeki **yapay kesik izleri** ve **su damlacıkları** doku karmaşasına neden olmuştur. Bu durum, modelin formdan ziyade doku (texture) özelliklerine odaklandığını göstermiş; eğitim stratejisi buna göre güncellenmiştir.

---

## 📂 Dosya Yapısı
*   `hucre_segmentasyon.ipynb`: Tüm eğitim ve analiz kodları.
*   `hucre_segmentasyon_modeli_final.keras`: Kaydedilmiş final model dosyası.
*   `images/`: Eğitim süreci ve TTA sonuç grafiklerini içerir.

---

## ✍️ Hazırlayan

**Serdar ÖNAL**  
*İnşaat Mühendisi (20+ Yıl Deneyim) & Yapay Zeka Araştırmacısı*  
*Derin Öğrenme ve Bilgisayarlı Görü Uzmanı*

---