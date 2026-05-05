import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# SAYFA YAPILANDIRMASI 
st.set_page_config(page_title="Hücre Segmentasyon Analizi", layout="wide")

# CUSTOM FUNCTIONS
def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.0)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + (1 - dice_coef(y_true, y_pred))

# MODELİ YÜKLEME
@st.cache_resource
def load_segmentation_model():
    try:
        model = tf.keras.models.load_model(
            'hucre_segmentasyon_modeli_final.keras', 
            custom_objects={
                'bce_dice_loss': bce_dice_loss, 
                'dice_coef': dice_coef
            }
        )
        return model
    except Exception as e:
        st.error(f"⚠️ Model yüklenirken hata oluştu: {e}")
        return None

model = load_segmentation_model()

# BAŞLIK 
st.title("🔬 Hücre Segmentasyonu ve Morfolojik Analiz")
st.markdown("""
**Geliştiren:** Serdar ÖNAL | İnşaat Mühendisi & AI Araştırmacısı
*Bu uygulama, 20 final projesi hedefinin 5. adımı kapsamında geliştirilmiştir.*
""")

# YAN PANEL (SIDEBAR)
st.sidebar.header("⚙️ Analiz Ayarları")
threshold = st.sidebar.slider("Segmentasyon Eşiği (Threshold)", 0.0, 1.0, 0.5)
st.sidebar.info("Not: Doku karmaşası ve su damlacığı hatalarını önlemek için eşik değerini optimize edebilirsiniz.")

# DOSYA YÜKLEME 
uploaded_file = st.file_uploader("Bir mikroskop görüntüsü seçin...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # GÖRÜNTÜYÜ OKU VE GRİ TONLAMAYA (L) ÇEVİR (ValueError Çözümü)
    image_pil = Image.open(uploaded_file).convert('L') 
    img_array = np.array(image_pil)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Orijinal Görüntü (Grayscale)")
        st.image(image_pil, use_container_width=True)
    
    # MODEL TAHMİNİ (PREDICTION) 
    with st.spinner('Hücre dokusu analiz ediliyor...'):
        # Modelin beklediği boyuta getir (512x512)
        img_resized = cv2.resize(img_array, (512, 512))
        img_normalized = img_resized / 255.0
        
        # Giriş şeklini düzenle: (1, 512, 512, 1)
        img_input = np.expand_dims(img_normalized, axis=(0, -1))
        
        # Tahmin ve Maskeleme
        prediction = model.predict(img_input)[0]
        mask = (prediction > threshold).astype(np.uint8)
        
        # Hücre Sayımı (Kontur Analizi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cell_count = len(contours)
        
    with col2:
        st.subheader("🎭 Model Tahmini (Maske)")
        st.image(mask * 255, use_container_width=True)

    st.divider()
    
    # ANALİZ SONUÇLARI 
    st.subheader("📊 Analiz Sonuçları")
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.metric("Tahmin Edilen Hücre Sayısı", f"{cell_count} Adet")
    
    with m_col2:
        density = (np.sum(mask) / (mask.shape[0] * mask.shape[1])) * 100
        st.metric("Hücre Yoğunluğu", f"%{density:.2f}")
        
    with m_col3:
        st.metric("Seçilen Eşik Değeri", threshold)

   # OVERLAY (GÖRSEL DOĞRULAMA) 
    st.subheader("🔍 Detaylı İnceleme (Overlay)")
    
    #  Orijinal gri tonlamalı görüntüyü görselleştirme için RGB'ye çevirelim
    raw_img_resized = cv2.resize(img_array, (512, 512))
    overlay_img = cv2.cvtColor(raw_img_resized, cv2.COLOR_GRAY2RGB)
    
    #  HATA ÇÖZÜMÜ: Maskeyi (512, 512) -> (512, 512, 1) yaparak axis uyumu sağlayalım
    # mask_resized zaten (512, 512) boyutunda geliyor
    mask_boolean = (mask == 1).reshape(512, 512)
    
    #  Hücreleri yeşile boya (Boolean indexing artık axis 0 ve 1 üzerinde çalışacak)
    overlay_img[mask_boolean] = [0, 255, 0] 
    
    #  Şeffaflıkla birleştirme
    blended = cv2.addWeighted(overlay_img, 0.4, cv2.cvtColor(raw_img_resized, cv2.COLOR_GRAY2RGB), 0.6, 0)
    st.image(blended, caption="Hücre Sınırlarının Görsel Doğrulaması", use_container_width=True)

else:
    if model is None:
        st.error("Model dosyası bulunamadı. Lütfen 'hucre_segmentasyon_modeli_final.keras' dosyasını kontrol edin.")
    else:
        st.warning("Lütfen analiz için bir mikroskop görüntüsü yükleyin.")

# İMZA
st.divider()
st.caption("© 2026 İnşaat Mühendisi & Yapay Zeka Uygulayıcısı")