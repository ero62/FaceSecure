# Yüz Tanıma Sistemi (Face Recognition System)

ArcFace tabanlı yüz tanıma sistemi - Streamlit arayüzü ve Flask API ile.

## 🏗️ Proje Yapısı

```
bitirme-projesi/
├── app.py                 # Ana Streamlit uygulaması
├── api/                   # Flask API
│   └── main.py           # API endpoint'leri
├── models/               # Model dosyaları
│   ├── arcface_model.py  # ArcFace model wrapper
│   ├── face_embedder.py  # Embedding çıkarımı
│   ├── face_mesh_detector.py # Yüz tespiti
│   ├── database.py       # MongoDB işlemleri
│   └── saved/            # Kaydedilmiş modeller
│       ├── pca_arcface_model.joblib
│       └── facenet_keras.h5
├── tests/                # Test dosyaları
│   ├── test_api.py
│   ├── test_face_detection.py
│   └── ...
├── config/               # Konfigürasyon
│   └── setup_arcface.py
├── logs/                 # Log dosyaları
└── requirements.txt      # Python bağımlılıkları
```

## 🚀 Kurulum

1. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

2. **MongoDB'yi başlatın:**
```bash
# MongoDB servisini başlatın
```

3. **ArcFace modellerini indirin:**
```bash
python config/setup_arcface.py
```

## 🎯 Kullanım

### Streamlit Arayüzü
```bash
streamlit run app.py
```
- **URL**: http://localhost:8501
- **Admin**: `admin123`

### Flask API
```bash
python -m api.main
```
- **URL**: http://localhost:5000
- **Admin**: `aeren` / `eren1234`

## 🔧 Özellikler

- ✅ **ArcFace** yüz tanıma
- ✅ **10+ poz** kayıt sistemi
- ✅ **PCA** boyut indirgeme
- ✅ **MongoDB** veritabanı
- ✅ **JWT** API koruması
- ✅ **Streamlit** arayüzü
- ✅ **Hata logları**

## 📊 Test

```bash
# API testi
python tests/test_api_simple.py

# Yüz tespit testi
python tests/test_face_detection.py

# Kamera testi
python tests/test_camera_indices.py
```

## 🐳 Docker

```bash
docker build -t face-recognition .
docker run -p 8501:8501 face-recognition
``` 