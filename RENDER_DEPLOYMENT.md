# Render Deployment Rehberi

Bu proje Render'da canlıya çekilmek için hazırlanmıştır.

## Özellikler

- ✅ Render'da çalışacak şekilde optimize edilmiş
- ✅ Kamera erişimi olmadığı için dosya yükleme özelliği
- ✅ Streamlit uygulaması
- ✅ Yüz tanıma sistemi

## Render'da Deployment Adımları

### 1. GitHub'a Yükleme
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git
git push -u origin main
```

### 2. Render'da Yeni Servis Oluşturma

1. [Render Dashboard](https://dashboard.render.com/)'a gidin
2. "New +" butonuna tıklayın
3. "Web Service" seçin
4. GitHub repository'nizi bağlayın
5. Aşağıdaki ayarları yapın:

#### Temel Ayarlar:
- **Name**: `yuz-tanima-sistemi`
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

#### Environment Variables (Gerekirse):
- `RENDER`: `true`

### 3. Deployment

Render otomatik olarak deployment'ı başlatacaktır. İlk build biraz zaman alabilir (5-10 dakika).

## Kullanım

Render'da çalışan uygulama:
- Kamera erişimi olmadığı için fotoğraf yükleme kullanır
- Kullanıcı kaydı için fotoğraf yükleyin
- Yüz doğrulama için fotoğraf yükleyin

## Sorun Giderme

### Build Hatası
- `requirements.txt` dosyasının doğru olduğundan emin olun
- Python versiyonunun 3.10 olduğunu kontrol edin

### Runtime Hatası
- Logları kontrol edin
- Environment variables'ları kontrol edin

### Model Dosyaları
- `models/saved/` klasörünün oluşturulduğundan emin olun
- Gerekli model dosyalarının yüklendiğini kontrol edin

## Notlar

- Render'da kamera erişimi olmadığı için dosya yükleme özelliği kullanılır
- İlk kullanımda model dosyaları otomatik olarak indirilir
- Uygulama 15 dakika inaktif kaldıktan sonra uyku moduna geçer 