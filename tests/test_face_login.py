import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time
from models.face_embedder import FaceEmbedder
from models.database import FaceDatabase
from models.face_mesh_detector import FaceMeshDetector

def test_face_login():
    """Yüz tanıma ile giriş testi"""
    print("🔍 Yüz Tanıma Giriş Testi Başlatılıyor...")
    
    # 1. Sistemleri başlat
    print("\n1️⃣ Sistemler başlatılıyor...")
    try:
        embedder = FaceEmbedder(pca_model_path="models/saved/pca_arcface_model.joblib")
        detector = FaceMeshDetector()
        db = FaceDatabase()
        print("   ✅ Sistemler başarıyla başlatıldı")
    except Exception as e:
        print(f"   ❌ Sistem başlatma hatası: {e}")
        return False
    
    # 2. Veritabanı durumu
    print("\n2️⃣ Veritabanı kontrolü...")
    users = db.get_all_embeddings()
    print(f"   Toplam kullanıcı: {len(users)}")
    print(f"   Örnek kullanıcılar: {[uid for uid, _ in users[:3]]}")
    
    # 3. Kamera testi
    print("\n3️⃣ Kamera testi...")
    cap = cv2.VideoCapture(1)  # OBS Virtual Camera
    if not cap.isOpened():
        print("   ⚠️ OBS Virtual Camera bulunamadı, fiziksel kamera deneniyor...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("   ❌ Kamera açılamadı!")
        return False
    
    print("   ✅ Kamera başarıyla açıldı")
    
    # 4. Yüz tespit testi
    print("\n4️⃣ Yüz tespit testi...")
    ret, frame = cap.read()
    if not ret:
        print("   ❌ Kameradan görüntü alınamadı!")
        cap.release()
        return False
    
    # Yüz tespit et
    face_img = detector.get_face_crop(frame)
    if face_img is None:
        print("   ⚠️ Yüz tespit edilemedi! Lütfen kameraya bakın.")
        print("   💡 Test için yüzünüzü kameraya gösterin ve Enter'a basın...")
        input()
        ret, frame = cap.read()
        face_img = detector.get_face_crop(frame)
    
    if face_img is None:
        print("   ❌ Yüz tespit edilemedi!")
        cap.release()
        return False
    
    print(f"   ✅ Yüz tespit edildi (boyut: {face_img.shape})")
    
    # 5. Embedding çıkarımı
    print("\n5️⃣ Embedding çıkarımı...")
    embedding = embedder.extract_embedding(face_img)
    
    if embedding is None:
        print("   ❌ Embedding çıkarılamadı!")
        cap.release()
        return False
    
    print(f"   ✅ Embedding çıkarıldı (boyut: {embedding.shape})")
    
    # 6. Kullanıcı eşleştirme
    print("\n6️⃣ Kullanıcı eşleştirme...")
    best_match = None
    best_similarity = 0
    
    for user_id, stored_embedding in users:
        similarity = embedder.calculate_similarity(embedding, stored_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = user_id
    
    print(f"   En iyi eşleşme: {best_match}")
    print(f"   Benzerlik skoru: %{best_similarity:.2f}")
    
    # 7. Giriş kararı
    print("\n7️⃣ Giriş kararı...")
    threshold = 0.7  # %70 eşik değeri
    
    if best_similarity >= threshold:
        print(f"   ✅ GİRİŞ BAŞARILI!")
        print(f"   👤 Kullanıcı: {best_match}")
        print(f"   📊 Benzerlik: %{best_similarity:.2f}")
        print(f"   🎯 Eşik değeri: %{threshold*100:.0f}")
        
        # Log başarılı giriş
        with open("logs/successful_logins.log", "a") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - Başarılı giriş: {best_match} (Benzerlik: %{best_similarity:.2f})\n")
        
        result = True
    else:
        print(f"   ❌ GİRİŞ REDDEDİLDİ!")
        print(f"   📊 En yüksek benzerlik: %{best_similarity:.2f}")
        print(f"   🎯 Gerekli eşik: %{threshold*100:.0f}")
        
        # Log başarısız giriş
        with open("logs/failed_logins.log", "a") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - Başarısız giriş: {best_match if best_match else 'Bilinmeyen'} (Benzerlik: %{best_similarity:.2f})\n")
        
        result = False
    
    # 8. Temizlik
    cap.release()
    cv2.destroyAllWindows()
    
    return result

if __name__ == "__main__":
    success = test_face_login()
    if success:
        print("\n🎉 TEST BAŞARILI! Yüz tanıma ile giriş sistemi çalışıyor!")
    else:
        print("\n❌ TEST BAŞARISIZ! Yüz tanıma ile giriş sistemi çalışmıyor!") 