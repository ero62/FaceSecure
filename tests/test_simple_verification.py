#!/usr/bin/env python3
"""
Basit Yüz Doğrulama Testi
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
from models.face_embedder import FaceEmbedder
from models.face_mesh_detector import FaceMeshDetector
from models.database import FaceDatabase

def test_face_verification():
    """Basit yüz doğrulama testi"""
    print("🔍 Yüz Doğrulama Testi Başlatılıyor...")
    
    # Modelleri yükle
    print("📦 Modeller yükleniyor...")
    detector = FaceMeshDetector()
    embedder = FaceEmbedder(pca_model_path="models/saved/pca_arcface_model.joblib")
    db = FaceDatabase()
    
    # Veritabanındaki kullanıcıları listele
    all_users = db.get_all_embeddings()
    print(f"📊 Veritabanında {len(all_users)} kullanıcı bulundu")
    
    if not all_users:
        print("❌ Veritabanında hiç kullanıcı yok!")
        return
    
    # İlk kullanıcıyı al
    first_user_id, first_user_embedding = all_users[0]
    print(f"👤 Test kullanıcısı: {first_user_id}")
    
    # Kamera başlat
    print("📷 Kamera başlatılıyor...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera açılamadı!")
        return
    
    print("🔍 Yüz aranıyor... Kameraya bakın (5 saniye)")
    
    # 5 saniye boyunca yüz ara
    start_time = time.time()
    face_found = False
    
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Yüz sayısını kontrol et
        face_count = detector.count_faces(frame)
        
        if face_count == 1:
            print("✅ Yüz tespit edildi!")
            face_found = True
            
            # Yüz crop'u al
            face_crop = detector.get_face_crop(frame)
            if face_crop is not None:
                print("✅ Yüz crop'u alındı")
                
                # Embedding çıkar
                embedding = embedder.get_embedding(face_crop)
                if embedding is not None:
                    print("✅ Embedding çıkarıldı")
                    
                    # Benzerlik hesapla
                    similarity = embedder.calculate_similarity(embedding, first_user_embedding)
                    print(f"📊 Benzerlik: %{similarity:.2f}")
                    
                    # Sonuç
                    if similarity >= 70:
                        print("🎉 DOĞRULAMA BAŞARILI!")
                    else:
                        print("❌ DOĞRULAMA BAŞARISIZ!")
                    
                    break
                else:
                    print("❌ Embedding çıkarılamadı")
            else:
                print("❌ Yüz crop'u alınamadı")
        elif face_count > 1:
            print("⚠️ Birden fazla yüz tespit edildi")
        else:
            print("🔍 Yüz aranıyor...")
        
        time.sleep(0.5)
    
    if not face_found:
        print("❌ 5 saniye içinde yüz tespit edilemedi")
    
    cap.release()
    print("✅ Test tamamlandı")

if __name__ == "__main__":
    test_face_verification() 