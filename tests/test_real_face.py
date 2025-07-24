#!/usr/bin/env python3
"""
Gerçek Yüz Testi
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from models.arcface_model import ArcFace
from models.face_mesh_detector import FaceMeshDetector

def test_real_face():
    """Gerçek yüz görüntüsü ile test"""
    print("🔍 Gerçek Yüz Testi Başlatılıyor...")
    
    # ArcFace modelini yükle
    print("📦 ArcFace modeli yükleniyor...")
    try:
        arcface = ArcFace()
        print("✅ ArcFace modeli yüklendi")
    except Exception as e:
        print(f"❌ ArcFace modeli yüklenemedi: {e}")
        return
    
    # Face detector yükle
    print("📦 Face detector yükleniyor...")
    try:
        detector = FaceMeshDetector()
        print("✅ Face detector yüklendi")
    except Exception as e:
        print(f"❌ Face detector yüklenemedi: {e}")
        return
    
    # Kamera başlat
    print("📷 Kamera başlatılıyor...")
    cap = cv2.VideoCapture(1)  # OBS Virtual Camera
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Normal kamera
        if cap.isOpened():
            print("✅ Normal kamera açıldı")
        else:
            print("❌ Hiçbir kamera açılamadı!")
            return
    else:
        print("✅ OBS Virtual Camera açıldı")
    
    print("🔍 10 saniye boyunca yüz aranacak...")
    print("💡 Kameraya bakın ve yüzünüzü gösterin!")
    
    # 10 saniye boyunca yüz ara
    start_time = cv2.getTickCount()
    frame_count = 0
    face_found = False
    
    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < 10:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # Yüz sayısını kontrol et
        face_count = detector.count_faces(frame)
        
        if face_count > 0:
            print(f"✅ Frame {frame_count}: {face_count} yüz tespit edildi!")
            
            # Yüz crop'u al
            face_crop = detector.get_face_crop(frame)
            if face_crop is not None:
                print(f"✅ Yüz crop alındı: {face_crop.shape}")
                
                # ArcFace ile test et
                print("🔍 ArcFace test ediliyor...")
                
                # 1. get_face_info testi
                face_info = arcface.get_face_info(face_crop)
                if face_info is not None:
                    print(f"✅ get_face_info başarılı!")
                    print(f"   Embedding shape: {face_info.embedding.shape}")
                    print(f"   Embedding dtype: {face_info.embedding.dtype}")
                    print(f"   Embedding range: {face_info.embedding.min():.3f} - {face_info.embedding.max():.3f}")
                    face_found = True
                    break
                else:
                    print("❌ get_face_info başarısız")
                
                # 2. get_embedding testi
                embedding = arcface.get_embedding(face_crop)
                if embedding is not None:
                    print(f"✅ get_embedding başarılı!")
                    print(f"   Embedding shape: {embedding.shape}")
                    print(f"   Embedding dtype: {embedding.dtype}")
                    print(f"   Embedding range: {embedding.min():.3f} - {embedding.max():.3f}")
                    face_found = True
                    break
                else:
                    print("❌ get_embedding başarısız")
            else:
                print("❌ Yüz crop alınamadı")
        else:
            if frame_count % 30 == 0:  # Her 30 frame'de bir mesaj
                print(f"🔍 Frame {frame_count}: Yüz aranıyor...")
        
        # Görüntüyü göster
        cv2.imshow('Yüz Testi', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if not face_found:
        print("❌ 10 saniye içinde yüz tespit edilemedi")
        print("💡 Lütfen şunları deneyin:")
        print("   • Kameraya daha yakın durun")
        print("   • Daha iyi aydınlatma sağlayın")
        print("   • Yüzünüzü kameraya doğru çevirin")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Gerçek yüz testi tamamlandı")

if __name__ == "__main__":
    test_real_face() 