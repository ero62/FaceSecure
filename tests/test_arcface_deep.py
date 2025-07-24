#!/usr/bin/env python3
"""
ArcFace Derin Test
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from models.arcface_model import ArcFace

def test_arcface_model():
    """ArcFace modelini derinlemesine test et"""
    print("🔍 ArcFace Model Derin Testi Başlatılıyor...")
    
    # ArcFace modelini yükle
    print("📦 ArcFace modeli yükleniyor...")
    try:
        arcface = ArcFace()
        print("✅ ArcFace modeli yüklendi")
    except Exception as e:
        print(f"❌ ArcFace modeli yüklenemedi: {e}")
        return
    
    # Test görüntüsü oluştur
    print("🖼️ Test görüntüsü oluşturuluyor...")
    test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    print(f"✅ Test görüntüsü oluşturuldu: {test_img.shape}")
    
    # 1. get_face_info testi
    print("\n🔍 1. get_face_info testi...")
    try:
        face_info = arcface.get_face_info(test_img)
        if face_info is not None:
            print(f"✅ get_face_info başarılı: {face_info}")
            print(f"   Embedding shape: {face_info.embedding.shape}")
            print(f"   Embedding dtype: {face_info.embedding.dtype}")
            print(f"   Embedding range: {face_info.embedding.min():.3f} - {face_info.embedding.max():.3f}")
        else:
            print("❌ get_face_info None döndü")
    except Exception as e:
        print(f"❌ get_face_info hatası: {e}")
    
    # 2. get_embedding testi
    print("\n🔍 2. get_embedding testi...")
    try:
        embedding = arcface.get_embedding(test_img)
        if embedding is not None:
            print(f"✅ get_embedding başarılı: {embedding.shape}")
            print(f"   Embedding dtype: {embedding.dtype}")
            print(f"   Embedding range: {embedding.min():.3f} - {embedding.max():.3f}")
        else:
            print("❌ get_embedding None döndü")
    except Exception as e:
        print(f"❌ get_embedding hatası: {e}")
    
    # 3. Farklı boyutlarda test
    print("\n🔍 3. Farklı boyutlarda test...")
    sizes = [(64, 64), (96, 96), (112, 112), (128, 128), (160, 160)]
    
    for size in sizes:
        print(f"   Boyut {size} test ediliyor...")
        test_img_resized = cv2.resize(test_img, size)
        try:
            embedding = arcface.get_embedding(test_img_resized)
            if embedding is not None:
                print(f"   ✅ {size}: Başarılı - {embedding.shape}")
            else:
                print(f"   ❌ {size}: None döndü")
        except Exception as e:
            print(f"   ❌ {size}: Hata - {e}")
    
    # 4. InsightFace doğrudan testi
    print("\n🔍 4. InsightFace doğrudan testi...")
    try:
        import insightface
        from insightface.app import FaceAnalysis
        
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # RGB'ye çevir
        rgb_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        
        faces = app.get(rgb_img)
        print(f"✅ InsightFace doğrudan: {len(faces)} yüz bulundu")
        
        if len(faces) > 0:
            face = faces[0]
            print(f"   Embedding shape: {face.embedding.shape}")
            print(f"   Embedding dtype: {face.embedding.dtype}")
            print(f"   Embedding range: {face.embedding.min():.3f} - {face.embedding.max():.3f}")
        else:
            print("   ❌ Hiç yüz bulunamadı")
            
    except Exception as e:
        print(f"❌ InsightFace doğrudan hatası: {e}")
    
    print("\n✅ ArcFace derin testi tamamlandı")

if __name__ == "__main__":
    test_arcface_model() 