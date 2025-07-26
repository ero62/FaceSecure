# simple_arcface_test.py
import cv2
import numpy as np
import sys
import os

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_arcface_simple():
    """
    ArcFace'in temel işlevselliğini test eder.
    """
    print("🔍 ArcFace Basit Test Başlatılıyor...")
    
    try:
        # 1. ArcFace modelini import et
        print("1. ArcFace modeli yükleniyor...")
        from models.arcface_model import ArcFace
        arcface = ArcFace()
        print("✅ ArcFace modeli yüklendi!")
        
        # 2. FaceEmbedder'ı test et
        print("2. FaceEmbedder test ediliyor...")
        from models.face_embedder import FaceEmbedder
        embedder = FaceEmbedder()
        print("✅ FaceEmbedder yüklendi!")
        
        # 3. Kamera testi
        print("3. Kamera testi başlatılıyor...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Kamera açılamadı!")
            return False
        
        print("Kameradan görüntü alınıyor...")
        
        # Tek bir frame al
        ret, frame = cap.read()
        if not ret:
            print("❌ Kameradan görüntü alınamadı!")
            cap.release()
            return False
        
        print(f"✅ Görüntü alındı! Boyut: {frame.shape}")
        
        # Yüz tespiti dene
        try:
            embedding = arcface.get_embedding(frame)
            if embedding is not None:
                print(f"✅ Yüz bulundu! Embedding boyutu: {len(embedding)}")
            else:
                print("⚠️ Yüz bulunamadı, bu normal olabilir")
        except Exception as e:
            print(f"⚠️ Embedding çıkarma hatası: {e}")
        
        cap.release()
        
        print("✅ ArcFace testi başarılı!")
        return True
        
    except Exception as e:
        print(f"❌ Test hatası: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_arcface_simple() 