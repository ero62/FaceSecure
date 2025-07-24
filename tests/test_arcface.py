# test_arcface.py
import cv2
import numpy as np
import sys
import os

def test_arcface_basic():
    """
    ArcFace'in temel işlevselliğini test eder.
    """
    print("🔍 ArcFace Temel Test Başlatılıyor...")
    
    try:
        # 1. Kütüphaneleri import et
        print("1. Kütüphaneler import ediliyor...")
        from models.arcface_model import ArcFace
        from models.face_embedder import FaceEmbedder
        print("✅ Import başarılı!")
        
        # 2. Test görüntüsü oluştur
        print("2. Test görüntüsü oluşturuluyor...")
        test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        print("✅ Test görüntüsü oluşturuldu!")
        
        # 3. ArcFace modelini test et
        print("3. ArcFace modeli test ediliyor...")
        arcface = ArcFace()
        embedding = arcface.get_embedding(test_img)
        
        if embedding is not None:
            print(f"✅ ArcFace embedding çıkarıldı! Boyut: {len(embedding)}")
        else:
            print("❌ ArcFace embedding çıkarılamadı!")
            return False
        
        # 4. FaceEmbedder'ı test et
        print("4. FaceEmbedder test ediliyor...")
        embedder = FaceEmbedder()
        embedding2 = embedder.get_embedding(test_img)
        
        if embedding2 is not None:
            print(f"✅ FaceEmbedder embedding çıkarıldı! Boyut: {len(embedding2)}")
        else:
            print("❌ FaceEmbedder embedding çıkarılamadı!")
            return False
        
        # 5. Benzerlik hesaplama testi
        print("5. Benzerlik hesaplama test ediliyor...")
        similarity = embedder.calculate_similarity(embedding, embedding2)
        print(f"✅ Benzerlik hesaplandı: %{similarity:.2f}")
        
        # 6. Yüz kalite testi
        print("6. Yüz kalite testi...")
        quality = embedder.get_face_quality(test_img)
        print(f"✅ Yüz kalite skoru: {quality:.4f}")
        
        print("\n🎉 Tüm testler başarılı!")
        return True
        
    except Exception as e:
        print(f"❌ Test hatası: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_camera():
    """
    Kameradan görüntü alarak ArcFace'i test eder.
    """
    print("\n📷 Kamera Testi Başlatılıyor...")
    
    try:
        from models.arcface_model import ArcFace
        from models.face_embedder import FaceEmbedder
        
        # Kamera aç
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Kamera açılamadı!")
            return False
        
        arcface = ArcFace()
        embedder = FaceEmbedder()
        
        print("Kameraya bakın ve 'q' tuşuna basın...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Kameradan görüntü alınamadı!")
                break
            
            # Görüntüyü göster
            cv2.imshow('ArcFace Test', frame)
            
            # 'q' tuşuna basılırsa çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Her 30 frame'de bir embedding çıkar
            if cv2.getTickCount() % 30 == 0:
                try:
                    embedding = arcface.get_embedding(frame)
                    if embedding is not None:
                        print(f"✅ Kamera embedding: {len(embedding)} boyut")
                except:
                    pass
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Kamera testi tamamlandı!")
        return True
        
    except Exception as e:
        print(f"❌ Kamera test hatası: {e}")
        return False

def main():
    """
    Ana test fonksiyonu.
    """
    print("🚀 ArcFace Test Başlatılıyor...")
    print("=" * 50)
    
    # Temel test
    if not test_arcface_basic():
        print("❌ Temel test başarısız!")
        return
    
    # Kamera testi (opsiyonel)
    camera_test = input("\n📷 Kamera testi yapmak ister misiniz? (y/n): ").lower()
    if camera_test == 'y':
        test_with_camera()
    
    print("\n" + "=" * 50)
    print("🎉 ArcFace testi başarıyla tamamlandı!")
    print("Artık uygulamayı başlatabilirsiniz: streamlit run app.py")

if __name__ == "__main__":
    main() 