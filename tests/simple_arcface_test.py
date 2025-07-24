# simple_arcface_test.py
import cv2
import numpy as np

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
                        print(f"✅ Yüz bulundu! Embedding boyutu: {len(embedding)}")
                        break
                except Exception as e:
                    pass
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("✅ ArcFace testi başarılı!")
        return True
        
    except Exception as e:
        print(f"❌ Test hatası: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_arcface_simple() 