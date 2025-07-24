# setup_arcface.py
import os
import sys
import subprocess
import cv2
import numpy as np

def install_requirements():
    """
    Gerekli kütüphaneleri yükler.
    """
    print("Gerekli kütüphaneler yükleniyor...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Kütüphaneler başarıyla yüklendi!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Kütüphane yükleme hatası: {e}")
        return False
    return True

def test_arcface_import():
    """
    ArcFace kütüphanesinin doğru yüklendiğini test eder.
    """
    print("ArcFace kütüphanesi test ediliyor...")
    try:
        import insightface
        from insightface.app import FaceAnalysis
        print("✅ ArcFace kütüphanesi başarıyla import edildi!")
        return True
    except ImportError as e:
        print(f"❌ ArcFace import hatası: {e}")
        return False

def test_arcface_model():
    """
    ArcFace modelinin yüklenip yüklenmediğini test eder.
    """
    print("ArcFace modeli test ediliyor...")
    try:
        from models.arcface_model import ArcFace
        
        # Test görüntüsü oluştur
        test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        # ArcFace modelini yükle
        arcface = ArcFace()
        
        # Test embedding'i çıkar
        embedding = arcface.get_embedding(test_img)
        
        if embedding is not None:
            print(f"✅ ArcFace modeli başarıyla çalışıyor! Embedding boyutu: {len(embedding)}")
            return True
        else:
            print("❌ ArcFace modeli embedding çıkaramadı!")
            return False
            
    except Exception as e:
        print(f"❌ ArcFace model test hatası: {e}")
        return False

def test_face_embedder():
    """
    FaceEmbedder sınıfının ArcFace ile çalışıp çalışmadığını test eder.
    """
    print("FaceEmbedder test ediliyor...")
    try:
        from models.face_embedder import FaceEmbedder
        
        # Test görüntüsü oluştur
        test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        # FaceEmbedder'ı test et
        embedder = FaceEmbedder()
        embedding = embedder.get_embedding(test_img)
        
        if embedding is not None:
            print(f"✅ FaceEmbedder başarıyla çalışıyor! Embedding boyutu: {len(embedding)}")
            return True
        else:
            print("❌ FaceEmbedder embedding çıkaramadı!")
            return False
            
    except Exception as e:
        print(f"❌ FaceEmbedder test hatası: {e}")
        return False

def main():
    """
    Ana kurulum ve test fonksiyonu.
    """
    print("🚀 ArcFace Kurulum ve Test Başlatılıyor...")
    print("=" * 50)
    
    # 1. Kütüphaneleri yükle
    if not install_requirements():
        print("❌ Kurulum başarısız! Kütüphaneler yüklenemedi.")
        return
    
    # 2. ArcFace import testi
    if not test_arcface_import():
        print("❌ Kurulum başarısız! ArcFace kütüphanesi yüklenemedi.")
        return
    
    # 3. ArcFace model testi
    if not test_arcface_model():
        print("❌ Kurulum başarısız! ArcFace modeli çalışmıyor.")
        return
    
    # 4. FaceEmbedder testi
    if not test_face_embedder():
        print("❌ Kurulum başarısız! FaceEmbedder çalışmıyor.")
        return
    
    print("=" * 50)
    print("🎉 ArcFace kurulumu başarıyla tamamlandı!")
    print("\n📝 Sonraki adımlar:")
    print("1. Streamlit uygulamasını başlatın: streamlit run app.py")
    print("2. Kullanıcı kaydı yapın")
    print("3. PCA modelini eğitin: python models/train_arcface_pca.py")
    print("4. Yüz doğrulama testini yapın")

if __name__ == "__main__":
    main() 