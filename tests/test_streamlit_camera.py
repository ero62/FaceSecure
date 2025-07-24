# test_streamlit_camera.py
import cv2
import time
import numpy as np

def test_camera_in_streamlit():
    """Streamlit ortamında kamera erişimini test eder"""
    print("🔍 Streamlit Kamera Testi Başlatılıyor...")
    
    # Farklı kamera indekslerini dene
    camera_indices = [0, 1, 2, -1]
    
    for idx in camera_indices:
        print(f"\n📷 Kamera indeksi {idx} test ediliyor...")
        
        try:
            # Kamera aç
            cap = cv2.VideoCapture(idx)
            print(f"   Kamera açıldı: {cap.isOpened()}")
            
            if not cap.isOpened():
                print(f"   ❌ Kamera {idx} açılamadı!")
                cap.release()
                continue
            
            # Kamera özelliklerini kontrol et
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"   📐 Çözünürlük: {width}x{height}, FPS: {fps}")
            
            # Birkaç frame al
            success_count = 0
            for i in range(5):
                ret, frame = cap.read()
                if ret:
                    success_count += 1
                    print(f"   Frame {i+1}: ✅ Alındı (Boyut: {frame.shape})")
                    
                    # Frame'in boş olup olmadığını kontrol et
                    if frame is not None and frame.size > 0:
                        mean_val = np.mean(frame)
                        print(f"      Ortalama piksel değeri: {mean_val:.2f}")
                        if mean_val < 10:
                            print(f"      ⚠️ Frame çok karanlık olabilir")
                    else:
                        print(f"      ❌ Frame boş")
                else:
                    print(f"   Frame {i+1}: ❌ Alınamadı")
                time.sleep(0.2)
            
            cap.release()
            
            if success_count > 0:
                print(f"   ✅ Kamera {idx} çalışıyor! ({success_count}/5 frame başarılı)")
                return idx
            else:
                print(f"   ❌ Kamera {idx} görüntü alamıyor")
                
        except Exception as e:
            print(f"   ❌ Kamera {idx} hatası: {e}")
            continue
    
    print("\n❌ Hiçbir kamera çalışmıyor!")
    return None

def test_face_detection():
    """Yüz tespiti testi"""
    print("\n🔍 Yüz Tespiti Testi...")
    
    try:
        from models.face_mesh_detector import FaceMeshDetector
        detector = FaceMeshDetector()
        print("✅ FaceMeshDetector yüklendi")
        
        # Test görüntüsü oluştur
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (128, 128, 128)  # Gri arka plan
        
        face_count = detector.count_faces(test_image)
        print(f"   Test görüntüsünde {face_count} yüz tespit edildi")
        
        return True
        
    except Exception as e:
        print(f"❌ Yüz tespiti hatası: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Streamlit Kamera Testi Başlatılıyor...")
    
    # Kamera testi
    working_camera = test_camera_in_streamlit()
    
    # Yüz tespiti testi
    face_detection_ok = test_face_detection()
    
    print("\n" + "="*50)
    print("📊 TEST SONUÇLARI:")
    print("="*50)
    
    if working_camera is not None:
        print(f"✅ Çalışan kamera indeksi: {working_camera}")
    else:
        print("❌ Kamera erişimi başarısız")
    
    if face_detection_ok:
        print("✅ Yüz tespiti çalışıyor")
    else:
        print("❌ Yüz tespiti başarısız")
    
    print("\n🔧 Öneriler:")
    print("1. Kameranızın başka bir uygulama tarafından kullanılmadığından emin olun")
    print("2. Windows kamera izinlerini kontrol edin")
    print("3. Kamera sürücülerini güncelleyin")
    print("4. Farklı bir kamera deneyin") 