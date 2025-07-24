# test_camera.py
import cv2
import time

def test_camera():
    print("🔍 Kamera Testi Başlatılıyor...")
    
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
            
            # Birkaç frame al
            success_count = 0
            for i in range(3):
                ret, frame = cap.read()
                if ret:
                    success_count += 1
                    print(f"   Frame {i+1}: ✅ Alındı (Boyut: {frame.shape})")
                else:
                    print(f"   Frame {i+1}: ❌ Alınamadı")
                time.sleep(0.5)
            
            cap.release()
            
            if success_count > 0:
                print(f"   ✅ Kamera {idx} çalışıyor! ({success_count}/3 frame başarılı)")
                return idx
            else:
                print(f"   ❌ Kamera {idx} görüntü alamıyor")
                
        except Exception as e:
            print(f"   ❌ Kamera {idx} hatası: {e}")
            continue
    
    print("\n❌ Hiçbir kamera çalışmıyor!")
    return None

if __name__ == "__main__":
    working_camera = test_camera()
    if working_camera is not None:
        print(f"\n🎉 Çalışan kamera indeksi: {working_camera}")
    else:
        print("\n🔧 Kamera sorunu çözümleri:")
        print("1. Kameranızın başka bir uygulama tarafından kullanılmadığından emin olun")
        print("2. Kamera izinlerini kontrol edin")
        print("3. Kamera sürücülerini güncelleyin")
        print("4. Farklı bir kamera deneyin") 