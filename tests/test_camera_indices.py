# test_camera_indices.py
import cv2
import time

def test_all_camera_indices():
    print("🔍 Tüm Kamera İndeksleri Test Ediliyor...")
    print("=" * 60)
    
    # Test edilecek indeksler
    indices_to_test = [0, 1, 2, 3, 4, 5]
    
    working_cameras = []
    
    for idx in indices_to_test:
        print(f"\n📷 Kamera İndeksi {idx} Test Ediliyor...")
        
        try:
            cap = cv2.VideoCapture(idx)
            
            if not cap.isOpened():
                print(f"   ❌ Kamera {idx} açılamadı")
                continue
            
            print(f"   ✅ Kamera {idx} açıldı")
            
            # Kamera özelliklerini göster
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"   📊 Çözünürlük: {width:.0f}x{height:.0f}")
            print(f"   📊 FPS: {fps:.1f}")
            
            # Birkaç frame test et
            success_count = 0
            for i in range(3):
                ret, frame = cap.read()
                if ret and frame is not None:
                    success_count += 1
                    print(f"   Frame {i+1}: ✅ {frame.shape}")
                else:
                    print(f"   Frame {i+1}: ❌ Alınamadı")
                time.sleep(0.5)
            
            cap.release()
            
            if success_count > 0:
                working_cameras.append({
                    'index': idx,
                    'resolution': f"{width:.0f}x{height:.0f}",
                    'fps': fps,
                    'success_rate': f"{(success_count/3)*100:.1f}%"
                })
                print(f"   🎉 Kamera {idx} çalışıyor!")
            else:
                print(f"   ❌ Kamera {idx} görüntü alamıyor")
                
        except Exception as e:
            print(f"   ❌ Kamera {idx} hatası: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("📊 ÇALIŞAN KAMERALAR:")
    print("=" * 60)
    
    if working_cameras:
        for i, camera in enumerate(working_cameras, 1):
            print(f"{i}. İndeks {camera['index']}: {camera['resolution']} @ {camera['fps']:.1f}fps ({camera['success_rate']} başarı)")
    else:
        print("❌ Hiçbir kamera çalışmıyor!")
    
    print("\n💡 Öneriler:")
    print("1. OBS Virtual Camera'yı başlatın")
    print("2. Bilgisayar kamerasını OBS'e ekleyin")
    print("3. 'Start Virtual Camera' butonuna tıklayın")
    print("4. Testi tekrar çalıştırın")
    
    return working_cameras

def identify_obs_camera(working_cameras):
    """OBS Virtual Camera'yı tanımlamaya çalış"""
    print("\n🎥 OBS Virtual Camera Tanımlama...")
    
    if not working_cameras:
        print("❌ Çalışan kamera yok!")
        return None
    
    # OBS Virtual Camera genellikle yüksek çözünürlükte olur
    for camera in working_cameras:
        width, height = map(int, camera['resolution'].split('x'))
        
        # OBS Virtual Camera genellikle 1920x1080 veya 1280x720 olur
        if width >= 1280 and height >= 720:
            print(f"🎯 OBS Virtual Camera olabilir: İndeks {camera['index']} ({camera['resolution']})")
            return camera['index']
    
    # Eğer yüksek çözünürlük yoksa, ilk çalışan kamerayı öner
    print(f"🎯 İlk çalışan kamera: İndeks {working_cameras[0]['index']}")
    return working_cameras[0]['index']

if __name__ == "__main__":
    print("🚀 Kamera İndeks Testi Başlatılıyor...")
    
    # Tüm kameraları test et
    working_cameras = test_all_camera_indices()
    
    # OBS Virtual Camera'yı tanımla
    recommended_index = identify_obs_camera(working_cameras)
    
    if recommended_index is not None:
        print(f"\n✅ Önerilen kamera indeksi: {recommended_index}")
        print(f"💡 Bu indeksi app.py'de kullanabilirsiniz")
    else:
        print("\n❌ Uygun kamera bulunamadı!")
        print("💡 OBS Virtual Camera'yı başlatmayı deneyin") 