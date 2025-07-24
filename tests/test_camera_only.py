#!/usr/bin/env python3
"""
Sadece Kamera Testi
"""

import cv2
import time

def test_camera():
    """Sadece kamera testi"""
    print("📷 Kamera Testi Başlatılıyor...")
    
    # Kamera başlat
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera açılamadı!")
        return
    
    print("✅ Kamera açıldı!")
    print("🔍 5 saniye boyunca görüntü alınacak...")
    
    # 5 saniye boyunca görüntü al
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            print(f"Frame {frame_count}: {frame.shape}")
            
            # Görüntüyü göster
            cv2.imshow('Kamera Testi', frame)
            cv2.waitKey(1)
        else:
            print("❌ Frame alınamadı!")
        
        time.sleep(0.1)
    
    print(f"✅ {frame_count} frame alındı")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera() 