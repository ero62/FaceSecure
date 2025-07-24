# test_face_detection.py
import cv2
import numpy as np
import mediapipe as mp
from models.face_mesh_detector import FaceMeshDetector

def test_face_detection_methods():
    print("🔍 Yüz Tespiti Yöntemleri Testi...")
    
    # Test görüntüsü oluştur (gri arka plan)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (128, 128, 128)  # Gri arka plan
    
    detector = FaceMeshDetector()
    
    print("\n📊 Test Görüntüsü Analizi:")
    print(f"   Görüntü boyutu: {test_image.shape}")
    
    # MediaPipe test
    mp_count = detector.count_faces(test_image)
    print(f"   MediaPipe yüz sayısı: {mp_count}")
    
    # OpenCV test
    opencv_count = detector.count_faces_opencv(test_image)
    print(f"   OpenCV yüz sayısı: {opencv_count}")
    
    # Kombine test
    combined_count = detector.count_faces(test_image)
    print(f"   Kombine yüz sayısı: {combined_count}")
    
    return detector

def test_camera_face_detection(detector):
    print("\n📷 Kamera ile Yüz Tespiti Testi...")
    
    cap = cv2.VideoCapture(1)  # Kamera indeksi 1
    
    if not cap.isOpened():
        print("❌ Kamera açılamadı!")
        return False
    
    print("✅ Kamera açıldı")
    
    # Kamera özelliklerini göster
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"   Çözünürlük: {width:.0f}x{height:.0f}")
    print(f"   FPS: {fps:.1f}")
    
    success_count = 0
    total_frames = 10
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"\n   Frame {i+1}:")
            
            # Frame özelliklerini göster
            print(f"     Boyut: {frame.shape}")
            print(f"     Veri tipi: {frame.dtype}")
            print(f"     Min/Max değerler: {frame.min()}/{frame.max()}")
            
            # Yüz tespiti
            mp_count = detector.count_faces(frame)
            opencv_count = detector.count_faces_opencv(frame)
            combined_count = detector.count_faces(frame)
            
            print(f"     MediaPipe: {mp_count} yüz")
            print(f"     OpenCV: {opencv_count} yüz")
            print(f"     Kombine: {combined_count} yüz")
            
            if combined_count > 0:
                success_count += 1
                print(f"     ✅ Yüz tespit edildi!")
            else:
                print(f"     ❌ Yüz tespit edilemedi")
        else:
            print(f"   Frame {i+1}: ❌ Alınamadı")
    
    cap.release()
    
    print(f"\n📊 Test Sonuçları:")
    print(f"   Toplam frame: {total_frames}")
    print(f"   Başarılı frame: {success_count}")
    print(f"   Başarı oranı: {(success_count/total_frames)*100:.1f}%")
    
    return success_count > 0

def test_face_cropping(detector):
    print("\n✂️ Yüz Kırpma Testi...")
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Kamera açılamadı!")
        return False
    
    for i in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"\n   Frame {i+1}:")
            
            # MediaPipe ile crop
            mp_crop = detector.get_face_crop(frame)
            if mp_crop is not None:
                print(f"     MediaPipe crop: ✅ {mp_crop.shape}")
            else:
                print(f"     MediaPipe crop: ❌")
            
            # OpenCV ile crop
            opencv_crop = detector.get_face_crop_opencv(frame)
            if opencv_crop is not None:
                print(f"     OpenCV crop: ✅ {opencv_crop.shape}")
            else:
                print(f"     OpenCV crop: ❌")
    
    cap.release()
    return True

def create_test_face_image():
    """Basit bir yüz benzeri şekil oluştur"""
    print("\n🎨 Test Yüz Görüntüsü Oluşturuluyor...")
    
    # 400x400 görüntü
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[:] = (200, 200, 200)  # Açık gri arka plan
    
    # Yüz benzeri oval çiz
    cv2.ellipse(img, (200, 200), (80, 100), 0, 0, 360, (150, 150, 150), -1)
    
    # Gözler
    cv2.circle(img, (170, 170), 8, (100, 100, 100), -1)
    cv2.circle(img, (230, 170), 8, (100, 100, 100), -1)
    
    # Ağız
    cv2.ellipse(img, (200, 240), (30, 10), 0, 0, 180, (100, 100, 100), 2)
    
    return img

def test_with_artificial_face(detector):
    print("\n🎭 Yapay Yüz ile Test...")
    
    # Yapay yüz oluştur
    test_face = create_test_face_image()
    
    print(f"   Yapay yüz boyutu: {test_face.shape}")
    
    # Yüz tespiti
    mp_count = detector.count_faces(test_face)
    opencv_count = detector.count_faces_opencv(test_face)
    combined_count = detector.count_faces(test_face)
    
    print(f"   MediaPipe: {mp_count} yüz")
    print(f"   OpenCV: {opencv_count} yüz")
    print(f"   Kombine: {combined_count} yüz")
    
    # Crop test
    mp_crop = detector.get_face_crop(test_face)
    opencv_crop = detector.get_face_crop_opencv(test_face)
    
    print(f"   MediaPipe crop: {'✅' if mp_crop is not None else '❌'}")
    print(f"   OpenCV crop: {'✅' if opencv_crop is not None else '❌'}")
    
    return combined_count > 0

if __name__ == "__main__":
    print("🚀 Kapsamlı Yüz Tespiti Testi Başlatılıyor...")
    print("=" * 60)
    
    # Detector oluştur
    detector = test_face_detection_methods()
    
    # Yapay yüz testi
    artificial_success = test_with_artificial_face(detector)
    
    # Kamera testi
    camera_success = test_camera_face_detection(detector)
    
    # Crop testi
    crop_success = test_face_cropping(detector)
    
    print("\n" + "=" * 60)
    print("📊 GENEL TEST SONUÇLARI:")
    print("=" * 60)
    print(f"✅ Yapay yüz testi: {'Başarılı' if artificial_success else 'Başarısız'}")
    print(f"✅ Kamera testi: {'Başarılı' if camera_success else 'Başarısız'}")
    print(f"✅ Crop testi: {'Başarılı' if crop_success else 'Başarısız'}")
    
    if not camera_success:
        print("\n💡 Kamera Testi Başarısız - Öneriler:")
        print("1. Kameraya daha yakın durun (20-40 cm)")
        print("2. İyi aydınlatma sağlayın")
        print("3. Yüzünüzü kameraya doğru çevirin")
        print("4. Farklı açılardan deneyin")
        print("5. Kameranın çalıştığından emin olun")
    
    print("\n�� Test tamamlandı!") 