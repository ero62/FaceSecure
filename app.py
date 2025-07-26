import streamlit as st
import cv2
import numpy as np
from models.face_mesh_detector import FaceMeshDetector
from models.face_embedder import FaceEmbedder
from models.database import FaceDatabase
import os
from datetime import datetime
import time

ADMIN_PASSWORD = "admin123"

st.set_page_config(page_title="Yüz Tanıma Sistemi", layout="centered")
st.title("Yüz Tanıma Sistemi")

menu = ["Kullanıcı Kaydı (Admin)", "Yüz Doğrulama", "Kullanıcı Sil (Admin)", "Hatalı Giriş Logları"]
choice = st.sidebar.selectbox("İşlem Seçin", menu)

def get_camera_frame():
    """
    Kameradan bir kare alır.
    """
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        st.error("Kamera açılamadı!")
        return None
    return frame

def show_log_file():
    """
    Hatalı giriş loglarını gösterir.
    """
    log_path = os.path.join("logs", "failed_logins.log")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            logs = f.read()
        st.text_area("Hatalı Giriş Logları", logs, height=200)
    else:
        st.info("Henüz log kaydı yok.")

def log_failed_attempt(user_id, similarity, ip="localhost"):
    """
    Hatalı girişleri timestamp, user_id, IP ve benzerlik ile loglar.
    """
    log_dir = os.path.join("logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'failed_logins.log')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f'{timestamp}, {user_id}, {ip}, %{similarity:.2f}\n')

if choice == "Kullanıcı Kaydı (Admin)":
    st.header("Kullanıcı Kaydı (Admin)")
    admin_pass = st.text_input("Admin Parolası", type="password")
    user_name = st.text_input("Kullanıcı Adı")
    if st.button("Kameradan Kayıt Al", key="enroll_button"):
        if admin_pass != ADMIN_PASSWORD:
            st.error("Yetkisiz erişim!")
        elif not user_name:
            st.error("Kullanıcı adı girilmelidir.")
        else:
            # Kamera başlat - Önce OBS Virtual Camera, sonra bilgisayar kamerası dene
            cap = cv2.VideoCapture(1)  # OBS Virtual Camera (İndeks 1)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)  # Bilgisayar kamerası (İndeks 0)
                if cap.isOpened():
                    st.info("📷 OBS Virtual Camera bulunamadı, bilgisayar kamerası kullanılıyor")
                else:
                    st.error("❌ Hiçbir kamera bulunamadı!")
                    cap.release()
                    st.stop()
            else:
                st.success("📷 OBS Virtual Camera kullanılıyor")
            detector = FaceMeshDetector()
            embedder = FaceEmbedder()
            st.info("🎥 Kameraya bakın ve tek bir yüzünüzün görünmesini sağlayın...")
            
            # Basit bir yaklaşım - tek seferlik kamera görüntüsü
            st.info("📷 Kamera görüntüsü alınıyor...")
            
            # Kamera durumunu kontrol et
            if not cap.isOpened():
                st.error("❌ Kamera açılamadı!")
                cap.release()
            else:
                # Kamera özelliklerini göster
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                st.info(f"📷 Kamera çözünürlüğü: {width:.0f}x{height:.0f}")
                
                # Kullanıcı talimatları
                st.success("🎯 Yüz Tespiti İpuçları:")
                st.write("• Kameraya yaklaşın (20-40 cm mesafe)")
                st.write("• İyi aydınlatma sağlayın")
                st.write("• Yüzünüzün tam görünür olduğundan emin olun")
                st.write("• Farklı açılardan deneyin")
                st.write("• Yüzünüzü kameraya doğru çevirin")
                
                # Görsel debug için ilk frame'i göster
                st.info("🔍 Görsel Debug: İlk frame'i gösteriliyor...")
                ret, debug_frame = cap.read()
                if ret and debug_frame is not None:
                    st.image(cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB), 
                           channels="RGB", 
                           caption="🔍 Debug: Kameradan alınan ilk frame")
                    
                    # Frame özelliklerini göster
                    st.write(f"📊 Frame Özellikleri:")
                    st.write(f"• Boyut: {debug_frame.shape}")
                    st.write(f"• Veri tipi: {debug_frame.dtype}")
                    st.write(f"• Min/Max değerler: {debug_frame.min()}/{debug_frame.max()}")
                    
                    # Yüz tespiti sonuçlarını göster
                    mp_count = detector.count_faces(debug_frame)
                    opencv_count = detector.count_faces_opencv(debug_frame)
                    
                    st.write(f"🔍 Yüz Tespiti Sonuçları:")
                    st.write(f"• MediaPipe: {mp_count} yüz")
                    st.write(f"• OpenCV: {opencv_count} yüz")
                    
                    if mp_count == 0 and opencv_count == 0:
                        st.warning("⚠️ Hiçbir yöntemle yüz tespit edilemedi!")
                        st.info("💡 Lütfen kameraya bakın ve yüzünüzün görünür olduğundan emin olun")
                else:
                    st.error("❌ Debug frame alınamadı!")
                
                # Birkaç frame al ve en iyisini seç
                best_frame = None
                best_face_count = 0
                frame_count = 0
                
                with st.spinner("📷 Kamera görüntüsü alınıyor..."):
                    for i in range(20):  # 20 frame dene (daha fazla)
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            frame_count += 1
                            
                            # Frame'in boş olup olmadığını kontrol et
                            if frame.size > 0:
                                # Önce MediaPipe ile dene
                                face_count = detector.count_faces(frame)
                                
                                # MediaPipe başarısız olursa OpenCV ile dene
                                if face_count == 0:
                                    face_count_opencv = detector.count_faces_opencv(frame)
                                    if face_count_opencv > 0:
                                        st.write(f"🔍 Frame {i+1}: OpenCV ile {face_count_opencv} yüz tespit edildi!")
                                        face_count = face_count_opencv
                                
                                # Sadece yüz bulunduğunda rapor ver
                                if face_count > 0:
                                    st.write(f"🎉 Frame {i+1}: ✅ {face_count} yüz tespit edildi!")
                                    
                                    if face_count == 1:  # Tek yüz varsa
                                        best_frame = frame
                                        best_face_count = face_count
                                        st.success("✅ Mükemmel! Tek yüz bulundu!")
                                        break
                                    elif face_count > best_face_count:
                                        best_frame = frame
                                        best_face_count = face_count
                                else:
                                    # Her 5 frame'de bir durum raporu
                                    if i % 5 == 0:
                                        st.write(f"🔍 Frame {i+1}: Yüz aranıyor... Kameraya bakın")
                            else:
                                st.warning(f"   Frame {i+1}: Boş görüntü")
                        else:
                            st.error(f"Frame {i+1}: ❌ Alınamadı")
                
                st.write(f"📊 Toplam {frame_count} frame kontrol edildi")
                
                if best_frame is not None:
                    st.image(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="✅ Yüz Tespit Edildi!")
                    
                    if best_face_count == 1:
                        # Önce MediaPipe ile crop dene
                        face = detector.get_face_crop(best_frame)
                        
                        # MediaPipe başarısız olursa OpenCV ile dene
                        if face is None:
                            face = detector.get_face_crop_opencv(best_frame)
                            if face is not None:
                                st.success("✅ OpenCV ile yüz kırpıldı")
                        
                        if face is not None:
                            st.success("✅ Yüz kırpıldı")
                            
                            # Çoklu poz kayıt sistemi
                            st.info("📸 Çoklu poz kayıt sistemi başlatılıyor...")
                            st.write("🔄 10 farklı poz için yüzünüzü farklı açılardan gösterin")
                            
                            embeddings = []
                            poses_needed = 10
                            current_pose = 0
                            
                            # Progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Kamera döngüsü - 10 farklı poz için
                            while current_pose < poses_needed:
                                ret, frame = cap.read()
                                if ret:
                                    # Yüz tespit et
                                    face_info = embedder.embedder.get_face_info(frame)
                                    
                                    if face_info is not None:
                                        # Embedding çıkar
                                        embedding = embedder.get_embedding(frame, skip_pca=True)  # PCA'sız
                                        
                                        if embedding is not None:
                                            embeddings.append(embedding)
                                            current_pose += 1
                                            
                                            # Progress güncelle
                                            progress = current_pose / poses_needed
                                            progress_bar.progress(progress)
                                            status_text.write(f"✅ Poz {current_pose}/{poses_needed} kaydedildi")
                                            
                                            # Kısa bekleme
                                            time.sleep(0.5)
                                        else:
                                            status_text.write("⚠️ Embedding çıkarılamadı, tekrar deneyin")
                                    else:
                                        status_text.write("🔍 Yüz aranıyor... Farklı açıdan bakın")
                                
                                # Her 10 frame'de bir güncelle
                                if current_pose % 2 == 0:
                                    time.sleep(0.1)
                            
                            # Tüm pozlar tamamlandı
                            if len(embeddings) == poses_needed:
                                st.success(f"✅ {poses_needed} farklı poz kaydedildi!")
                                
                                # Veritabanına kaydet
                                db = FaceDatabase()
                                for i, embedding in enumerate(embeddings):
                                    db.save_user_embedding(f"{user_name}_pose_{i+1}", embedding)
                                
                                st.success(f"🎉 {user_name} için {poses_needed} farklı poz ile kayıt tamamlandı!")
                                
                                # PCA modelini güncelle
                                st.info("🔄 PCA modeli güncelleniyor...")
                                try:
                                    from models.train_arcface_pca import train_arcface_pca
                                    train_arcface_pca()
                                    st.success("✅ PCA modeli güncellendi!")
                                except Exception as e:
                                    st.warning(f"⚠️ PCA güncelleme hatası: {e}")
                                    st.info("💡 Manuel güncelleme için: python models/train_arcface_pca.py")
                            else:
                                st.error(f"❌ Sadece {len(embeddings)}/{poses_needed} poz kaydedildi")
                                st.info("💡 Daha fazla farklı açı deneyin")
                        else:
                            st.error("❌ Yüz tespit edilemedi.")
                    else:
                        st.error(f"❌ {best_face_count} yüz algılandı. Lütfen sadece bir kişi kamerada olsun.")
                else:
                    st.error("❌ Hiçbir frame'de yüz tespit edilemedi.")
                    st.info("💡 Lütfen şunları deneyin:")
                    st.write("• Kameraya daha yakın durun (20-40 cm)")
                    st.write("• Daha iyi aydınlatma sağlayın")
                    st.write("• Yüzünüzün tam görünür olduğundan emin olun")
                    st.write("• Farklı açılardan deneyin")
                    st.write("• Yüzünüzü kameraya doğru çevirin")
                cap.release()

elif choice == "Yüz Doğrulama":
    st.header("Yüz Doğrulama (Face Verification)")
    user_name = st.text_input("Kullanıcı Adı")
    threshold = st.slider("Eşik Değeri (%)", 50, 100, 70)
    if st.button("Kameradan Doğrula", key="verify_button"):
        # Kamera başlat - Önce OBS Virtual Camera, sonra bilgisayar kamerası dene
        cap = cv2.VideoCapture(1)  # OBS Virtual Camera (İndeks 1)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)  # Bilgisayar kamerası (İndeks 0)
            if cap.isOpened():
                st.info("📷 OBS Virtual Camera bulunamadı, bilgisayar kamerası kullanılıyor")
            else:
                st.error("❌ Hiçbir kamera bulunamadı!")
                cap.release()
                st.stop()
        else:
            st.success("📷 OBS Virtual Camera kullanılıyor")
        detector = FaceMeshDetector()
        embedder = FaceEmbedder(pca_model_path="models/saved/pca_arcface_model.joblib")
        db = FaceDatabase()
        all_users = db.get_all_embeddings()
        # Kullanıcının tüm pozlarını bul (user_name ile başlayan)
        user_embeddings = [emb for uid, emb in all_users if uid.startswith(user_name)]
        if not user_embeddings:
            st.error(f"Kullanıcı bulunamadı: {user_name}")
            # Mevcut kullanıcıları göster
            unique_users = set()
            for uid, _ in all_users:
                base_name = uid.split('_pose_')[0] if '_pose_' in uid else uid
                unique_users.add(base_name)
            st.info("💡 Mevcut kullanıcılar: " + ", ".join(unique_users))
        else:
            verified = False
            status_text = st.empty()
            info_text = st.empty()
            
            st.info("🔍 Yüz doğrulama başlatılıyor... Kameraya bakın")
            
            # Doğrulama için maksimum deneme sayısı
            max_attempts = 50
            attempt_count = 0
            
            while attempt_count < max_attempts:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Kamera görüntüsü alınamadı.")
                    break
                
                # Kamera görüntüsünü göster
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Kamera Görüntüsü")
                
                face_count = detector.count_faces(frame)
                if face_count > 1:
                    info_text.warning("⚠️ Birden fazla yüz algılandı! Lütfen sadece bir kişi kamerada olsun.")
                    attempt_count += 1
                    time.sleep(0.5)
                    continue
                elif face_count == 0:
                    info_text.info("🔍 Yüz aranıyor... Kameraya bakın")
                    attempt_count += 1
                    time.sleep(0.3)
                else:
                    face = detector.get_face_crop(frame)
                    if face is not None:
                        embedding = embedder.get_embedding(face)
                        if embedding is not None:
                            similarities = [embedder.calculate_similarity(embedding, db_emb) for db_emb in user_embeddings]
                            max_similarity = max(similarities)
                            
                            # Benzerlik durumuna göre renk kodlaması
                            if max_similarity >= threshold:
                                status_text.success(f"✅ Benzerlik: %{max_similarity:.2f} (Eşik: %{threshold})")
                                info_text.success(f"🎉 Giriş başarılı! Hoş geldiniz, {user_name}!")
                                verified = True
                                break
                            else:
                                status_text.error(f"❌ Benzerlik: %{max_similarity:.2f} (Eşik: %{threshold})")
                                info_text.error(f"🚫 Giriş reddedildi! Benzerlik yetersiz.")
                                log_failed_attempt(user_name, max_similarity, ip="localhost")
                                break
                        else:
                            info_text.warning("⚠️ Yüz embedding'i çıkarılamadı. Pozunuzu değiştirin.")
                            attempt_count += 1
                            time.sleep(0.5)
                    else:
                        info_text.info("🔍 Yüz tespit ediliyor...")
                        attempt_count += 1
                        time.sleep(0.3)
            
            # Maksimum deneme sayısına ulaşıldıysa
            if attempt_count >= max_attempts and not verified:
                st.error("⏰ Zaman aşımı! Yüz doğrulama tamamlanamadı.")
                st.info("💡 Lütfen şunları deneyin:")
                st.write("• Kameraya daha yakın durun")
                st.write("• Daha iyi aydınlatma sağlayın")
                st.write("• Yüzünüzü kameraya doğru çevirin")
            
            cap.release()

elif choice == "Kullanıcı Sil (Admin)":
    st.header("Kullanıcı Sil (Admin)")
    admin_pass = st.text_input("Admin Parolası", type="password")
    user_name = st.text_input("Silinecek Kullanıcı Adı")
    if st.button("Kullanıcıyı Sil", key="delete_button"):
        if admin_pass != ADMIN_PASSWORD:
            st.error("Yetkisiz erişim!")
        elif not user_name:
            st.error("Kullanıcı adı girilmelidir.")
        else:
            db = FaceDatabase()
            # Kullanıcının tüm pozlarını sil
            deleted_count = 0
            all_users = db.get_all_embeddings()
            for uid, _ in all_users:
                if uid.startswith(user_name):
                    db.delete_user(uid)
                    deleted_count += 1
            
            if deleted_count > 0:
                st.success(f"{user_name} için {deleted_count} poz silindi!")
            else:
                st.warning(f"{user_name} kullanıcısı bulunamadı!")

elif choice == "Hatalı Giriş Logları":
    st.header("Hatalı Giriş Logları")
    show_log_file() 