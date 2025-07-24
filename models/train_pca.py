# train_pca.py
from face_embedder import FaceEmbedder
from sklearn.decomposition import PCA
from joblib import dump
import cv2
import numpy as np

TARGET_COUNT = 128  # İstenen embedding sayısı

embedder = FaceEmbedder()  # PCA olmadan embed alıyoruz
cap = cv2.VideoCapture(0)
embeddings = []

print(f"📸 Farklı pozlardan {TARGET_COUNT} yüz örneği toplanacak...")

while len(embeddings) < TARGET_COUNT:
    ret, frame = cap.read()
    if not ret:
        print("❌ Kamera görüntüsü alınamadı.")
        break

    face_crop = cv2.resize(frame, (160, 160))
    embedding = embedder.get_embedding(face_crop, skip_pca=True)
    embeddings.append(embedding)

    print(f"✅ {len(embeddings)}/{TARGET_COUNT} embedding toplandı.")
    cv2.imshow("PCA için Kayıt", frame)

    # Bu sadece ESC/Q kaçış opsiyonu. Otomatik bitmeyi etkilemez.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("👋 Kullanıcı çıkışı algılandı.")
        break

cap.release()
cv2.destroyAllWindows()

# PCA Modeli Eğitimi ve Kaydı
if len(embeddings) >= 10:  # Minimum garanti
    embeddings = np.array(embeddings)
    print("🧠 PCA modeli eğitiliyor...")
    pca = PCA(n_components=128)
    pca.fit(embeddings)
    dump(pca, "pca_model.joblib")
    print("✅ PCA modeli kaydedildi: pca_model.joblib")
else:
    print("⚠️ Yeterli veri toplanamadı.")
