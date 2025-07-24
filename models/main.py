# main.py
import cv2
from face_mesh_detector import FaceMeshDetector
from face_embedder import FaceEmbedder

MAX_EMBEDDINGS = 128  # Belirli embedding sayısı


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera açılamadı!")
        return

    print("✅ Kamera açıldı")
    detector = FaceMeshDetector()
    embedder = FaceEmbedder(pca_model_path="pca_model.joblib")
    embeddings_collected = 0

    while embeddings_collected < MAX_EMBEDDINGS:
        ret, frame = cap.read()
        if not ret:
            print("❌ Kare alınamadı.")
            break

        face_crop = detector.get_face_crop(frame)
        if face_crop is not None:
            embedding = embedder.get_embedding(face_crop)
            embeddings_collected += 1
            print(
                f"✅ {embeddings_collected}/{MAX_EMBEDDINGS} - Embedding shape: {embedding.shape}"
            )
        else:
            print("⚠️ Yüz algılanamadı.")

        frame = detector.detect_and_draw(frame)
        cv2.imshow("Yüz Takibi ve PCA Embedding", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("👋 Kullanıcı çıkışı algıladı.")
            break

    print("🛑 Hedef embedding sayısına ulaşıldı. Sistem durduruluyor.")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
