# models/face_embedder.py
import cv2
import numpy as np
from sklearn.preprocessing import Normalizer
from joblib import load
from models.arcface_model import ArcFace

class FaceEmbedder:
    """
    Yüz embedding çıkarımı ve karşılaştırması için yardımcı sınıf.
    ArcFace modelini kullanır.
    """
    def __init__(self, pca_model_path=None):
        self.embedder = ArcFace()
        self.normalizer = Normalizer(norm="l2")
        self.pca = load(pca_model_path) if pca_model_path else None

    def preprocess_face(self, face_img):
        """
        Yüz görüntüsünü ArcFace için uygun hale getirir.
        ArcFace kendi preprocessing'ini yapar, bu yüzden sadece boyut kontrolü yapıyoruz.
        """
        # ArcFace için minimum boyut kontrolü
        min_size = 112  # ArcFace için önerilen minimum boyut
        h, w = face_img.shape[:2]
        
        if h < min_size or w < min_size:
            # Küçük yüzleri büyüt
            scale = max(min_size / h, min_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            face_img = cv2.resize(face_img, (new_w, new_h))
        
        return face_img

    def get_embedding(self, face_img, skip_pca=False):
        """
        Yüz embedding'ini ArcFace ile çıkarır, opsiyonel olarak PCA uygular.
        """
        try:
            print(f"Debug: get_embedding çağrıldı, face_img shape: {face_img.shape}")
            
            # 1. Doğrudan ArcFace embedding dene (en hızlı)
            try:
                embedding = self.embedder.get_embedding(face_img)
                if embedding is not None:
                    print(f"Debug: Doğrudan embedding başarılı, shape: {embedding.shape}")
                    # L2 normalizasyonu uygula
                    norm_embed = self.normalizer.transform([embedding])[0]
                    
                    if self.pca and not skip_pca:
                        if hasattr(self.pca, 'n_components_') and self.pca.n_components_ > 0:
                            norm_embed = self.pca.transform([norm_embed])[0]
                            print(f"Debug: PCA uygulandı, boyut: {self.pca.n_components_}")
                        else:
                            print("Debug: PCA boyutu 0, PCA atlandı")
                    
                    print(f"Debug: Final embedding shape: {norm_embed.shape}")
                    return norm_embed
            except Exception as e:
                print(f"Debug: Doğrudan embedding hatası: {e}")
            
            # 2. ArcFace face_info ile dene
            try:
                print("Debug: face_info deneniyor")
                face_info = self.embedder.get_face_info(face_img)
                if face_info is not None:
                    embedding = face_info.embedding
                    print(f"Debug: face_info ile embedding başarılı, shape: {embedding.shape}")
                    # L2 normalizasyonu uygula
                    norm_embed = self.normalizer.transform([embedding])[0]
                    
                    if self.pca and not skip_pca:
                        if hasattr(self.pca, 'n_components_') and self.pca.n_components_ > 0:
                            norm_embed = self.pca.transform([norm_embed])[0]
                            print(f"Debug: PCA uygulandı, boyut: {self.pca.n_components_}")
                        else:
                            print("Debug: PCA boyutu 0, PCA atlandı")
                    
                    print(f"Debug: Final embedding shape: {norm_embed.shape}")
                    return norm_embed
            except Exception as e:
                print(f"Debug: face_info hatası: {e}")
            
            # 3. Preprocessing ile dene
            try:
                print("Debug: preprocessing deneniyor")
                preprocessed = self.preprocess_face(face_img)
                embedding = self.embedder.get_embedding(preprocessed)
                if embedding is not None:
                    print(f"Debug: Preprocessing ile embedding başarılı, shape: {embedding.shape}")
                    # L2 normalizasyonu uygula
                    norm_embed = self.normalizer.transform([embedding])[0]
                    
                    if self.pca and not skip_pca:
                        if hasattr(self.pca, 'n_components_') and self.pca.n_components_ > 0:
                            norm_embed = self.pca.transform([norm_embed])[0]
                            print(f"Debug: PCA uygulandı, boyut: {self.pca.n_components_}")
                        else:
                            print("Debug: PCA boyutu 0, PCA atlandı")
                    
                    print(f"Debug: Final embedding shape: {norm_embed.shape}")
                    return norm_embed
            except Exception as e:
                print(f"Debug: preprocessing hatası: {e}")
            
            # 4. OpenCV fallback
            try:
                print("Debug: OpenCV fallback deneniyor")
                from models.face_mesh_detector import FaceMeshDetector
                detector = FaceMeshDetector()
                face_crop = detector.get_face_crop_opencv(face_img)
                if face_crop is not None:
                    print(f"Debug: OpenCV crop başarılı, shape: {face_crop.shape}")
                    embedding = self.embedder.get_embedding(face_crop)
                    if embedding is not None:
                        print(f"Debug: OpenCV crop ile embedding başarılı, shape: {embedding.shape}")
                        # L2 normalizasyonu uygula
                        norm_embed = self.normalizer.transform([embedding])[0]
                        
                        if self.pca and not skip_pca:
                            if hasattr(self.pca, 'n_components_') and self.pca.n_components_ > 0:
                                norm_embed = self.pca.transform([norm_embed])[0]
                                print(f"Debug: PCA uygulandı, boyut: {self.pca.n_components_}")
                            else:
                                print("Debug: PCA boyutu 0, PCA atlandı")
                        
                        print(f"Debug: Final embedding shape: {norm_embed.shape}")
                        return norm_embed
            except Exception as e:
                print(f"Debug: OpenCV fallback hatası: {e}")
            
            print("Debug: Tüm yöntemler başarısız")
            return None
            
        except Exception as e:
            print(f"Debug: get_embedding genel hatası: {e}")
            return None

    def calculate_similarity(self, emb1, emb2):
        """
        İki embedding arasındaki benzerliği yüzde olarak döndürür (cosine similarity).
        ArcFace için optimize edilmiş.
        """
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        
        # Embedding boyut kontrolü
        if emb1.size == 0 or emb2.size == 0:
            print("Debug: Boş embedding tespit edildi")
            return 0.0
        
        # Boyut uyumsuzluğu kontrolü - PCA uygulanmış embedding ile karşılaştırma
        if emb1.shape != emb2.shape:
            print(f"Debug: Embedding boyutları uyumsuz: {emb1.shape} vs {emb2.shape}")
            
            # Eğer emb1 PCA uygulanmış (53 boyutlu) ve emb2 orijinal (512 boyutlu) ise
            if emb1.shape[0] == 53 and emb2.shape[0] == 512:
                print("Debug: PCA uygulanmış embedding ile orijinal embedding karşılaştırılıyor")
                # emb2'ye de PCA uygula
                if self.pca:
                    emb2_pca = self.pca.transform([emb2])[0]
                    emb2 = emb2_pca
                    print(f"Debug: emb2'ye PCA uygulandı, yeni boyut: {emb2.shape}")
                else:
                    print("Debug: PCA modeli bulunamadı, karşılaştırma yapılamıyor")
                    return 0.0
            else:
                print("Debug: Farklı boyut türleri, karşılaştırma yapılamıyor")
                return 0.0
        
        # Cosine similarity hesapla
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # ArcFace için benzerlik skorunu yüzdeye çevir
        # ArcFace genellikle 0.5-1.0 arasında skorlar verir
        similarity_percent = float(cos_sim) * 100
        
        return similarity_percent

    def get_face_quality(self, face_img):
        """
        Yüz kalitesini değerlendirir (ArcFace'in kendi kalite skorunu kullanır).
        """
        face_info = self.embedder.get_face_info(face_img)
        if face_info is None:
            return 0.0
        
        # ArcFace'in kendi kalite skorunu kullan
        return getattr(face_info, 'det_score', 0.0)
