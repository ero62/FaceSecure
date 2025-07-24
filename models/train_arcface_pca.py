# models/train_arcface_pca.py
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from joblib import dump
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.face_embedder import FaceEmbedder
from models.database import FaceDatabase

def train_arcface_pca():
    """
    ArcFace embedding'leri için PCA modeli eğitir.
    """
    print("ArcFace PCA modeli eğitiliyor...")
    
    # Veritabanından tüm embedding'leri al
    db = FaceDatabase()
    all_embeddings = db.get_all_embeddings()
    
    if not all_embeddings:
        print("Veritabanında embedding bulunamadı! Önce kullanıcı kaydı yapın.")
        return
    
    # Embedding boyutlarını kontrol et
    embeddings_list = []
    arcface_embeddings = []
    for user_id, emb in all_embeddings:
        if emb is not None:
            print(f"  {user_id}: {emb.shape if hasattr(emb, 'shape') else type(emb)}")
            embeddings_list.append(emb)
            # Sadece 512 boyutlu ArcFace embedding'lerini al
            if hasattr(emb, 'shape') and emb.shape == (512,):
                arcface_embeddings.append(emb)
        else:
            print(f"  {user_id}: None embedding!")
    
    if not arcface_embeddings:
        print("512 boyutlu ArcFace embedding bulunamadı!")
        return
    
    print(f"\n✅ {len(arcface_embeddings)} adet 512 boyutlu ArcFace embedding bulundu.")
    
    # Embedding'leri numpy array'e çevir
    try:
        embeddings_array = np.array(arcface_embeddings)
        print(f"Embedding array boyutu: {embeddings_array.shape}")
    except ValueError as e:
        print(f"Embedding array oluşturma hatası: {e}")
        return
    
    # L2 normalizasyonu uygula
    normalizer = Normalizer(norm="l2")
    normalized_embeddings = normalizer.fit_transform(embeddings_array)
    
    # Embedding sayısına göre PCA boyutunu belirle
    if len(embeddings_array) >= 128:
        n_components = 128  # 128 boyutlu PCA
    else:
        n_components = min(len(embeddings_array), 64)  # Mevcut embedding sayısına göre
        print(f"⚠️ Sadece {len(embeddings_array)} embedding var, {n_components} boyutlu PCA oluşturuluyor")
        print("💡 Daha iyi sonuç için en az 128 embedding önerilir")
    
    # PCA modelini eğit
    pca = PCA(n_components=n_components, random_state=42)
    pca_embeddings = pca.fit_transform(normalized_embeddings)
    
    # Açıklanan varyans oranını hesapla
    explained_variance_ratio = pca.explained_variance_ratio_.sum()
    print(f"PCA boyutu: {n_components}")
    print(f"Açıklanan varyans oranı: {explained_variance_ratio:.4f}")
    
    # PCA modelini kaydet
    model_path = os.path.join(os.path.dirname(__file__), "saved", "pca_arcface_model.joblib")
    dump(pca, model_path)
    print(f"PCA modeli kaydedildi: {model_path}")
    
    # Test embedding'i oluştur ve PCA uygula
    test_embedding = normalized_embeddings[0]
    pca_test = pca.transform([test_embedding])
    print(f"Test embedding boyutu: {test_embedding.shape} -> {pca_test.shape}")
    
    return pca

if __name__ == "__main__":
    train_arcface_pca() 