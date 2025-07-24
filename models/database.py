from pymongo import MongoClient
import datetime
import numpy as np
import hashlib


class FaceDatabase:
    """
    MongoDB üzerinde yüz embedding verilerini saklamak ve yönetmek için yardımcı sınıf.
    """
    def __init__(
        self,
        uri="mongodb://localhost:27017",
        db_name="face_secure",
        collection_name="embeddings",
    ):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def save_user_embedding(self, user_id, embedding):
        """
        Kullanıcıya ait embedding'i hash ile birlikte veritabanına kaydeder.
        """
        emb_bytes = np.array(embedding).astype(np.float32).tobytes()
        emb_hash = hashlib.sha256(emb_bytes).hexdigest()
        doc = {
            "user_id": user_id,
            "embedding": embedding.tolist(),
            "embedding_hash": emb_hash,
            "created_at": datetime.datetime.now(),
        }
        self.collection.insert_one(doc)
        print(f"✅ '{user_id}' kullanıcısı embedding ve hash ile kaydedildi.")

    def get_all_embeddings(self, with_hash=False):
        """
        Tüm kullanıcı embedding'lerini (isteğe bağlı hash ile) döndürür.
        """
        users = self.collection.find()
        if with_hash:
            return [(user["user_id"], np.array(user["embedding"]), user["embedding_hash"]) for user in users]
        else:
            return [(user["user_id"], np.array(user["embedding"])) for user in users]

    def delete_user(self, user_id):
        """
        Belirtilen kullanıcıya ait tüm embedding kayıtlarını siler.
        """
        result = self.collection.delete_many({"user_id": user_id})
        print(f"🗑️ Silinen belge sayısı: {result.deleted_count}")
