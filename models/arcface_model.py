# models/arcface_model.py
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import threading

class ArcFace:
    """
    ArcFace modelini thread-safe şekilde yükler ve erişim sağlar.
    """
    _app = None
    _lock = threading.Lock()

    def __init__(self, model_name='buffalo_l'):
        self.model_name = model_name
        # Modeli thread-safe şekilde yükle
        with ArcFace._lock:
            if ArcFace._app is None:
                ArcFace._app = FaceAnalysis(name=model_name)
                # Daha agresif yüz tespit için parametreler
                ArcFace._app.prepare(ctx_id=0, det_size=(320, 320))  # Daha küçük det_size

    @property
    def app(self):
        """
        Yüklü ArcFace uygulamasını döndürür.
        """
        return ArcFace._app

    def get_embedding(self, face_img):
        """
        Yüz görüntüsünden ArcFace embedding'ini çıkarır.
        """
        try:
            # BGR'den RGB'ye çevir
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Yüz analizi yap
            faces = self.app.get(face_rgb)
            
            if len(faces) == 0:
                return None
            
            # İlk yüzün embedding'ini al
            embedding = faces[0].embedding
            
            return embedding
        except Exception as e:
            return None

    def get_face_info(self, face_img):
        """
        Yüz görüntüsünden detaylı bilgi çıkarır (embedding, landmarks, vb.).
        """
        try:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            faces = self.app.get(face_rgb)
            
            if len(faces) == 0:
                return None
            
            return faces[0]
        except Exception as e:
            return None 