# face_mesh_detector.py
import cv2
import mediapipe as mp
import numpy as np


class FaceMeshDetector:
    """
    OpenCV ve Mediapipe ile yüz algılama ve yüz mesh işlemleri için yardımcı sınıf.
    """
    def __init__(
        self,
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.1,  # Çok düşük eşik (0.3'ten 0.1'e)
        min_tracking_confidence=0.1,   # Çok düşük eşik (0.3'ten 0.1'e)
    ):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        # Alternatif yüz tespiti için OpenCV Cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_and_draw(self, image):
        """
        Görüntüdeki yüzleri tespit eder ve mesh çizgilerini çizer.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = self.face_mesh.process(rgb_image)
        rgb_image.flags.writeable = True
        image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec,
                )
        return image

    def get_face_crop(self, image):
        """
        Görüntüdeki ilk yüzü crop (kırpılmış) olarak döndürür.
        Önce MediaPipe, sonra OpenCV dener.
        """
        # MediaPipe ile dene
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = image.shape
                x_coords = [int(landmark.x * w) for landmark in face_landmarks.landmark]
                y_coords = [int(landmark.y * h) for landmark in face_landmarks.landmark]
                x_min, x_max = max(min(x_coords) - 10, 0), min(max(x_coords) + 10, w)
                y_min, y_max = max(min(y_coords) - 10, 0), min(max(y_coords) + 10, h)
                return image[y_min:y_max, x_min:x_max]
        
        # MediaPipe başarısız olursa OpenCV ile dene
        print("Debug: MediaPipe başarısız, OpenCV deneniyor...")
        return self.get_face_crop_opencv(image)

    def count_faces(self, image):
        """
        Görüntüdeki yüz sayısını döndürür. Önce MediaPipe, sonra OpenCV dener.
        """
        # MediaPipe ile dene
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        if results.multi_face_landmarks:
            print(f"Debug: MediaPipe ile {len(results.multi_face_landmarks)} yüz tespit edildi")
            return len(results.multi_face_landmarks)
        
        # MediaPipe başarısız olursa OpenCV ile dene
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # Daha hassas
            minNeighbors=2,    # Daha düşük eşik
            minSize=(20, 20)   # Daha küçük yüzler
        )
        print(f"Debug: OpenCV ile {len(faces)} yüz tespit edildi")
        return len(faces)

    def count_faces_opencv(self, image):
        """
        Sadece OpenCV ile yüz sayısını döndürür.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=3, 
            minSize=(30, 30)
        )
        return len(faces)

    def get_face_crop_opencv(self, image):
        """
        OpenCV ile yüz crop'u döndürür.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=3, 
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            x, y, w, h = faces[0]  # İlk yüzü al
            # Biraz daha geniş crop
            x_min = max(0, x - 20)
            y_min = max(0, y - 20)
            x_max = min(image.shape[1], x + w + 20)
            y_max = min(image.shape[0], y + h + 20)
            return image[y_min:y_max, x_min:x_max]
        
        return None
