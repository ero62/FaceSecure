from flask import Flask, request, jsonify
import jwt
import datetime
import numpy as np
from models.face_embedder import FaceEmbedder
from models.database import FaceDatabase
import os
from models.face_mesh_detector import FaceMeshDetector
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

THRESHOLD = 70.0  # Benzerlik için eşik değer (%)

app = Flask(__name__)

def create_jwt(username, is_admin=False):
    """
    Kullanıcı adı ve adminlik bilgisiyle JWT token üretir.
    """
    payload = {
        "username": username,
        "is_admin": is_admin,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def log_failed_attempt(user_id, similarity, ip):
    """
    Hatalı girişleri timestamp, user_id, IP ve benzerlik ile loglar.
    """
    log_dir = os.path.join(os.path.dirname(__file__), '../logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'failed_logins.log')
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f'{timestamp}, {user_id}, {ip}, %{similarity:.2f}\n')

@app.route("/login", methods=["POST"])
def login():
    """
    Admin girişi için endpoint. Başarılıysa JWT token döner.
    """
    data = request.json
    username = data.get("username")
    password = data.get("password")
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        token = create_jwt(username, is_admin=True)
        return jsonify({"token": token})
    return jsonify({"error": "Geçersiz kimlik bilgisi"}), 401

@app.route("/enroll", methods=["POST"])
def enroll():
    """
    Yeni kullanıcı embedding'lerini kaydeder (sadece admin token ile).
    """
    auth = request.headers.get("Authorization", "").replace("Bearer ", "")
    try:
        payload = jwt.decode(auth, SECRET_KEY, algorithms=["HS256"])
        if not payload.get("is_admin"):
            return jsonify({"error": "Yetkisiz"}), 403
    except Exception:
        return jsonify({"error": "Token geçersiz"}), 401
    data = request.json
    user_id = data.get("user_id")
    embeddings = data.get("embeddings")
    if not user_id or not embeddings:
        return jsonify({"error": "user_id ve embeddings gerekli"}), 400
    db = FaceDatabase()
    for emb in embeddings:
        db.save_user_embedding(user_id, np.array(emb))
    return jsonify({"success": True})

@app.route("/verify", methods=["POST"])
def verify():
    """
    Kullanıcıyı embedding ile doğrular.
    """
    data = request.json
    user_id = data.get("user_id")
    embedding = data.get("embedding")
    if not user_id or not embedding:
        return jsonify({"error": "user_id ve embedding gerekli"}), 400
    db = FaceDatabase()
    all_users = db.get_all_embeddings()
    user_embeddings = [emb for uid, emb in all_users if uid == user_id]
    if not user_embeddings:
        return jsonify({"error": "Kullanıcı bulunamadı"}), 404
    embedder = FaceEmbedder(pca_model_path="models/saved/pca_arcface_model.joblib")
    similarities = [embedder.calculate_similarity(np.array(embedding), db_emb) for db_emb in user_embeddings]
    max_similarity = max(similarities)
    if max_similarity >= THRESHOLD:
        return jsonify({"success": True, "similarity": max_similarity})
    else:
        log_failed_attempt(user_id, max_similarity, request.remote_addr)
        return jsonify({"success": False, "similarity": max_similarity, "error": "Eşik altında"}), 403

@app.route("/detect_faces", methods=["POST"])
def detect_faces():
    """
    Base64 resimdeki yüz sayısını döner.
    """
    import base64
    import cv2
    data = request.json
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"error": "image gerekli"}), 400
    img_bytes = base64.b64decode(image_b64)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    detector = FaceMeshDetector()
    face_count = detector.count_faces(img)
    return jsonify({"face_count": face_count})

@app.route("/delete_user", methods=["POST"])
def delete_user():
    """
    Kullanıcıyı siler (sadece admin token ile).
    """
    auth = request.headers.get("Authorization", "").replace("Bearer ", "")
    try:
        payload = jwt.decode(auth, SECRET_KEY, algorithms=["HS256"])
        if not payload.get("is_admin"):
            return jsonify({"error": "Yetkisiz"}), 403
    except Exception:
        return jsonify({"error": "Token geçersiz"}), 401
    data = request.json
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id gerekli"}), 400
    db = FaceDatabase()
    db.delete_user(user_id)
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 