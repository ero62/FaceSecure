import requests
import base64
import numpy as np
import cv2

API_URL = "http://localhost:5000"

# 1. Admin login (token al)
login_resp = requests.post(f"{API_URL}/login", json={"username": "aeren", "password": "eren1234"})
assert login_resp.status_code == 200, login_resp.text
token = login_resp.json()["token"]
headers = {"Authorization": f"Bearer {token}"}
print("[+] Login başarılı, token alındı.")

# 2. Enroll (örnek kullanıcı ve 10 random embedding)
user_id = "test_user"
embeddings = [list(np.random.rand(128)) for _ in range(10)]
enroll_resp = requests.post(f"{API_URL}/enroll", json={"user_id": user_id, "embeddings": embeddings}, headers=headers)
print("[+] Enroll yanıtı:", enroll_resp.status_code, enroll_resp.text)

# 3. Verify (doğrulama)
verify_resp = requests.post(f"{API_URL}/verify", json={"user_id": user_id, "embedding": embeddings[0]})
print("[+] Verify yanıtı:", verify_resp.status_code, verify_resp.text)

# 4. Detect faces (örnek: tek yüzlü bir resim base64)
# Burada örnek olarak siyah bir resim kullanıyoruz, gerçek testte bir yüz resmi base64 ile gönderilmeli.
img = np.zeros((160, 160, 3), dtype=np.uint8)
_, img_encoded = cv2.imencode('.jpg', img)
img_b64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
detect_resp = requests.post(f"{API_URL}/detect_faces", json={"image": img_b64})
print("[+] Detect faces yanıtı:", detect_resp.status_code, detect_resp.text) 