import requests
import json

API_URL = "http://localhost:5000"

print("🔍 API Test Başlatılıyor...")

# 1. Admin login test
print("\n1️⃣ Login Test:")
login_data = {"username": "aeren", "password": "eren1234"}
try:
    login_resp = requests.post(f"{API_URL}/login", json=login_data, timeout=5)
    print(f"   Status: {login_resp.status_code}")
    print(f"   Response: {login_resp.text}")
    
    if login_resp.status_code == 200:
        token = login_resp.json()["token"]
        print("   ✅ Login başarılı!")
        headers = {"Authorization": f"Bearer {token}"}
    else:
        print("   ❌ Login başarısız!")
        exit(1)
        
except Exception as e:
    print(f"   ❌ Login hatası: {e}")
    exit(1)

# 2. Enroll test
print("\n2️⃣ Enroll Test:")
enroll_data = {
    "user_id": "test_user_api",
    "embeddings": [[0.1, 0.2, 0.3] * 170]  # 512 boyutlu embedding
}
try:
    enroll_resp = requests.post(f"{API_URL}/enroll", json=enroll_data, headers=headers, timeout=5)
    print(f"   Status: {enroll_resp.status_code}")
    print(f"   Response: {enroll_resp.text}")
    
    if enroll_resp.status_code == 200:
        print("   ✅ Enroll başarılı!")
    else:
        print("   ❌ Enroll başarısız!")
        
except Exception as e:
    print(f"   ❌ Enroll hatası: {e}")

# 3. Verify test
print("\n3️⃣ Verify Test:")
verify_data = {
    "user_id": "test_user_api",
    "embedding": [0.1, 0.2, 0.3] * 170
}
try:
    verify_resp = requests.post(f"{API_URL}/verify", json=verify_data, timeout=5)
    print(f"   Status: {verify_resp.status_code}")
    print(f"   Response: {verify_resp.text}")
    
    if verify_resp.status_code == 200:
        print("   ✅ Verify başarılı!")
    else:
        print("   ❌ Verify başarısız!")
        
except Exception as e:
    print(f"   ❌ Verify hatası: {e}")

print("\n🎉 API Test Tamamlandı!") 