from database import FaceDatabase

def admin_delete_user():
    user_id = input("Silinecek kullanıcı adı: ").strip()
    if not user_id:
        print("Kullanıcı adı boş olamaz.")
        return
    db = FaceDatabase()
    db.delete_user(user_id)

if __name__ == "__main__":
    admin_delete_user() 