import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import Neo4jConnection, URI, USER, PASSWORD
from security.security import Authenticator

queryCreateUser = """MERGE (u:User {id: $id})
        SET u.email = $email,
        u.password = $password,
        u.role = $role
"""

DEFAULT_USERS = [
    {"id": "1", "email": "admin@email.com", "role": "admin"},
    {"id": "2", "email": "user@email.com", "role": "user"},
]

DEFAULT_PASSWORD = "senha"


def createDefaultUsers(db=None, password=DEFAULT_PASSWORD):
    db = db or Neo4jConnection(URI, USER, PASSWORD)
    auth = Authenticator()
    hashed = auth.hashPassword(password)

    for user in DEFAULT_USERS:
        db.executeQuery(queryCreateUser, parameters={
            "id": user["id"],
            "email": user["email"],
            "password": hashed,
            "role": user["role"]
        })
        print(f"User {user['email']} created successfully.")


if __name__ == "__main__":
    createDefaultUsers()
