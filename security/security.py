from datetime import datetime, timedelta, timezone
from jose import jwt
from passlib.context import CryptContext

SECRET_KEY = "secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


class Autentify:

    def __init__(self):
        self.pwd_context = CryptContext(
            schemes=["argon2"],
            deprecated="auto"
        )

    def hash_password(self, password: str):
        return self.pwd_context.hash(password)

    def verify_password(self, password: str, hashed: str):
        return self.pwd_context.verify(password, hashed)

    def create_access_token(self, data: dict):
        to_encode = data.copy()

        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

        to_encode.update({"exp": expire})

        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    def decode(self, token: str):
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])