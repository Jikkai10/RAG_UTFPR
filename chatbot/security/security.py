from datetime import datetime, timedelta, timezone
from jose import jwt
from passlib.context import CryptContext
from dotenv import load_dotenv
import logging
import os

load_dotenv()

logger = logging.getLogger("uvicorn.error")

SECRET_KEY = os.getenv("JWT_SECRET")
if not SECRET_KEY:
    SECRET_KEY = "dev-secret-change-me"
    logger.warning(
        "JWT_SECRET nao definido, usando chave de desenvolvimento. "
        "Defina JWT_SECRET no .env antes de expor a API."
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


class Authenticator:

    def __init__(self):
        self.pwdContext = CryptContext(
            schemes=["argon2"],
            deprecated="auto"
        )

    def hashPassword(self, password: str):
        return self.pwdContext.hash(password)

    def verifyPassword(self, password: str, hashed: str):
        return self.pwdContext.verify(password, hashed)

    def createAccessToken(self, data: dict):
        toEncode = data.copy()

        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

        toEncode.update({"exp": expire})

        return jwt.encode(toEncode, SECRET_KEY, algorithm=ALGORITHM)

    def decode(self, token: str):
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
