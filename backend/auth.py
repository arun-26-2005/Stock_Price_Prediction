import os
from datetime import datetime, timedelta
from typing import Optional
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import hashlib
import secrets
from sqlalchemy.orm import Session
from backend.database import get_db, User

# JWT Configuration
SECRET_KEY = os.environ.get("JWT_SECRET", "SUPER_SECRET_STOCK_PREDICTION_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# Password Hashing
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify standard PBKDF2-HMAC-SHA256 hash."""
    try:
        salt, hash_hex = hashed_password.split("$")
        pw_bytes = plain_password.encode('utf-8')
        salt_bytes = salt.encode('utf-8')
        dk = hashlib.pbkdf2_hmac('sha256', pw_bytes, salt_bytes, 100000)
        return dk.hex() == hash_hex
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with a random salt."""
    salt = secrets.token_hex(16)
    pw_bytes = password.encode('utf-8')
    salt_bytes = salt.encode('utf-8')
    dk = hashlib.pbkdf2_hmac('sha256', pw_bytes, salt_bytes, 100000)
    hash_hex = dk.hex()
    return f"{salt}${hash_hex}"

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception
        
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
        
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception
        
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

async def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: Admin privilege required",
        )
    return current_user
