from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.dependencies import get_db
from sqlalchemy.orm import Session
from app.models.user import User



router = APIRouter()        

@router.get("/users")
def get_users(db: Session = Depends(get_db)):
    def add_user(name: str, email: str, db: Session = Depends(get_db)):
    return create_user(db, name, email)





def create_user(db: Session, name: str, email: str):
    user = User(name=name, email=email)

    db.add(user)
    db.commit()
    db.refresh(user)

    return user