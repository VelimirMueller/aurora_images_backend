import uuid
import base64
import shutil

from pathlib import Path
from typing import Union
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def create_file(image: UploadFile = File(...)):
    save_path = Path(f"uploads/{image.filename}")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    return {"message": image.filename}