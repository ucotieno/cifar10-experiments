from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .inference import CIFAR10Inference


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"


app = FastAPI(
    title="CIFAR-10 Inference API",
    version="1.0.0",
)


predictor = CIFAR10Inference()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "cifar10_net",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        )

    result = predictor.predict(image)

    return {
        "filename": file.filename,
        **result,
    }


app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)
