import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import OUTPUT_IMAGES

router = APIRouter()


@router.get("/images/{filename}")
def get_image(filename: str):
    # os.path.basename strips any directory components the caller might
    # try to smuggle in (e.g. "../../etc/passwd") before it ever touches
    # the filesystem — the only path this ever resolves is directly inside
    # OUTPUT_IMAGES.
    safe_filename = os.path.basename(filename)
    path = os.path.join(OUTPUT_IMAGES, safe_filename)

    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(path, media_type="image/png")
