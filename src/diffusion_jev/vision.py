"""Local image inputs and prepared galleries. Dataset labels never reach the model."""

import base64
import io
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageOps

DATA_ROOT = Path(os.getenv("DIFFUSION_JEV_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
DATASETS = {"flowers", "quickdraw"}


@lru_cache(maxsize=2)
def dataset_manifest(dataset_id="flowers"):
    if dataset_id not in DATASETS:
        raise ValueError("Unknown image dataset")
    return json.loads((DATA_ROOT / dataset_id / "manifest.json").read_text())


flower_manifest = dataset_manifest  # Keep existing flower benchmark imports working.


def dataset_image_path(dataset_id, image_id):
    if dataset_id not in DATASETS:
        raise ValueError("Unknown image dataset")
    if not re.fullmatch(r"[0-9a-f]{64}", image_id):
        raise ValueError("Invalid gallery image ID")
    path = DATA_ROOT / dataset_id / "images" / f"{image_id}.jpg"
    if not path.is_file():
        raise ValueError("Gallery image not found")
    return path


def flower_image_path(image_id):
    return dataset_image_path("flowers", image_id)


def resolve_images(images):
    results = []
    for source in images:
        if re.fullmatch(r"[0-9a-f]{64}", source):
            data = flower_image_path(source).read_bytes()
        elif re.fullmatch(r"[a-z]+:[0-9a-f]{64}", source):
            data = dataset_image_path(*source.split(":", 1)).read_bytes()
        else:
            if not re.match(r"^data:image/(jpeg|png|webp);base64,", source):
                raise ValueError("Use a gallery image ID or a JPEG, PNG or WebP data URL")
            try:
                raw = base64.b64decode(source.split(",", 1)[1], validate=True)
                if len(raw) > 6_000_000:
                    raise ValueError("Each image must be at most 6 MB")
                with Image.open(io.BytesIO(raw)) as original:
                    if original.width * original.height > 20_000_000:
                        raise ValueError("Image exceeds 20 megapixels")
                    image = ImageOps.exif_transpose(original).convert("RGB")
                    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
                    output = io.BytesIO()
                    image.save(output, format="JPEG", quality=92)
                    data = output.getvalue()
            except (OSError, Image.DecompressionBombError) as error:
                raise ValueError("Cannot decode image") from error
        results.append("data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"))
    return results
