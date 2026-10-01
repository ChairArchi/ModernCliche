"""Synthetic verification fixtures, explicitly not a collected subject dataset."""
import numpy as np
from PIL import Image, ImageDraw
from .data import decode_image
from .export import png_bytes


def demo_samples():
    result = []
    rng = np.random.default_rng(17)
    for i in range(16):
        im = Image.new("RGBA", (180, 180), (255, 255, 255, 0))
        draw = ImageDraw.Draw(im)
        radius = 22 + (i % 8) * 2
        if i < 8:
            draw.ellipse((90 - radius, 25, 90 + radius, 155), fill=(45, 45, 45, 255))
        else:
            draw.rectangle((75, 25, 105, 155), fill=(45, 45, 45, 255))
            draw.rectangle((25, 70 - radius // 2, 155, 70 + radius // 2), fill=(45, 45, 45, 255))
        result.append(decode_image(png_bytes(im), dict(title=f"검증용 합성 실루엣 {i + 1}",
                                                     provider="synthetic verification", dataset_label="ellipse" if i < 8 else "cross")))
    return result
