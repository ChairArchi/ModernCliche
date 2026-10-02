from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
import re

import numpy as np
import requests
from PIL import Image, ImageOps
from scipy.fft import dctn

USER_AGENT = "ModernCliche/2.0 (academic image-shape study; https://github.com/ChairArchi/ModernCliche)"
ALIASES = {"가고일": "gargoyle", "개": "dog", "고양이": "cat", "새": "bird", "문": "door"}


@dataclass
class Sample:
    id: str
    image: Image.Image
    metadata: dict = field(default_factory=dict)


def decode_image(raw: bytes, metadata: dict | None = None) -> Sample:
    if len(raw) > 12_000_000:
        raise ValueError("이미지는 12MB 이하로 업로드하십시오.")
    with Image.open(BytesIO(raw)) as im:
        if im.width * im.height > 30_000_000:
            raise ValueError("이미지 해상도가 제한을 초과했습니다. 3천만 픽셀 이하로 줄이십시오.")
        image = ImageOps.exif_transpose(im).convert("RGBA")
        image.thumbnail((768, 768))
    return Sample(sha256(raw).hexdigest()[:16], image, metadata or {})


def clean_text(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value or "").strip()


def search_images(query: str, limit: int = 36, provider: str = "commons") -> tuple[list[dict], str]:
    query = ALIASES.get(query.strip(), query.strip())
    if not query:
        raise ValueError("검색할 대상을 입력하십시오.")
    limit = max(1, min(int(limit), 80))
    items: list[dict] = []
    if provider == "commons":
        continuation = {}
        while len(items) < limit:
            params = dict(action="query", generator="search", gsrsearch=query,
                          gsrnamespace=6, gsrlimit=min(limit - len(items), 50),
                          prop="imageinfo", iiprop="url|extmetadata|mime", iiurlwidth=512,
                          format="json", **continuation)
            response = requests.get("https://commons.wikimedia.org/w/api.php", params=params,
                                    headers={"User-Agent": USER_AGENT}, timeout=(8, 25))
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise ValueError(payload["error"].get("info", "Commons 검색 오류"))
            pages = sorted(payload.get("query", {}).get("pages", {}).values(),
                           key=lambda page: page.get("index", 0))
            for page in pages:
                info = (page.get("imageinfo") or [{}])[0]
                if not info.get("mime", "").startswith("image/"):
                    continue
                meta = info.get("extmetadata", {})
                get = lambda key: clean_text(meta.get(key, {}).get("value", ""))
                url = info.get("thumburl") or info.get("url")
                if url:
                    items.append(dict(title=page.get("title", ""), download_url=url,
                                      source_url=info.get("descriptionurl", ""),
                                      creator=get("Artist"), license=get("LicenseShortName"),
                                      license_url=get("LicenseUrl"), provider="Wikimedia Commons", query=query))
            continuation = payload.get("continue", {})
            if not continuation or not pages:
                break
    elif provider == "openverse":
        for page in range(1, 5):
            response = requests.get("https://api.openverse.org/v1/images/",
                                    params={"q": query, "page_size": min(limit, 20), "page": page},
                                    headers={"User-Agent": USER_AGENT}, timeout=(8, 25))
            response.raise_for_status()
            results = response.json().get("results", [])
            for item in results:
                items.append(dict(title=item.get("title", ""),
                                  download_url=item.get("thumbnail") or item.get("url"),
                                  source_url=item.get("foreign_landing_url", ""),
                                  creator=item.get("creator", ""), license=item.get("license", ""),
                                  license_url=item.get("license_url", ""),
                                  provider="Openverse", query=query))
            if len(items) >= limit or not results:
                break
    else:
        raise ValueError("지원하지 않는 검색 제공자입니다.")
    # Search engines provide retrieval relevance, not verified category membership.
    return items[:limit], query


def download_images(items: list[dict], progress=None) -> tuple[list[Sample], list[dict]]:
    def fetch(item):
        with requests.get(item["download_url"], headers={"User-Agent": USER_AGENT},
                          timeout=(8, 20), stream=True) as response:
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > 12_000_000:
                    raise ValueError("download larger than 12MB")
                chunks.append(chunk)
        return decode_image(b"".join(chunks), item)

    samples, failures = {}, []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, item): index for index, item in enumerate(items)}
        for completed, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            try:
                samples[index] = future.result()
            except Exception as exc:
                failures.append({"title": items[index].get("title", ""), "error": str(exc)})
            if progress:
                progress(completed / max(1, len(items)))
    return [samples[index] for index in sorted(samples)], failures


def perceptual_hash(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.convert("L").resize((32, 32)), dtype=float)
    low = dctn(pixels, norm="ortho")[:8, :8].ravel()[1:]
    return low > np.median(low)


def deduplicate(samples: list[Sample], hamming: int = 3) -> tuple[list[Sample], list[dict]]:
    kept, hashes, removed = [], [], []
    for sample in samples:
        phash = perceptual_hash(sample.image)
        pixels = np.asarray(sample.image.convert("RGBA").resize((64, 64)), dtype=float) / 255
        ratio = sample.image.width / sample.image.height
        match = next((i for i, (sid, previous, old_pixels, old_ratio) in enumerate(hashes)
                      if sample.id == sid or (np.count_nonzero(previous != phash) <= hamming
                                             and abs(ratio - old_ratio) < .01
                                             and np.mean(np.abs(pixels - old_pixels)) < .001)), None)
        if match is not None:
            removed.append({"id": sample.id, "duplicate_of": kept[match].id,
                            "title": sample.metadata.get("title", "")})
        else:
            kept.append(sample)
            hashes.append((sample.id, phash, pixels, ratio))
    return kept, removed
