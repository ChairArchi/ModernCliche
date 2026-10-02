from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse, urlunparse

import requests
from PIL import Image, ImageOps

USER_AGENT = "ModernClicheDatasetCollector/0.1 (personal research tool)"
OPENVERSE_ENDPOINT = "https://api.openverse.org/v1/images/"
WIKIMEDIA_ENDPOINT = "https://commons.wikimedia.org/w/api.php"


@dataclass
class Candidate:
    source: str
    query: str
    title: str
    image_url: str
    thumbnail_url: str = ""
    page_url: str = ""
    width: int | None = None
    height: int | None = None
    license: str = ""
    license_url: str = ""
    creator: str = ""
    provider: str = ""
    source_id: str = ""

    def record(self) -> dict:
        return asdict(self)


GATE_STARTER_QUERIES = [
    "torii", "鳥居", "shinto torii gate", "hongsalmun", "홍살문",
    "paifang", "pailou", "牌坊", "牌楼", "ceremonial gateway",
    "historic gateway", "monumental gateway", "temple gateway",
    "shrine gateway", "stone gateway", "post and lintel gate",
    "trilithon gate", "ritual gateway", "symbolic gateway",
    "triumphal arch", "torana gateway", "gopuram gateway",
    "Japanese temple gate", "Korean temple gate", "Chinese ceremonial gate",
]


def clean_queries(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in text.splitlines():
        q = re.sub(r"\s+", " ", line).strip()
        if q and q.casefold() not in seen:
            seen.add(q.casefold())
            out.append(q)
    return out


def expand_queries(base: Iterable[str], suffixes: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    suffixes = [s.strip() for s in suffixes if s.strip()]
    for q in base:
        for value in [q] + [f"{q} {suffix}" for suffix in suffixes]:
            key = value.casefold()
            if key not in seen:
                seen.add(key)
                out.append(value)
    return out


def _int_or_none(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _meta_value(meta: dict, key: str) -> str:
    raw = meta.get(key, {})
    if isinstance(raw, dict):
        return str(raw.get("value") or "")
    return str(raw or "")


def search_ddgs(query: str, max_results: int, *, region: str = "us-en", backend: str = "auto", license_filter: str | None = None, photo_only: bool = True) -> list[Candidate]:
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError("DDGS is not installed. Run: pip install -r requirements.txt") from exc
    rows = DDGS(timeout=15).images(
        query=query, region=region, safesearch="moderate", max_results=max_results,
        backend=backend, type_image="photo" if photo_only else None,
        license_image=license_filter,
    )
    out: list[Candidate] = []
    for row in rows:
        url = str(row.get("image") or "").strip()
        if not url:
            continue
        out.append(Candidate(
            source=f"ddgs:{row.get('source') or backend}", query=query,
            title=str(row.get("title") or ""), image_url=url,
            thumbnail_url=str(row.get("thumbnail") or ""),
            page_url=str(row.get("url") or ""),
            width=_int_or_none(row.get("width")), height=_int_or_none(row.get("height")),
            provider=str(row.get("source") or backend),
        ))
    return out


def search_wikimedia(query: str, max_results: int, session: requests.Session | None = None) -> list[Candidate]:
    session = session or requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    results: list[Candidate] = []
    continuation: dict[str, str | int] = {}
    while len(results) < max_results:
        params: dict[str, str | int] = {
            "action": "query", "format": "json", "formatversion": 2,
            "generator": "search", "gsrsearch": query, "gsrnamespace": 6,
            "gsrlimit": min(50, max_results - len(results)),
            "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 512,
        }
        params.update(continuation)
        response = session.get(WIKIMEDIA_ENDPOINT, params=params, timeout=25)
        response.raise_for_status()
        payload = response.json()
        pages = payload.get("query", {}).get("pages", [])
        if isinstance(pages, dict):
            pages = list(pages.values())
        for page in pages:
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            mime = str(info.get("mime") or "")
            if mime and not mime.startswith("image/"):
                continue
            ext = info.get("extmetadata") or {}
            results.append(Candidate(
                source="wikimedia", query=query, title=str(page.get("title") or ""),
                image_url=str(info.get("url") or ""), thumbnail_url=str(info.get("thumburl") or ""),
                page_url=str(info.get("descriptionurl") or ""), width=_int_or_none(info.get("width")),
                height=_int_or_none(info.get("height")),
                license=_meta_value(ext, "LicenseShortName") or _meta_value(ext, "UsageTerms"),
                license_url=_meta_value(ext, "LicenseUrl"), creator=_meta_value(ext, "Artist"),
                provider="Wikimedia Commons", source_id=str(page.get("pageid") or ""),
            ))
            if len(results) >= max_results:
                break
        if "continue" not in payload or not pages:
            break
        continuation = payload["continue"]
        time.sleep(0.08)
    return results


def search_openverse(query: str, max_results: int, session: requests.Session | None = None) -> list[Candidate]:
    session = session or requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    out: list[Candidate] = []
    page = 1
    while len(out) < max_results:
        page_size = min(50, max_results - len(out))
        response = session.get(OPENVERSE_ENDPOINT, params={"q": query, "page": page, "page_size": page_size}, timeout=25)
        if response.status_code == 429:
            retry_after = min(int(response.headers.get("Retry-After", "2") or 2), 15)
            time.sleep(retry_after)
            continue
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("results") or []
        for row in rows:
            url = str(row.get("url") or "")
            if not url:
                continue
            out.append(Candidate(
                source="openverse", query=query, title=str(row.get("title") or ""), image_url=url,
                thumbnail_url=str(row.get("thumbnail") or ""), page_url=str(row.get("foreign_landing_url") or ""),
                width=_int_or_none(row.get("width")), height=_int_or_none(row.get("height")),
                license=str(row.get("license") or ""), license_url=str(row.get("license_url") or ""),
                creator=str(row.get("creator") or ""), provider=str(row.get("provider") or row.get("source") or "Openverse"),
                source_id=str(row.get("id") or ""),
            ))
            if len(out) >= max_results:
                break
        if not rows or not payload.get("next"):
            break
        page += 1
        time.sleep(0.1)
    return out


def normalized_url(url: str) -> str:
    try:
        parsed = urlparse(url.strip())
        return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, parsed.params, parsed.query, ""))
    except Exception:
        return url.strip()


def dedupe_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    out: list[Candidate] = []
    for item in candidates:
        key = normalized_url(item.image_url)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def candidate_passes_dimensions(item: Candidate, min_short_edge: int, min_long_edge: int, allow_unknown: bool) -> bool:
    if item.width is None or item.height is None:
        return allow_unknown
    short_edge, long_edge = sorted((item.width, item.height))
    return short_edge >= min_short_edge and long_edge >= min_long_edge


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    gray = ImageOps.grayscale(image).resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            value = (value << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _safe_ext(fmt: str | None, content_type: str) -> str:
    by_format = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tif", "BMP": ".bmp"}
    fmt = (fmt or "").upper()
    if fmt in by_format:
        return by_format[fmt]
    content_type = content_type.lower()
    if "jpeg" in content_type: return ".jpg"
    if "png" in content_type: return ".png"
    if "webp" in content_type: return ".webp"
    return ".img"


def _fetch_bytes(session: requests.Session, url: str, max_bytes: int, retries: int = 2) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with session.get(url, stream=True, timeout=(10, 35), allow_redirects=True) as response:
                if response.status_code in {401, 403}:
                    raise RuntimeError(f"HTTP {response.status_code}: access denied")
                if response.status_code == 429:
                    retry_after = min(int(response.headers.get("Retry-After", "2") or 2), 15)
                    time.sleep(retry_after)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                if "svg" in content_type.lower():
                    raise RuntimeError("SVG skipped: raster photographs only")
                buf = bytearray()
                for chunk in response.iter_content(chunk_size=256 * 1024):
                    if chunk:
                        buf.extend(chunk)
                        if len(buf) > max_bytes:
                            raise RuntimeError(f"file exceeds {max_bytes // (1024*1024)} MB limit")
                return bytes(buf), content_type
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.7 * (attempt + 1))
    raise RuntimeError(str(last_error or "download failed"))


def download_dataset(candidates: list[Candidate], output_root: str | Path, *, min_short_edge: int = 700, min_long_edge: int = 1200, max_file_mb: int = 25, near_duplicate_distance: int = 3, progress: Callable[[int, int, str], None] | None = None) -> tuple[Path, list[dict]]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = Path(output_root) / f"dataset_{stamp}"
    image_dir = dataset_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"})
    sha_seen: set[str] = set()
    hashes: list[int] = []
    metadata: list[dict] = []
    max_bytes = max_file_mb * 1024 * 1024

    for index, item in enumerate(candidates, start=1):
        if progress:
            progress(index, len(candidates), item.title or item.image_url)
        base = item.record()
        base.update({"status": "failed", "error": "", "filename": "", "sha256": "", "dhash": ""})
        try:
            raw, content_type = _fetch_bytes(session, item.image_url, max_bytes=max_bytes)
            sha = hashlib.sha256(raw).hexdigest()
            if sha in sha_seen:
                base.update({"status": "duplicate_exact", "sha256": sha})
                metadata.append(base)
                continue
            with Image.open(io.BytesIO(raw)) as image:
                image.verify()
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
                fmt = image.format
                short_edge, long_edge = sorted((width, height))
                if short_edge < min_short_edge or long_edge < min_long_edge:
                    base.update({"status": "rejected_low_resolution", "width_actual": width, "height_actual": height, "sha256": sha})
                    metadata.append(base)
                    continue
                p_hash = dhash(image)
            if any(hamming(p_hash, previous) <= near_duplicate_distance for previous in hashes):
                base.update({"status": "duplicate_near", "width_actual": width, "height_actual": height, "sha256": sha, "dhash": f"{p_hash:016x}"})
                metadata.append(base)
                continue
            ext = _safe_ext(fmt, content_type)
            stem = hashlib.sha1(item.image_url.encode("utf-8", errors="ignore")).hexdigest()[:16]
            filename = f"{stem}{ext}"
            (image_dir / filename).write_bytes(raw)
            sha_seen.add(sha)
            hashes.append(p_hash)
            base.update({"status": "downloaded", "filename": filename, "width_actual": width, "height_actual": height, "sha256": sha, "dhash": f"{p_hash:016x}", "bytes": len(raw), "content_type": content_type})
        except Exception as exc:
            base["error"] = str(exc)
        metadata.append(base)

    dataset_dir.joinpath("metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if metadata:
        columns = sorted({key for row in metadata for key in row.keys()})
        with dataset_dir.joinpath("metadata.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader(); writer.writerows(metadata)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested": len(candidates),
        "downloaded": sum(1 for r in metadata if r["status"] == "downloaded"),
        "duplicate_exact": sum(1 for r in metadata if r["status"] == "duplicate_exact"),
        "duplicate_near": sum(1 for r in metadata if r["status"] == "duplicate_near"),
        "rejected_low_resolution": sum(1 for r in metadata if r["status"] == "rejected_low_resolution"),
        "failed": sum(1 for r in metadata if r["status"] == "failed"),
        "settings": {"min_short_edge": min_short_edge, "min_long_edge": min_long_edge, "max_file_mb": max_file_mb, "near_duplicate_distance": near_duplicate_distance},
    }
    dataset_dir.joinpath("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return dataset_dir, metadata


def make_zip(dataset_dir: str | Path) -> Path:
    dataset_dir = Path(dataset_dir)
    zip_path = dataset_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for path in dataset_dir.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(dataset_dir.parent))
    return zip_path


def export_candidate_index(candidates: Iterable[Candidate]) -> bytes:
    rows = [item.record() for item in candidates]
    if not rows:
        return b""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")
