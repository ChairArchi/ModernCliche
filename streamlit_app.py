from pathlib import Path
import base64

root = Path(__file__).parent / 'spatial_payload'
raw = b''.join(
    base64.b64decode((root / f'part{i:02d}.b64').read_text())
    for i in range(8)
)
code = raw.decode('utf-8')
exec(compile(code, 'modern_cliche_spatial.py', 'exec'), globals())
