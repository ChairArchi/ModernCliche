from pathlib import Path
import base64

root = Path(__file__).parent / 'app_payload'
raw = b''.join(
    base64.b64decode((root / f'part{i:02d}.b64').read_text())
    for i in range(1, 7)
)
code = raw.decode('utf-8')
exec(compile(code, 'eigendoor_app.py', 'exec'), globals())
