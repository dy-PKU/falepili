from pathlib import Path

p = Path("/private/tmp/falepili-redditapis-curl.conf")
if p.exists():
    p.write_text("", encoding="utf-8")
    p.unlink()
