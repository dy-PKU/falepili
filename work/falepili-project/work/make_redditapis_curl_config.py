from pathlib import Path

key_path = Path("work/extracted/falepili-project/key.txt")
target = Path("/private/tmp/falepili-redditapis-curl.conf")
key = next(line.strip() for line in key_path.read_text(encoding="utf-8").splitlines() if line.strip())
target.write_text(
    'silent\nshow-error\nmax-time = 120\n'
    f'header = "Authorization: Bearer {key}"\n'
    'header = "User-Agent: falepili-research-test/0.1"\n',
    encoding="utf-8",
)
target.chmod(0o600)
print(target)
