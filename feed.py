import sys, re
from store import save

path = sys.argv[1]
text = open(path, encoding="utf-8").read()

# split on lines starting with "## "
parts = re.split(r"^## +", text, flags=re.MULTILINE)
for part in parts:
    part = part.strip()
    if not part:
        continue
    heading, _, body = part.partition("\n")
    heading = heading.strip("* ")
    body = body.strip()
    if not body:
        continue
    print(f"Saving: {heading}")
    save(body, source="memory-import", title=f"Piyush's notes: {heading}")