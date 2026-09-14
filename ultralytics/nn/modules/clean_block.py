from pathlib import Path

path = Path("block.py")
text = path.read_text()

remove_blocks = [
    ("class RCSFusion_MM", "class LCRB"),
    ("class LCRB_LC", None),
]

for start, end in remove_blocks:
    s = text.find(start)
    if s == -1:
        continue

    if end:
        e = text.find(end, s)
        text = text[:s] + text[e:]
    else:
        text = text[:s]

path.write_text(text)