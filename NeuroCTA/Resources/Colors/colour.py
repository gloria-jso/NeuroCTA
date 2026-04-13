import re

input_file = "/Users/gloriaso/Desktop/BME499/NeuroCTA/NeuroCTA/Resources/Colors/labelmap_topbrain_ct.txt"
output_file = "labelmap_topbrain_ct.txt"

rows = []
with open(input_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+[\d.]+\s+\d+\s+\d+\s+"([^"]+)"', line)
        if match:
            idx, r, g, b, label = match.groups()
            rows.append(f"{idx} {label} {r} {g} {b} 255")

with open(output_file, "w") as f:
    f.write("\n".join(rows))

print(f"Wrote {len(rows)} rows to {output_file}")