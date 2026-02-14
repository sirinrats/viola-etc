from pathlib import Path
import json
import re

GRID_DIR = Path("NewEra_grids_R300K")

pat = re.compile(r"lte(\d{5})")

teffs = []
for p in GRID_DIR.glob("*.csv"):
    m = pat.search(p.name)
    if m:
        teffs.append(int(m.group(1)))

teffs = sorted(set(teffs))
out = Path("viola_etc/assets/newera_teff_list.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"teff_list": teffs}, indent=2))
print("Wrote:", out, "N=", len(teffs))
