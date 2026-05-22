#!/usr/bin/env python3

"""

Generate chip-art GDS containing the designer name on Metal4.
Merged into the final design GDS by LibreLane via EXTRA_GDS_FILES.

GF180MCU GDS layers used here:
  Metal4 = layer 46, datatype 0   (verify in KLayout if uncertain)

Metal5 (layer 81) is RESERVED by Tiny Tapeout for power - do not use.

"""

if __name__ == "__main__":

	import gdstk
	import os

	NAME			= "L. GODDIJN"
	METAL4_LAYER	= 46
	METAL4_DT		= 0
	TEXT_HEIGHT_UM	= 8.0
	ORIGIN			= (450.0, 80.0)  # (x, y) in µm, inside the 704x317 core area

	# Resolve output path relative to the repo root (parent of tools/),
	# so the script works regardless of the caller's cwd.
	SCRIPT_DIR	= os.path.dirname(os.path.abspath(__file__))
	REPO_ROOT	= os.path.dirname(SCRIPT_DIR)
	OUT_DIR		= os.path.join(REPO_ROOT, "src")
	OUT_PATH	= os.path.join(OUT_DIR, "chip_art.gds")

	# Make sure src/ exists before gdstk tries to open the file for write.
	os.makedirs(OUT_DIR, exist_ok=True)

	lib		= gdstk.Library(name="chip_art")
	cell	= lib.new_cell("chip_art")

	polys = gdstk.text(
		NAME,
		size		= TEXT_HEIGHT_UM,
		position	= ORIGIN,
		layer		= METAL4_LAYER,
		datatype	= METAL4_DT,
	)

	for p in polys:
		cell.add(p)

	lib.write_gds(OUT_PATH)
	print(f"Wrote {OUT_PATH}: '{NAME}' on Metal4 at {ORIGIN}, {TEXT_HEIGHT_UM} um tall")
