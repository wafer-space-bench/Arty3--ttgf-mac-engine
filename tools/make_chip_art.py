#!/usr/bin/env python3

"""

Generate chip-art GDS containing the designer name on Metal4.
Merged into the final design GDS by LibreLane via EXTRA_GDS_FILES.

The polygons live in a standalone cell named `chip_art`. LibreLane's
EXTRA_GDS_FILES merge keeps cells of the same name as the design top
separate (which produces a "multiple top cells" error in the KLayout
render step), so we deliberately use a unique cell name and rely on the
merge to add the geometry inside the top cell's coordinate frame.

GF180MCU GDS layers used here:
  Metal4 = layer 46, datatype 0   (verify in KLayout if uncertain)

Metal5 (layer 81) is RESERVED by Tiny Tapeout for power - do not use.

"""

if __name__ == "__main__":

	import math
	import gdstk
	import os

	CELL_NAME		= "chip_art"
	NAME			= "L. GODDIJN"
	METAL4_LAYER	= 46
	METAL4_DT		= 0
	TEXT_HEIGHT_UM	= 8.0
	# 1x2 GF180MCU core area is approximately
	#   X: 3.36 .. 355.6 µm  (~352 µm wide)
	#   Y: 3.92 .. 321.44 µm (~317 µm tall)
	#
	# Placing the signature in the bottom-left corner, rotated 90 deg CCW
	# so it reads vertically (baseline along +Y). At 8 µm character height
	# the text is ~80 µm long, so it occupies roughly
	#   X: 15 .. 23 µm
	#   Y: 15 .. 95 µm
	# which fits the empty Metal4 band visible in the synthesized view.
	ANCHOR			= (15.0, 15.0)   # bottom-left corner of the rotated text
	ROTATION_RAD	= math.pi / 2     # 90 deg CCW -> text reads upward

	# Resolve output path relative to the repo root (parent of tools/),
	# so the script works regardless of the caller's cwd.
	SCRIPT_DIR	= os.path.dirname(os.path.abspath(__file__))
	REPO_ROOT	= os.path.dirname(SCRIPT_DIR)
	OUT_DIR		= os.path.join(REPO_ROOT, "src")
	OUT_PATH	= os.path.join(OUT_DIR, "chip_art.gds")

	# Make sure src/ exists before gdstk tries to open the file for write.
	os.makedirs(OUT_DIR, exist_ok=True)

	lib		= gdstk.Library(name="chip_art")
	cell	= lib.new_cell(CELL_NAME)

	# Generate the text at the origin, then rotate it 90 deg CCW around
	# (0, 0) and translate it to ANCHOR. Rotating about the origin first
	# maps a glyph at (x, y) to (-y, x); after the rotation the text
	# extends in -X from x = 0, so the post-rotation bounding box is
	# x in [-TEXT_HEIGHT_UM, 0] and y in [0, text_length]. Shifting by
	# (ANCHOR.x + TEXT_HEIGHT_UM, ANCHOR.y) places the bottom-left of the
	# rotated text at ANCHOR.
	polys = gdstk.text(
		NAME,
		size		= TEXT_HEIGHT_UM,
		position	= (0.0, 0.0),
		layer		= METAL4_LAYER,
		datatype	= METAL4_DT,
	)

	dx = ANCHOR[0] + TEXT_HEIGHT_UM
	dy = ANCHOR[1]
	for p in polys:
		p.rotate(ROTATION_RAD, (0.0, 0.0))
		p.translate(dx, dy)
		cell.add(p)

	lib.write_gds(OUT_PATH)
	print(
		f"Wrote {OUT_PATH}: '{NAME}' on Metal4, rotated 90 deg, "
		f"anchored at {ANCHOR}, {TEXT_HEIGHT_UM} um glyph height"
	)
