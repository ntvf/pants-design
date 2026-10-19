# Pants design — Sisyphus embroidery walk cycle

Blender motion study: a mannequin walking in jeans with an embroidery across the back.
Sisyphus pushes the stone up on the left side, and the stone rolls down on the right.
The mountain peak sits on the center back seam.

- `pants_walk.blend` — the generated scene (open in Blender 5.2+)
- `scripts/build_scene.py` — builds the scene from scratch (rig, walk cycle, pants, materials, cameras)
- `scripts/embroidery_art.py` — draws the embroidery artwork into `textures/embroidery.png`
- `scripts/preview.py` — fast low-res MP4 previews into `render/`

```sh
BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
$BLENDER -b --python scripts/build_scene.py                 # rebuild pants_walk.blend
$BLENDER -b pants_walk.blend --python scripts/preview.py    # previews: CAM_Back, CAM_ThreeQuarter, CAM_CloseUp
```

Tweak parameters at the top of `build_scene.py` (denim color, design size and height, walk speed)
and `embroidery_art.py` (composition, figure scale, thread colors).
