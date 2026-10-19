# Pants design — Sisyphus embroidery on Levi's 501

Blender motion study: a headless mannequin walks in Levi's 501-style jeans, filmed from
left-back → back → right-back over 5 seconds. Sisyphus embroidery sits on the back of the
legs at the knees: he pushes the stone up on the left leg, and on the right leg the stone rolls down.
The walk comes from motion capture, so each stride flexes the artwork.

- `pants_walk.blend` — the generated scene (Blender 5.2+). `CAM_Orbit` is the main shot; `CAM_LeftBack`, `CAM_Back`, `CAM_RightBack` are fixed angles.
- `scripts/build_scene.py` — builds everything: rig + mocap retarget, pants mesh, mannequin, sneakers, PBR materials, HDRI, cameras
- `scripts/levis_trims.py` — paints the 501 back details (yoke, pockets + arcuate, belt loops, leather patch, red tab)
- `scripts/embroidery_art.py` — draws the Sisyphus artwork
- `scripts/preview.py` — fast previews (low-res MP4 or stills) into `render/`

```sh
BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
$BLENDER -b --python scripts/build_scene.py                                   # rebuild pants_walk.blend
$BLENDER -b pants_walk.blend --python scripts/preview.py                      # 5 s orbit preview, half res
$BLENDER -b pants_walk.blend --python scripts/preview.py -- --cams CAM_Back --stills 30 60
$BLENDER -b pants_walk.blend -a                                               # final render (full res, 64 samples)
```

Parameters (denim tint, wear, embroidery size/height, camera orbit) are at the top of `build_scene.py`.
Sources and licenses for the third-party assets: [assets/CREDITS.md](assets/CREDITS.md).
