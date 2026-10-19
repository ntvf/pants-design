"""
Fast previews of pants_walk.blend into render/.
  Blender -b pants_walk.blend --python scripts/preview.py                  # low-res MP4 of the orbit shot
  Blender -b pants_walk.blend --python scripts/preview.py -- --stills 1 60 # quick PNG stills
  Blender -b pants_walk.blend --python scripts/preview.py -- --cams CAM_Back CAM_LeftBack
"""
import sys

import bpy

scn = bpy.context.scene
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(flag):
    if flag not in argv:
        return []
    out = []
    for a in argv[argv.index(flag) + 1:]:
        if a.startswith("--"):
            break
        out.append(a)
    return out


cams = opt("--cams") or ["CAM_Orbit"]
stills = [int(f) for f in opt("--stills")]
scn.render.resolution_percentage = int((opt("--pct") or ["50"])[0])
scn.eevee.taa_render_samples = int((opt("--samples") or ["16"])[0])

for cam in cams:
    scn.camera = bpy.data.objects[cam]
    if stills:
        if hasattr(scn.render.image_settings, "media_type"):
            scn.render.image_settings.media_type = "IMAGE"
        scn.render.image_settings.file_format = "PNG"
        for f in stills:
            scn.frame_set(f)
            scn.render.filepath = f"//render/still_{cam}_{f:03d}.png"
            bpy.ops.render.render(write_still=True)
            print("PREVIEW", bpy.path.abspath(scn.render.filepath))
    else:
        scn.render.filepath = f"//render/preview_{cam}_"
        bpy.ops.render.render(animation=True)
        print("PREVIEW", bpy.path.abspath(scn.render.filepath))
