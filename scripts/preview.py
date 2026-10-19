"""
Fast low-res animation previews of pants_walk.blend -> render/preview_<camera>.mp4
  Blender -b pants_walk.blend --python scripts/preview.py [-- CAM_Back CAM_CloseUp ...]
"""
import sys

import bpy

scn = bpy.context.scene
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
cams = argv or ["CAM_Back", "CAM_ThreeQuarter", "CAM_CloseUp"]

scn.render.resolution_percentage = 50          # 540 x 960
scn.eevee.taa_render_samples = 8
scn.frame_start, scn.frame_end = 1, 64          # two walk cycles
if hasattr(scn.render.image_settings, "media_type"):
    scn.render.image_settings.media_type = "VIDEO"
scn.render.image_settings.file_format = "FFMPEG"
scn.render.ffmpeg.format = "MPEG4"
scn.render.ffmpeg.codec = "H264"
scn.render.ffmpeg.constant_rate_factor = "MEDIUM"

for cam in cams:
    scn.camera = bpy.data.objects[cam]
    scn.render.filepath = f"//render/preview_{cam}_"
    bpy.ops.render.render(animation=True)
    print("PREVIEW", bpy.path.abspath(scn.render.filepath))
