"""
Builds the "Sisyphus jeans" walk-cycle scene from scratch and saves pants_walk.blend.

  Blender -b --python scripts/build_scene.py            # build + save
  Blender -b --python scripts/build_scene.py -- --test  # also render test stills

Character faces -Y and walks toward -Y. Cameras sit behind (+Y) so we see the
embroidered back. Wearer's left = +X = viewer's left from behind.
"""
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import embroidery_art  # noqa: E402

# ------------------------------------------------------------------ parameters
CYCLE = 32                 # frames per full walk cycle (2 steps) at 24 fps
N_CYCLES = 4
HIP_Z, KNEE_Z, ANKLE_Z = 0.92, 0.50, 0.08
LEG_X = 0.095              # hip joint offset from center

DENIM_COLOR = (0.016, 0.026, 0.075)   # Levi's indigo. Dickies: (0.03,0.03,0.03) or (0.35,0.28,0.18)
DESIGN_WIDTH = 0.42        # meters the artwork spans across the back
DESIGN_ASPECT = 0.75       # artwork height / width (2 x 1.5 canvas)
DESIGN_CENTER_Z = 0.865    # height of the artwork center: peak ~0.98 under waistband, slopes end mid-thigh
TEE_COLOR = (0.30, 0.31, 0.33)  # heather grey t-shirt

# pants cross-section keys: z, pelvis half-width, back depth, front depth,
# blend (0 = pelvis "D" half, 1 = round leg), leg rx, leg ry back, leg ry front
PANTS_PROFILE = [
    (1.03, 0.165, 0.110, 0.100, 0.0, 0.08, 0.08, 0.08),
    (0.96, 0.180, 0.130, 0.110, 0.0, 0.08, 0.08, 0.08),
    (0.88, 0.186, 0.138, 0.110, 0.0, 0.09, 0.09, 0.09),
    (0.81, 0.180, 0.120, 0.105, 0.55, 0.092, 0.10, 0.09),
    (0.76, 0.170, 0.110, 0.100, 1.0, 0.090, 0.095, 0.088),
    (0.62, 0.170, 0.110, 0.100, 1.0, 0.078, 0.082, 0.078),
    (0.50, 0.170, 0.110, 0.100, 1.0, 0.066, 0.068, 0.066),
    (0.30, 0.170, 0.110, 0.100, 1.0, 0.062, 0.064, 0.062),
    (0.06, 0.170, 0.110, 0.100, 1.0, 0.062, 0.064, 0.062),
]
RING_N = 48

# ------------------------------------------------------------------ helpers


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def smoothstep(e0, e1, x):
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def fcurves_of(id_data):
    from bpy_extras.anim_utils import action_get_channelbag_for_slot
    ad = id_data.animation_data
    bag = action_get_channelbag_for_slot(ad.action, ad.action_slot)
    return bag.fcurves if bag else []


def mix_node(nt, data_type="RGBA"):
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type = data_type
    ins = [s for s in n.inputs if s.enabled]
    outs = [s for s in n.outputs if s.enabled]
    return n, ins[0], ins[1], ins[2], outs[0]   # node, fac, A, B, result


# ------------------------------------------------------------------ pants mesh


def ray_ellipse(cx, c, s, a, b):
    """Distance from (cx,0) along (c,s) to ellipse x²/a² + y²/b² = 1."""
    A = c * c / (a * a) + s * s / (b * b)
    B = 2 * cx * c / (a * a)
    C = cx * cx / (a * a) - 1
    return (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)


def ring_radius(theta, side, key):
    z, pa, pb_back, pb_front, blend, rx, ry_back, ry_front = key
    cx = side * LEG_X
    c, s = math.cos(theta), math.sin(theta)
    # pelvis half ("D" shape: half ellipse closed by the center seam at x=0)
    r_d = ray_ellipse(cx, c, s, pa, pb_back if s > 0 else pb_front)
    if side * c < -1e-6:
        r_d = min(r_d, -cx / c)
    # leg ellipse around the leg center
    ry = ry_back if s > 0 else ry_front
    r_e = 1.0 / math.sqrt(c * c / (rx * rx) + s * s / (ry * ry))
    return lerp(r_d, r_e, blend)


def interp_profile(z):
    keys = PANTS_PROFILE
    for k0, k1 in zip(keys[:-1], keys[1:]):
        if k1[0] <= z <= k0[0]:
            t = (k0[0] - z) / (k0[0] - k1[0])
            return tuple(lerp(a, b, t) for a, b in zip(k0, k1))
    return keys[0] if z > keys[0][0] else keys[-1]


def build_pants():
    bm = bmesh.new()
    top, bottom = PANTS_PROFILE[0][0], PANTS_PROFILE[-1][0]
    zs = []
    z = top
    while z > bottom:
        zs.append(z)
        z -= 0.02 if z > 0.7 else 0.035
    zs.append(bottom)

    for side in (1, -1):
        rings = []
        for z in zs:
            key = interp_profile(z)
            ring = []
            for i in range(RING_N):
                th = 2 * math.pi * i / RING_N
                r = ring_radius(th, side, key)
                ring.append(bm.verts.new((side * LEG_X + r * math.cos(th), r * math.sin(th), z)))
            rings.append(ring)
        for r0, r1 in zip(rings[:-1], rings[1:]):
            for i in range(RING_N):
                j = (i + 1) % RING_N
                bm.faces.new((r0[i], r0[j], r1[j], r1[i]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()

    me = bpy.data.meshes.new("Pants")
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True

    # Design UV = raw (x, z) of the rest shape in meters, so the print sticks to the fabric
    uv = me.uv_layers.new(name="DesignUV")
    for loop in me.loops:
        co = me.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = (co.x, co.z)
    # where the fabric faces backwards (+Y) at rest -> embroidery allowed
    mask = me.attributes.new("back_mask", "FLOAT", "POINT")
    for v in me.vertices:
        mask.data[v.index].value = smoothstep(0.1, 0.45, v.normal.y)

    obj = link(bpy.data.objects.new("Pants", me))

    # analytic skin weights
    groups = {n: obj.vertex_groups.new(name=n)
              for n in ("pelvis", "thigh.L", "shin.L", "thigh.R", "shin.R")}
    for v in me.vertices:
        x, z = v.co.x, v.co.z
        sfx = ".L" if x > 0 or (x == 0 and v.index < len(me.vertices) // 2) else ".R"
        w_pel = smoothstep(0.80, 0.99, z)
        seam = max(0.0, 1 - abs(x) / 0.06) * smoothstep(0.74, 0.84, z)
        w_pel = max(w_pel, 0.85 * seam)
        w_shin = smoothstep(KNEE_Z + 0.05, KNEE_Z - 0.06, z)
        w_thigh = (1 - w_shin) * (1 - w_pel)
        w_shin *= (1 - w_pel)
        for name, w in (("pelvis", w_pel), ("thigh" + sfx, w_thigh), ("shin" + sfx, w_shin)):
            if w > 1e-4:
                groups[name].add([v.index], w, "REPLACE")
    return obj


def build_torso():
    """Mannequin torso wearing a loose t-shirt whose hem hangs over the waistband."""
    keys = [(0.985, 0.192, 0.135), (1.02, 0.190, 0.132), (1.10, 0.178, 0.118),
            (1.22, 0.172, 0.112), (1.34, 0.182, 0.114), (1.43, 0.196, 0.112),
            (1.50, 0.200, 0.100), (1.555, 0.150, 0.085), (1.575, 0.075, 0.062),
            (1.60, 0.052, 0.052), (1.70, 0.050, 0.050)]
    n = 32
    bm = bmesh.new()
    rings = []
    for z, rx, ry in keys:
        rings.append([bm.verts.new((rx * math.cos(2 * math.pi * i / n),
                                    ry * math.sin(2 * math.pi * i / n), z)) for i in range(n)])
    for r0, r1 in zip(rings[:-1], rings[1:]):
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((r0[i], r0[j], r1[j], r1[i]))
    bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("Torso")
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
        p.material_index = 1 if p.center.z > 1.57 else 0   # 0 = tee, 1 = mannequin neck
    obj = link(bpy.data.objects.new("Torso_Tee", me))
    sol = obj.modifiers.new("Solidify", "SOLIDIFY")
    sol.thickness = 0.004
    obj.modifiers.new("Subsurf", "SUBSURF").levels = 2
    return obj


def build_shoe(name, x):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = link(bpy.data.objects.new(name, me))
    obj.scale = (0.10, 0.27, 0.085)
    obj.location = (x, -0.06, 0.0425)
    bev = obj.modifiers.new("Bevel", "BEVEL")
    bev.width = 0.3
    bev.segments = 4
    obj.modifiers.new("Subsurf", "SUBSURF").levels = 2
    for p in me.polygons:
        p.use_smooth = True
    return obj


# ------------------------------------------------------------------ rig


def build_rig():
    arm_data = bpy.data.armatures.new("Rig")
    arm = link(bpy.data.objects.new("Rig", arm_data))
    arm.show_in_front = True
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones

    def bone(name, head, tail, parent=None, connect=False):
        b = eb.new(name)
        b.head, b.tail = head, tail
        b.roll = 0
        if parent:
            b.parent = eb[parent]
            b.use_connect = connect
        return b

    bone("pelvis", (0, 0, HIP_Z), (0, 0, HIP_Z + 0.2))
    bone("spine", (0, 0, HIP_Z + 0.2), (0, 0, 1.45), "pelvis", True)
    for sfx, s in ((".L", 1), (".R", -1)):
        x = s * LEG_X
        bone("thigh" + sfx, (x, 0, HIP_Z), (x, 0, KNEE_Z), "pelvis")
        bone("shin" + sfx, (x, 0, KNEE_Z), (x, 0, ANKLE_Z), "thigh" + sfx, True)
        bone("foot" + sfx, (x, 0, ANKLE_Z), (x, -0.13, 0.02), "shin" + sfx, True)
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    return arm


def parent_to_bone(obj, arm, bone_name):
    b = arm.data.bones[bone_name]
    mw = Matrix.LocRotScale(obj.location, obj.rotation_euler, obj.scale)
    obj.parent = arm
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    tail_mat = arm.matrix_world @ b.matrix_local @ Matrix.Translation((0, b.length, 0))
    obj.matrix_parent_inverse = tail_mat.inverted()
    obj.matrix_basis = mw


def world_rot_to_local(arm, bone_name, world_rot):
    """Rotation expressed around world axes -> pose bone local quaternion."""
    m = arm.data.bones[bone_name].matrix_local.to_3x3()
    return (m.inverted() @ world_rot.to_3x3() @ m).to_quaternion()


def periodic_cubic(keys, t):
    """Catmull-Rom through (phase 0..1, value) keys, periodic."""
    ks = sorted(keys)
    n = len(ks)
    t %= 1.0
    i = n - 1
    for k in range(n):
        if ks[k][0] <= t:
            i = k
    t0 = ks[i][0]
    t1 = ks[(i + 1) % n][0] + (1.0 if i == n - 1 else 0.0)
    if t < t0:
        t += 1.0
    u = (t - t0) / (t1 - t0)
    v0, v1, v2, v3 = (ks[(i + d) % n][1] for d in (-1, 0, 1, 2))
    return 0.5 * ((2 * v1) + (-v0 + v2) * u + (2 * v0 - 5 * v1 + 4 * v2 - v3) * u * u
                  + (-v0 + 3 * v1 - 3 * v2 + v3) * u ** 3)


# walk curves for the LEFT leg, phase 0 = left heel strike. Degrees.
THIGH_FLEX = [(0.0, 25), (0.15, 20), (0.3, 6), (0.5, -15), (0.6, -10), (0.75, 16), (0.88, 28)]
KNEE_BEND = [(0.0, 3), (0.1, 16), (0.3, 6), (0.45, 10), (0.55, 38), (0.65, 58), (0.78, 35), (0.92, 4)]
ANKLE_DORSI = [(0.0, 15), (0.1, 0), (0.35, 8), (0.5, -10), (0.58, -22), (0.7, 0), (0.85, 8)]
PELVIS_BOB = [(0.0, -0.015), (0.25, 0.012), (0.5, -0.015), (0.75, 0.012)]
PELVIS_YAW = [(0.0, -5), (0.5, 5)]          # left hip forward at left contact
PELVIS_TILT = [(0.0, 0), (0.25, -3), (0.5, 0), (0.75, 3)]


def animate_walk(arm):
    X, Y, Z = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
    rot = lambda axis, deg: Matrix.Rotation(math.radians(deg), 4, axis)  # noqa: E731
    pb = arm.pose.bones
    step = 2
    for f in range(0, CYCLE + 1, step):
        ph = f / CYCLE
        frame = 1 + f
        for sfx, off in ((".L", 0.0), (".R", 0.5)):
            p = ph + off
            pb["thigh" + sfx].rotation_quaternion = world_rot_to_local(
                arm, "thigh" + sfx, rot(X, -periodic_cubic(THIGH_FLEX, p)))
            pb["shin" + sfx].rotation_quaternion = world_rot_to_local(
                arm, "shin" + sfx, rot(X, periodic_cubic(KNEE_BEND, p)))
            pb["foot" + sfx].rotation_quaternion = world_rot_to_local(
                arm, "foot" + sfx, rot(X, -periodic_cubic(ANKLE_DORSI, p)))
            for b in ("thigh", "shin", "foot"):
                pb[b + sfx].keyframe_insert("rotation_quaternion", frame=frame)
        prot = rot(Z, periodic_cubic(PELVIS_YAW, ph)) @ rot(Y, periodic_cubic(PELVIS_TILT, ph))
        pb["pelvis"].rotation_quaternion = world_rot_to_local(arm, "pelvis", prot)
        pb["pelvis"].location = (0, periodic_cubic(PELVIS_BOB, ph), 0)  # local Y = world Z
        # counter-rotate the chest so shoulders stay calmer than hips
        pb["spine"].rotation_quaternion = world_rot_to_local(
            arm, "spine", rot(Z, -0.6 * periodic_cubic(PELVIS_YAW, ph)))
        for b, path in (("pelvis", "rotation_quaternion"), ("pelvis", "location"),
                        ("spine", "rotation_quaternion")):
            pb[b].keyframe_insert(path, frame=frame)

    for fc in fcurves_of(arm):
        fc.modifiers.new("CYCLES")

    # forward speed = how fast the stance foot slides back relative to the body
    scn = bpy.context.scene
    ys = []
    for f in (1, 1 + CYCLE // 2):
        scn.frame_set(f)
        ys.append((arm.matrix_world @ pb["foot.L"].head).y)
    speed = (ys[1] - ys[0]) / (CYCLE / 2)          # m per frame
    print(f"walk speed: {speed * 24:.2f} m/s, stride {speed * CYCLE:.2f} m/cycle")
    return speed


def animate_linear_y(obj, speed, z_offset=None):
    obj.keyframe_insert("location", index=1, frame=1)
    obj.location.y -= speed * CYCLE
    obj.keyframe_insert("location", index=1, frame=1 + CYCLE)
    for fc in fcurves_of(obj):
        fc.extrapolation = "LINEAR"
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


# ------------------------------------------------------------------ materials


def denim_material(img_path):
    mat = bpy.data.materials.new("Denim_Embroidery")
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (1400, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (1100, 0)

    uvn = nodes.new("ShaderNodeUVMap")
    uvn.uv_map = "DesignUV"
    uvn.location = (-1200, 0)

    # --- embroidery placement: edit Scale/Location here to move/resize the art
    mapping = nodes.new("ShaderNodeMapping")
    mapping.name = mapping.label = "Embroidery Placement"
    mapping.location = (-950, 200)
    h = DESIGN_WIDTH * DESIGN_ASPECT
    mapping.inputs["Scale"].default_value = (-1 / DESIGN_WIDTH, 1 / h, 1)
    mapping.inputs["Location"].default_value = (0.5, 0.5 - DESIGN_CENTER_Z / h, 0)
    links.new(uvn.outputs["UV"], mapping.inputs["Vector"])

    tex = nodes.new("ShaderNodeTexImage")
    tex.name = tex.label = "Embroidery Art"
    tex.location = (-700, 250)
    tex.extension = "CLIP"
    tex.image = bpy.data.images.load(img_path)
    tex.image.alpha_mode = "STRAIGHT"
    links.new(mapping.outputs["Vector"], tex.inputs["Vector"])

    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "back_mask"
    attr.location = (-700, 450)
    emb = nodes.new("ShaderNodeMath")
    emb.operation = "MULTIPLY"
    emb.label = "embroidery coverage"
    emb.location = (-400, 350)
    links.new(tex.outputs["Alpha"], emb.inputs[0])
    links.new(attr.outputs["Fac"], emb.inputs[1])

    # --- denim base: twill + mottled fading
    twill = nodes.new("ShaderNodeTexWave")
    twill.wave_type = "BANDS"
    twill.bands_direction = "DIAGONAL"
    twill.inputs["Scale"].default_value = 900
    twill.location = (-700, -250)
    links.new(uvn.outputs["UV"], twill.inputs["Vector"])
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 6
    noise.inputs["Detail"].default_value = 6
    noise.location = (-700, -500)
    links.new(uvn.outputs["UV"], noise.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-450, -500)
    ramp.color_ramp.elements[0].color = (*DENIM_COLOR, 1)
    ramp.color_ramp.elements[1].color = (DENIM_COLOR[0] * 1.6, DENIM_COLOR[1] * 1.5,
                                         DENIM_COLOR[2] * 1.4, 1)
    ramp.color_ramp.elements[0].position = 0.4
    ramp.color_ramp.elements[1].position = 0.7
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])

    # --- satin stitch pattern on the thread
    stitch = nodes.new("ShaderNodeTexWave")
    stitch.wave_type = "BANDS"
    stitch.bands_direction = "Y"
    stitch.inputs["Scale"].default_value = 1400
    stitch.location = (-700, 0)
    links.new(uvn.outputs["UV"], stitch.inputs["Vector"])

    col_mix, fac, a, b, res = mix_node(nt)
    col_mix.location = (700, 250)
    links.new(emb.outputs[0], fac)
    links.new(ramp.outputs["Color"], a)
    links.new(tex.outputs["Color"], b)
    links.new(res, bsdf.inputs["Base Color"])

    rough_mix, rfac, ra, rb, rres = mix_node(nt, "FLOAT")
    rough_mix.location = (700, 0)
    links.new(emb.outputs[0], rfac)
    ra.default_value, rb.default_value = 0.9, 0.42
    links.new(rres, bsdf.inputs["Roughness"])

    # height: raised thread with satin ridges + faint denim twill
    th = nodes.new("ShaderNodeMath")
    th.operation = "MULTIPLY_ADD"
    th.location = (-150, 0)
    links.new(stitch.outputs["Fac"], th.inputs[0])
    th.inputs[1].default_value = 0.35
    th.inputs[2].default_value = 0.65
    th2 = nodes.new("ShaderNodeMath")
    th2.operation = "MULTIPLY"
    th2.location = (50, 0)
    links.new(th.outputs[0], th2.inputs[0])
    links.new(emb.outputs[0], th2.inputs[1])
    tw = nodes.new("ShaderNodeMath")
    tw.operation = "MULTIPLY_ADD"
    tw.location = (250, -150)
    links.new(twill.outputs["Fac"], tw.inputs[0])
    tw.inputs[1].default_value = 0.08
    links.new(th2.outputs[0], tw.inputs[2])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.7
    bump.inputs["Distance"].default_value = 0.002
    bump.location = (800, -250)
    links.new(tw.outputs[0], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def simple_material(name, color, rough=0.5):
    mat = bpy.data.materials.new(name)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = rough
    return mat


def floor_material():
    mat = bpy.data.materials.new("Floor")
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.inputs["Scale"].default_value = 1.5
    brick.inputs["Color1"].default_value = (0.42, 0.41, 0.40, 1)
    brick.inputs["Color2"].default_value = (0.38, 0.37, 0.36, 1)
    brick.inputs["Mortar"].default_value = (0.25, 0.25, 0.25, 1)
    brick.inputs["Mortar Size"].default_value = 0.006
    brick.offset = 0.0
    brick.inputs["Row Height"].default_value = 0.5
    brick.inputs["Brick Width"].default_value = 0.5
    nt.links.new(brick.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.8
    return mat


# ------------------------------------------------------------------ scene


def make_camera(name, loc, target_z, lens, rig):
    cam = link(bpy.data.objects.new(name, bpy.data.cameras.new(name)))
    cam.data.lens = lens
    cam.location = loc
    cam.parent = rig
    tgt = link(bpy.data.objects.new(name + "_target", None))
    tgt.location = (0, 0, target_z)
    tgt.parent = rig
    tgt.empty_display_size = 0.05
    c = cam.constraints.new("TRACK_TO")
    c.target = tgt
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    return cam


def build():
    clear_scene()
    scn = bpy.context.scene
    scn.render.fps = 24
    scn.frame_start, scn.frame_end = 1, CYCLE * N_CYCLES
    scn.render.resolution_x, scn.render.resolution_y = 1080, 1920
    scn.render.engine = "BLENDER_EEVEE"
    scn.view_settings.view_transform = "AgX"

    art = embroidery_art.save_png(os.path.join(ROOT, "textures", "embroidery.png"))

    arm = build_rig()
    pants = build_pants()
    pants.data.materials.append(denim_material(art))
    pants.parent = arm
    am = pants.modifiers.new("Armature", "ARMATURE")
    am.object = arm
    sol = pants.modifiers.new("Solidify", "SOLIDIFY")
    sol.thickness = 0.006
    sol.offset = 1.0
    pants.modifiers.new("Subsurf", "SUBSURF").levels = 2

    skin = simple_material("Mannequin", (0.80, 0.78, 0.74), 0.45)
    torso = build_torso()
    torso.data.materials.append(simple_material("Tee_Cotton", TEE_COLOR, 0.95))
    torso.data.materials.append(skin)
    parent_to_bone(torso, arm, "spine")
    shoe_mat = simple_material("Sneaker", (0.85, 0.85, 0.83), 0.6)
    for sfx, s in ((".L", 1), (".R", -1)):
        shoe = build_shoe("Shoe" + sfx, s * LEG_X)
        shoe.data.materials.append(shoe_mat)
        parent_to_bone(shoe, arm, "foot" + sfx)

    speed = animate_walk(arm)
    animate_linear_y(arm, speed)

    # floor
    bpy.ops.mesh.primitive_plane_add(size=1)
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.scale = (8, 40, 1)
    floor.location = (0, -15, 0)
    floor.data.materials.append(floor_material())

    # camera rig follows the walker (no bob)
    rig = link(bpy.data.objects.new("CameraRig", None))
    animate_linear_y(rig, speed)
    cam_back = make_camera("CAM_Back", (0, 3.4, 0.70), 0.66, 85, rig)
    make_camera("CAM_ThreeQuarter", (1.9, 2.8, 0.85), 0.68, 70, rig)
    make_camera("CAM_CloseUp", (0.25, 1.55, 0.85), 0.80, 85, rig)
    scn.camera = cam_back

    # lights
    sun = link(bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN")))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), math.radians(-25), math.radians(160))
    key = link(bpy.data.objects.new("Key", bpy.data.lights.new("Key", "AREA")))
    key.data.energy = 250
    key.data.size = 1.5
    key.location = (-1.6, 2.0, 2.2)
    key.parent = rig
    c = key.constraints.new("TRACK_TO")
    c.target = bpy.data.objects["CAM_Back_target"]
    world = bpy.data.worlds.new("World")
    scn.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.57, 0.6, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.7

    # video output
    if hasattr(scn.render.image_settings, "media_type"):
        scn.render.image_settings.media_type = "VIDEO"
    scn.render.image_settings.file_format = "FFMPEG"
    scn.render.ffmpeg.format = "MPEG4"
    scn.render.ffmpeg.codec = "H264"
    scn.render.filepath = "//render/walk_"
    scn.frame_set(1)

    path = os.path.join(ROOT, "pants_walk.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path, relative_remap=True)
    print("saved", path)


def test_renders(out_dir, frames=(1, 8, 15), cams=("CAM_Back", "CAM_ThreeQuarter", "CAM_CloseUp")):
    scn = bpy.context.scene
    if hasattr(scn.render.image_settings, "media_type"):
        scn.render.image_settings.media_type = "IMAGE"
    scn.render.image_settings.file_format = "PNG"
    scn.render.resolution_percentage = 40
    for cam in cams:
        scn.camera = bpy.data.objects[cam]
        for f in frames:
            scn.frame_set(f)
            scn.render.filepath = os.path.join(out_dir, f"{cam}_{f:03d}.png")
            bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    build()
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--test" in argv:
        test_renders(argv[argv.index("--test") + 1])
