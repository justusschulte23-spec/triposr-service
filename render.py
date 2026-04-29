#!/usr/bin/env python3
# Blender headless script — called via: blender -b -P render.py -- <glb_path> <output_dir>
import bpy
import sys
import os
import math

def get_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    return argv

args = get_args()
if len(args) < 2:
    print("Usage: blender -b -P render.py -- <glb_path> <output_dir>")
    sys.exit(1)

glb_path   = args[0]
output_dir = args[1]

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import GLB
bpy.ops.import_scene.gltf(filepath=glb_path)

# Get imported objects
objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not objects:
    print("[Blender] No mesh objects found in GLB")
    sys.exit(1)

# Center and normalize
bpy.ops.object.select_all(action='DESELECT')
for obj in objects:
    obj.select_set(True)
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
bpy.context.view_layer.objects.active = objects[0]
bpy.ops.object.location_clear()

# Get bounding box for camera distance
max_dim = max(obj.dimensions.length for obj in objects)
cam_dist = max(max_dim * 1.8, 0.5)

# Setup lighting — 3-point light rig
def add_light(name, loc, energy=500, ltype='AREA'):
    bpy.ops.object.light_add(type=ltype, location=loc)
    light = bpy.context.active_object
    light.name = name
    light.data.energy = energy
    return light

add_light('Key',   ( cam_dist,  -cam_dist, cam_dist), energy=800)
add_light('Fill',  (-cam_dist,  -cam_dist, 0),        energy=300)
add_light('Rim',   (0,           cam_dist, cam_dist), energy=200)

# Setup render settings
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 64
scene.render.resolution_x = 1024
scene.render.resolution_y = 1024
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'PNG'

# Add camera
bpy.ops.object.camera_add()
cam = bpy.context.active_object
scene.camera = cam

def set_camera(angle_deg, tilt_deg=15):
    # Position camera at given horizontal angle and slight downward tilt
    angle_rad = math.radians(angle_deg)
    tilt_rad  = math.radians(tilt_deg)
    x = cam_dist * math.sin(angle_rad)
    y = -cam_dist * math.cos(angle_rad)
    z = cam_dist * math.sin(tilt_rad)
    cam.location = (x, y, z)
    # Point camera at origin
    dx, dy, dz = -x, -y, -z
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    cam.rotation_euler[0] = math.atan2(math.sqrt(dx*dx + dy*dy), -dz) - math.pi
    cam.rotation_euler[2] = math.atan2(dx, dy)
    bpy.ops.object.select_all(action='DESELECT')
    cam.select_set(True)
    bpy.context.view_layer.objects.active = cam
    bpy.ops.view3d.camera_to_view_selected() if False else None
    # Use track-to constraint instead
    if not cam.constraints:
        ct = cam.constraints.new('TRACK_TO')
        ct.target = objects[0]
        ct.track_axis = 'TRACK_NEGATIVE_Z'
        ct.up_axis = 'UP_Y'
    cam.location = (x, y, z)

# Render 1: Front view (0°)
set_camera(0, tilt_deg=10)
scene.render.filepath = os.path.join(output_dir, 'front.png')
bpy.ops.render.render(write_still=True)
print(f"[Blender] Rendered front: {scene.render.filepath}")

# Render 2: Side angle (45°)
set_camera(45, tilt_deg=20)
scene.render.filepath = os.path.join(output_dir, 'side.png')
bpy.ops.render.render(write_still=True)
print(f"[Blender] Rendered side: {scene.render.filepath}")

print("[Blender] Done.")
