import random
import numpy as np
from mathutils import Vector
import time
import bpy

def computeCell(heightmap, i, j, dist, deltas):
    """
    Compute the value of a cell in the heightmap based on the surrounding cells.
    
    Args:
        heightmap (numpy.ndarray): The heightmap matrix.
        i (int): Row index of the current cell.
        j (int): Column index of the current cell.
        dist (int): Distance of the current cell from the averaging neighbors.
        deltas (list): List of coordinate deltas from the current cell to the averaging neighbors.
        
    Returns:
        float: The computed value of the current cell.
    """
    n = heightmap.shape[0] - 1
    res = 0
    for deltaI, deltaJ in deltas:
        res += heightmap[(i + deltaI * dist) % n, (j + deltaJ * dist) % n]
    return res / 4.0

def dSquareStep(heightmap, a, deviationBoundary):
    """
    Perform a single step of the diamond-square algorithm.
    
    Args:
        heightmap (numpy.ndarray): The heightmap matrix.
        a (int): Size of the square/diamond we are working with in this iteration.
        deviationBoundary (float): Boundary for the random offset we can add to the height.
    """
    n = heightmap.shape[0]
    aHalf = a // 2
    
    # coordinates of the averaging points relative to the current cell (normalized in distance):
    
    #   o----X----o
    #   |    |    |
    #   X----.----X
    #   |    |    |
    #   o----X----o
    deltaDiamond = [ (-1,-1), (-1,1), (1,1), (1,-1) ]
    
    #   X----o----X
    #   |    |    |
    #   o----.----o
    #   |    |    |
    #   X----o----X
    deltaSquare = [ (-1,0), (0,-1), (1,0), (0,1) ]

    for i in range(aHalf, n, a):
        for j in range(aHalf, n, a):
            heightmap[i, j] = computeCell(heightmap, i, j, aHalf, deltaDiamond) + random.uniform(-deviationBoundary, deviationBoundary)
            
    for i in range(aHalf, n, a):
        for j in range(0, n, a):
            heightmap[i, j] = computeCell(heightmap, i, j, aHalf, deltaSquare) + random.uniform(-deviationBoundary, deviationBoundary)

    for i in range(0, n, a):
        for j in range(aHalf, n, a):
            heightmap[i, j] = computeCell(heightmap, i, j, aHalf, deltaSquare) + random.uniform(-deviationBoundary, deviationBoundary)

def generate(size, roughness):
    """
    Generate a heightmap using the diamond-square algorithm.
    
    Args:
        size (int): Size of the heightmap.
        roughness (float): Constant between 0 and 1 which modifies how quickly the deviationBoundary shrinks.
        
    Returns:
        numpy.ndarray: The generated heightmap.
    """
    heightmap = np.zeros(size * size).reshape(size, size)
    deviationBoundary = 1.0
    a = size - 1
    while a > 1:
        dSquareStep(heightmap, a, deviationBoundary)
        a //= 2
        deviationBoundary *= roughness
    return heightmap
#==================================================================================================================================================================================================

def convertToMesh(heightmap, object_name="HeightmapMesh", height_scale=1.0, water_level_normalized=0.4):
    """
    Convert a heightmap to a mesh object.
    
    Args:
        heightmap (numpy.ndarray): The heightmap matrix.
        object_name (str): Name of the mesh object.
        height_scale (float): Scale for height differences.
        water_level_normalized (float): Normalized water level (0.0 to 1.0).
    """
    min_height = np.min(heightmap)
    max_height = np.max(heightmap)
    height_range = max_height - min_height
    
    absolute_water_level = min_height + water_level_normalized * height_range
    clipped_heightmap = np.maximum(heightmap, absolute_water_level)
    
    verts = []
    faces = []

    #vertices - coordinates
    for i in range(clipped_heightmap.shape[0] - 1):
        for j in range(clipped_heightmap.shape[1] - 1):
            z = (clipped_heightmap[i][j] - min_height) * height_scale
            verts.append((i, j, z))
    
    #faces - at least 3 vertices (4 in our case, since grid...) 
    for i in range(clipped_heightmap.shape[0] - 2):
        for j in range(clipped_heightmap.shape[1] - 2):
            start = i * (clipped_heightmap.shape[1] - 1) + j
            faces.append((start, start + 1, start + clipped_heightmap.shape[1], start + clipped_heightmap.shape[1] - 1))
            

    mesh = bpy.data.meshes.new(name=object_name)
    obj = bpy.data.objects.new(object_name, mesh)
    bpy.context.collection.objects.link(obj) # tldr -> 'show()'

    mesh.from_pydata(verts, [], faces) #second parameter - edges - empty 
    #because they will be fetched with the faces - standard practice
    
    mesh.update()

    bpy.context.view_layer.objects.active = obj # tldr -> allow modifications, such as mode switching
    obj.select_set(True) # tldr -> select with mouse
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.normals_make_consistent(inside=False) #make all faces the right orientation 
    bpy.ops.object.mode_set(mode='OBJECT')

def colorMesh(obj_name):
    """
    Color a mesh based on height to reflect fields and mountain ranges.
    
    Args:
        obj_name (str): Name of the mesh object.
    """
    bpy.ops.object.mode_set(mode='OBJECT')
    obj = bpy.context.scene.objects.get(obj_name) # fetch
    
    if not obj:
        return

    mesh = obj.data

    if not mesh.vertex_colors:
        mesh.vertex_colors.new()
    color_layer = mesh.vertex_colors.active.data #select

    min_height = min(v.co.z for v in mesh.vertices) #vertex.coordinates.[axis]
    max_height = max(v.co.z for v in mesh.vertices)
    height_range = max_height - min_height if max_height - min_height > 0 else 1

    water_color = (5/255, 143/255, 255/255, 1)
    grass_low = (98/255, 217/255, 0, 1)
    grass_high = (41/255, 92/255, 0, 1)
    mountain_low = (82/255, 87/255, 79/255, 1)
    mountain_high = (250/255, 250/255, 250/255, 1)

    def interpolate_color(low_color, high_color, t):
        return tuple(low + (high - low) * t for low, high in zip(low_color, high_color)) # low + delta*t for t 0-1 

    for poly in mesh.polygons:
        for idx in poly.loop_indices:
            loop = mesh.loops[idx]
            vertex = mesh.vertices[loop.vertex_index]
            normalized_height = (vertex.co.z - min_height) / height_range

            if normalized_height <= 0:
                color = water_color
            elif normalized_height <= 0.4:
                t = (normalized_height) / 0.4
                color = interpolate_color(grass_low, grass_high, t)
            elif normalized_height <= 1:
                t = (normalized_height - 0.4) / (0.6)
                color = interpolate_color(mountain_low, mountain_high, t)
            else:
                color = mountain_high

            color_layer[idx].color = color

    mesh.update()

def applyColorAsMaterial(obj_name):
    """
    Apply color to a mesh object as material.
    
    Args:
        obj_name (str): Name of the mesh object.
    """
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        return

    if not obj.data.materials:
        mat = bpy.data.materials.new(name="VertexColorMaterial") #create new material
        mat.use_nodes = True #enable "node based" mode
        
        #https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/principled.html#principled-bsdf
        bsdf = mat.node_tree.nodes.get('Principled BSDF') #fetch BSDF shader (umbrella node for material config)
        if bsdf is not None:
            vertex_color_node = mat.node_tree.nodes.new(type='ShaderNodeVertexColor')
            vertex_color_node.layer_name = "Col"
            mat.node_tree.links.new(bsdf.inputs['Base Color'], vertex_color_node.outputs['Color'])
            obj.data.materials.append(mat)
            #tldr -> if it exists (it should if we colored the mesh), we 'propagate' the colors to material / shader

def subdivideMesh(obj_name, subdivisions=2):
    """
    Subdivide a mesh object.
    
    Args:
        obj_name (str): Name of the mesh object.
        subdivisions (int): Number of subdivisions.
    """
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        return

    # make sure we select ONLY terrain
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    subdiv_modifier = obj.modifiers.new(name="Subdivision", type='SUBSURF')
    subdiv_modifier.levels = subdivisions
    subdiv_modifier.render_levels = subdivisions

    bpy.ops.object.modifier_apply(modifier=subdiv_modifier.name)
    bpy.ops.object.shade_smooth()

def rescaleAndCenter(obj_name, scale_factor=1.0):
    """
    Rescale and center a mesh object.
    
    Args:
        obj_name (str): Name of the mesh object.
        scale_factor (float): Scale variable.
    """
    bpy.ops.object.select_all(action='DESELECT')
    
    obj = bpy.data.objects.get(obj_name)
    
    if obj is None:
        return
    
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)

def placeOnTerrain(tree_name, terrain_name, minRelativeHeight=0.1, maxRelativeHeight=0.4, count=1):
    """
    Place objects on a terrain mesh.
    
    Args:
        tree_name (str): Name of the tree object.
        terrain_name (str): Name of the terrain mesh object.
        minRelativeHeight (float): Minimum relative height for object placement.
        maxRelativeHeight (float): Maximum relative height for object placement.
        count (int): Number of objects to place.
    """
    bpy.ops.object.select_all(action='DESELECT')
    
    terrain = bpy.data.objects.get(terrain_name)
    tree = bpy.data.objects.get(tree_name)

    if not terrain or not tree:
        return

    # '@' (matrix mult) decorator ; 'cast' of local coords to global coords
    z_coords = [terrain.matrix_world @ vertex.co for vertex in terrain.data.vertices]
     
    #tldr -> creating a bounding box around terrain
    min_z, max_z = min(z.z for z in z_coords), max(z.z for z in z_coords)
    absolute_min_height = min_z + (max_z - min_z) * minRelativeHeight
    absolute_max_height = min_z + (max_z - min_z) * maxRelativeHeight

    valid_locations = []

    for _ in range(100 * count):
        #random x,y
        rand_x = random.uniform(min(vert[0] for vert in terrain.bound_box), max(vert[0] for vert in terrain.bound_box))
        rand_y = random.uniform(min(vert[1] for vert in terrain.bound_box), max(vert[1] for vert in terrain.bound_box))
        
        start_point = terrain.matrix_world @ Vector((rand_x, rand_y, 1000))
        end_point = terrain.matrix_world @ Vector((rand_x, rand_y, -1000))
        
        #cast ray through the terrain
        #https://docs.blender.org/api/current/bpy.types.Scene.html#bpy.types.Scene.ray_cast
        hit, location, _, _, hit_obj, _ = bpy.context.scene.ray_cast(bpy.context.view_layer.depsgraph, start_point, end_point)
        
        #if i hit the terrain and the hit coordinates make sense -> note this location
        if hit and hit_obj == terrain and absolute_min_height <= location.z <= absolute_max_height:
            valid_locations.append(location)

        if len(valid_locations) >= count:
            break

    for location in valid_locations:
        tree_copy = tree.copy()
        tree_copy.data = tree.data.copy()
        bpy.context.collection.objects.link(tree_copy)
        tree_copy.location = location
        tree_copy.location.z -= 0.1

#==================================================================================================================================================================================================

object_name = "Terrain"
size = 1 + 2 ** 8
roughness = 0.58
terrain = generate(size, roughness)
convertToMesh(terrain, object_name, 21.0)
colorMesh(object_name)
subdivideMesh(object_name, 1)
applyColorAsMaterial(object_name)
rescaleAndCenter(object_name, 2.0)

placeOnTerrain("LeafTree1", "Terrain", count=300)
placeOnTerrain("LeafTree2", "Terrain", count=250)
placeOnTerrain("LeafTree3", "Terrain", count=150)

placeOnTerrain("Tree1", "Terrain", 0.35, 0.55, 180)
placeOnTerrain("Tree2", "Terrain", 0.35, 0.55, 150)
placeOnTerrain("Tree2", "Terrain", 0.35, 0.55, 130)

#placeOnTerrain("Cabin", "Terrain", 0.15, 0.4,2)
#placeOnTerrain("Watchtower", "Terrain", 0.55, 0.8,8)

