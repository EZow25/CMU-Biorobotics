import open3d as o3d

mesh = o3d.io.read_triangle_mesh("bend_tube_low_poly.stl")
mesh = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True, mesh_show_back_face=True)

# Plot each frame, then put them together in a gif
# Just Google stl files, or make them yourself with blender