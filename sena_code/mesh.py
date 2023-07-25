import open3d as o3d
import numpy

print("Let's draw some primitives")

mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
mesh_sphere.compute_vertex_normals()
mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[-2, -2, -2])

tris = numpy.asarray(mesh_sphere.adjacency_list)
numpy.savetxt('deneme.txt', tris, fmt='%1.3f', delimiter=' ')

print("We draw a few primitives using collection.")
o3d.visualization.draw_geometries(
    [ mesh_sphere,mesh_frame])

# print("We draw a few primitives using + operator of mesh.")
# o3d.visualization.draw_geometries(
#     [mesh_sphere + mesh_frame])