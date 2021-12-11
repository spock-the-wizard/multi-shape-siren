import open3d as o3d

fn='./data/pointcloud.ply'
pply=o3d.io.read_triangle_mesh(fn)
mesh=pply.compute_vertex_normals()
o3d.io.write_triangle_mesh('./data/tmp/cactus.ply',mesh)
pply=o3d.io.read_point_cloud('./data/tmp/cactus.ply')
o3d.io.write_point_cloud("./data/cactus.xyzn",pply)
print(pply)
"""
fn='./data/obj/chair.obj'
instance=fn.split('/')[-1].split('.')[0]
mesh=o3d.io.read_triangle_mesh(fn)
mesh.scale(100.0, [0,0,0])
mesh = mesh.normalize_normals()
ply=o3d.io.write_triangle_mesh("./data/tmp/{}.ply".format(instance),mesh)
pply=o3d.io.read_point_cloud("./data/tmp/{}.ply".format(instance))
o3d.io.write_point_cloud("./data/{}.xyzn".format(instance),pply)
print(mesh)
"""