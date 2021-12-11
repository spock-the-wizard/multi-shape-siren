from os import F_LOCK
import open3d as o3d
from plyfile import PlyData, PlyElement
import numpy as np



for i in range(1,5):
    fn='./data/c{}.ply'.format(i)

    mesh=o3d.io.read_triangle_mesh(fn)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(fn,mesh)
    ply=o3d.io.read_point_cloud(fn)
    o3d.io.write_point_cloud(fn.replace('ply','xyzn'),ply)
"""
fn='./data/fromdeepsdf.ply'

# plydata=PlyData.read(fn)
# vertices=plydata['vertex']
# faces=plydata['face']
# for i in vertices:
#     print(i)

# import pdb;pdb.set_trace()



pply=o3d.io.read_triangle_mesh(fn)
num_tri=len(np.asarray(pply.triangles))
pply.remove_triangles_by_index([i for i in range(num_tri)])

#mesh=pply.compute_vertex_normals()
o3d.io.write_triangle_mesh('./data/tmp/deep.ply',pply)
pply=o3d.io.read_point_cloud('./data/tmp/deep.ply')
o3d.io.write_point_cloud("./data/fromdeepsdf.xyzn",pply)
print(pply)

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