from typing import Type
import open3d as o3d
from plyfile import PlyData, PlyElement
import numpy as np
import os
import mesh_to_sdf
import trimesh


def ply_to_xyzn(path,outpath=None):
    if os.path.isdir(path):
        if not os.path.isdir(outpath):
            os.mkdir(outpath)
        tmppath=os.path.join(path,'tmp')
        if not os.path.isdir(tmppath):
            os.mkdir(tmppath)
        for file in os.listdir(path):
            if not file.endswith('ply'):
                continue
            
            if outpath is None:
                outfile=os.path.join(path,file).replace('ply','xyzn')
            else:
                outfile=os.path.join(outpath,file.replace('ply','xyzn'))

            if os.path.exists(outfile):
                continue
            # open as triangle mesh first
            mesh=o3d.io.read_triangle_mesh(os.path.join(path,file))
            mesh.compute_vertex_normals()
            tmp=os.path.join(path,'tmp',file)
            o3d.io.write_triangle_mesh(tmp,mesh)
            ply=o3d.io.read_point_cloud(tmp)
            o3d.io.write_point_cloud(outfile,ply)


def obj_to_ply(path,outname):
    fn=path
    if not fn.endswith('.obj'):
        raise TypeError()

    mesh=trimesh.load(fn)
    res=mesh_to_sdf.get_surface_point_cloud(mesh, surface_point_method='scan', bounding_radius=1, scan_count=100, scan_resolution=400, sample_point_count=10000000, calculate_normals=True)
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(res.points)
    pcd.normals=o3d.utility.Vector3dVector(res.normals)

    o3d.io.write_point_cloud(outname+'.ply',pcd)
    mesh=o3d.io.read_triangle_mesh(outname+'.ply')
    radii = [10,10,10,10]
    mesh=mesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector(radii))
    #mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(outname+'.ply',mesh)

# Subdividing meshes doesn't enforce points to be evenly distributed - so not gonna work!
def obj_subdivide_to_ply(path,outname):
    fn=path
    if not fn.endswith('obj'):
        raise TypeError()
    mesh=o3d.io.read_triangle_mesh(fn)
    print('{} has {} vertices and {} faces'.format(fn,len(mesh.vertices),len(mesh.triangles)))
    mesh = mesh.subdivide_loop(2)
    print('subdivided-{} has {} vertices and {} faces'.format(fn,len(mesh.vertices),len(mesh.triangles)))
    o3d.io.write_triangle_mesh(outname+'.ply',mesh)


if __name__ == '__main__':
    #obj_to_ply('./data/obj/chair.obj','./data/mesh_to_sdf-chair')
    ply_to_xyzn('../DeepSDF/exp-chairs/TrainingMeshes/2000/ShapeNetV2/03001627','./data/deepsdf')


"""
for i in range(1,5):
    fn='./data/c{}.ply'.format(i)

    mesh=o3d.io.read_triangle_mesh(fn)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(fn,mesh)
    ply=o3d.io.read_point_cloud(fn)
    o3d.io.write_point_cloud(fn.replace('ply','xyzn'),ply)
"""
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