import numpy as np
import json
import os
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

path = '/home/zzh/gs-indoor/data/scannet/scene0710_00/transforms_train.json'
j = open(path, 'r')
infos = json.load(j)
frames = infos['frames']
focal = frames[0]['fx']
width = 624
height = 468
focal_length = 579.97802734375
out = open('./input/images.txt', 'w')
out_str = ''
# out.write(out_str)

from database import COLMAPDatabase
db = COLMAPDatabase('/home/zzh/colmap_0910/database.db')
res = db.execute(
'''
SELECT *
FROM images
'''
)
res = [t[1] for t in res.fetchall()]

cam_id = len(res) + 1
# cam_id = 1
for i, name in enumerate(res, 0):
    img_id = i + 1
    frame = [f for f in frames if os.path.basename(f['file_path']) == name][0]
    trans = np.array(frame['transform_matrix'])
    R = trans[:3, :3]
    T = trans[:3, 3]
    quen = rotmat2qvec(R)
    file_name = name
    out_str += f'{img_id} {quen[0]} {quen[1]} {quen[2]} {quen[3]} {T[0]} {T[1]} {T[2]} {cam_id} {file_name}\n\n'
out.write(out_str)
out.close()

width = 624
height = 468
focal_length = 579.97802734375
cx = 312
cy = 234
db.add_camera(0, 624, 468, np.array((focal_length, cx, cy)), camera_id=cam_id)

# res = db.execute(
# '''
# SELECT *
# FROM cameras
# '''
# )
# print(res.fetchall())