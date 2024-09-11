import os

os.system('colmap feature_extractor --database_path /home/zzh/colmap_0910/database.db --image_path /home/zzh/gs-indoor/data/scannet/scene0710_00/train/rgb')
os.system('python spawn_data.py')
os.system('colmap exhaustive_matcher --database_path /home/zzh/colmap_0910/database.db')
os.system('colmap point_triangulator --database_path /home/zzh/colmap_0910/database.db --image_path /home/zzh/gs-indoor/data/scannet/scene0710_00/train/rgb --input_path /home/zzh/colmap_0910/input --output_path /home/zzh/colmap_0910/sparse')