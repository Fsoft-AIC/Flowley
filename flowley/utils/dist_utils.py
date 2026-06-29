import os


local_rank = int(os.getenv('LOCAL_RANK', 0))
world_size = int(os.getenv('WORLD_SIZE', 1))
