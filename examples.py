import logging

from streetview_dataset import StreetViewDataset

logging.basicConfig(level=logging.INFO)

'''
Progress bar
Currently the program exists even when no files seems to have been downloaded. Then sometimes later the files actually appears. The program should only exit once all the files has been downloaded, i assume this is a threading issue. 
I also do not seem to get error from the threads. Atleast no files ever got downloaded from the test. This makes it quite hard to debug:
This is the test:
sv = StreetViewDataset(
    "datasets/denmark", 
    storage="files",
    workers=4, 
    seed=42
)
result = sv.country("Denmark", count=8, download="flat", width=512, height=512, fov=70)
print(result)
'''

sv = StreetViewDataset(
    "datasets/denmark", 
    storage="files",
    file_sharding=False,
    workers=8, 
    seed=42
)
result = sv.country("Denmark", count=8, download="flat", width=512, height=512, fov=70)
print(result)

# 1) Resolve a specific point.
#pano = sv.nearest(56.1629, 10.2039)
#print(pano)

#if pano:
    # 2) Create a 180-degree panorama around the resolved point.
#    sv.download_view(pano, "datasets/denmark/example.jpg")

# 3) Build/resume a random country dataset.

# 4) Or collect near a point.
# sv.radius(56.1629, 10.2039, radius_km=25, count=1000, download="flat")

'''
from streetview_dataset import StreetViewDataset

sv = StreetViewDataset(
    "datasets/denmark",
    workers=32,
    storage="zip",
    shard_size=2000,
    seed=42,
)

result = sv.country(
    "Denmark",
    count=1_000_000,
    download="flat",
)

print(result)

Another small change. It should also be an option to enable, or not, the folder sharding with storage equal files. I.e. not making the two symbol subfolders.
'''
