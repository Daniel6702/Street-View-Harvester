from streetview_dataset import MonitorConfig, StreetViewDataset

sv = StreetViewDataset(
    "datasets/aarhus",
    storage="zip",
    file_sharding=True,
    shard_size=1000,
    workers=4,
    seed=42,
)

result = sv.geojson(
    path="streetview_dataset/data/aarhus_kommune.geojson",
    count=10_000,
    download="panorama",
    monitor=MonitorConfig(on_start=lambda url: print(f"Web progress: {url}")),
)

print(result)
