from streetview_dataset import StreetViewDataset

WORKFLOW = "geojson"
DOWNLOAD_MODE = "flat"

DATASET_ROOT = "datasets/denmark"
STORAGE = "files"
FILE_SHARDING = False
WORKERS = 8
SEED = 42

LAT = 56.1629
LON = 10.2039
COUNTRY = "Denmark"
COUNT = 3
RADIUS_KM = 10
BBOX_WEST = 9.0
BBOX_SOUTH = 55.0
BBOX_EAST = 11.0
BBOX_NORTH = 57.0

SINGLE_VIEW_PATH = "datasets/denmark/view.jpg"
HALF_PANORAMA_PATH = "datasets/denmark/half.jpg"
FULL_PANORAMA_PATH = "datasets/denmark/full.jpg"
VIEW_YAW = 45


def main() -> None:
    workflows = {
        "nearest",
        "single_view",
        "country",
        "radius",
        "bbox",
        "half_panorama",
        "panorama",
        "geojson"
    }
    if WORKFLOW not in workflows:
        raise ValueError(f"Unknown workflow: {WORKFLOW}")

    sv = StreetViewDataset(
        DATASET_ROOT,
        storage=STORAGE,
        file_sharding=FILE_SHARDING,
        workers=WORKERS,
        seed=SEED,
    )

    if WORKFLOW == "nearest":
        print(sv.nearest(LAT, LON))
    elif WORKFLOW == "country":
        print(sv.country(COUNTRY, count=COUNT, download=DOWNLOAD_MODE))
    elif WORKFLOW == "radius":
        print(
            sv.radius(
                lat=LAT,
                lon=LON,
                radius_km=RADIUS_KM,
                count=COUNT,
                download=DOWNLOAD_MODE,
            )
        )
    elif WORKFLOW == "bbox":
        print(
            sv.bbox(
                west=BBOX_WEST,
                south=BBOX_SOUTH,
                east=BBOX_EAST,
                north=BBOX_NORTH,
                count=COUNT,
                download=DOWNLOAD_MODE,
            )
        )
    elif WORKFLOW == "geojson":
        print(
            sv.geojson("streetview_dataset/data/aarhus_kommune.geojson",
                       count=COUNT,
                       download=DOWNLOAD_MODE
            )
        )
    else:
        pano = sv.nearest(LAT, LON)
        if pano is None:
            print("No panorama found.")
            return
        if WORKFLOW == "single_view":
            print(sv.download_view(pano, SINGLE_VIEW_PATH, yaw=VIEW_YAW))
        elif WORKFLOW == "half_panorama":
            print(
                sv.download_half_panorama(
                    pano,
                    HALF_PANORAMA_PATH,
                    center_yaw=VIEW_YAW,
                )
            )
        elif WORKFLOW == "panorama":
            print(sv.download_panorama(pano,
                                       "detailed.jpg",
                                       span=360,
                                       fov=45,
                                       vertical_span=90,
                                       pitch_overlap=0.30,
                                       view_width=1024,
                                       view_height=1024,
                                       ))

if __name__ == "__main__":
    main()
