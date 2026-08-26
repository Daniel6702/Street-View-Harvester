from streetview_dataset.storage import FileImageStore


def test_file_image_store_without_sharding_uses_images_directory(tmp_path):
    store = FileImageStore(tmp_path, file_sharding=False)

    stored = store.store("panoid", "view.jpg", b"image")

    assert stored.image_path == "images/view.jpg"
    assert (tmp_path / stored.image_path).read_bytes() == b"image"


def test_file_image_store_sharding_is_enabled_by_default(tmp_path):
    store = FileImageStore(tmp_path)

    stored = store.store("panoid", "view.jpg", b"image")

    assert stored.image_path.startswith("images/")
    assert stored.image_path != "images/view.jpg"
