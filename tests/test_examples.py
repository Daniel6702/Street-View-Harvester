import importlib.util
import sys
from pathlib import Path

import pytest

from streetview_dataset import Panorama

EXAMPLES_PATH = Path(__file__).resolve().parents[1] / "examples.py"


def test_importing_examples_does_not_construct_dataset(monkeypatch):
    constructors = []
    calls = []

    class RecordingDataset:
        def __init__(self, *_args, **_kwargs):
            constructors.append(True)

        def radius(self, *_args, **_kwargs):
            calls.append(True)

    monkeypatch.setattr("streetview_dataset.StreetViewDataset", RecordingDataset)

    # Given: examples.py is loaded without the __main__ execution path.
    # When: the module is imported.
    _import_examples(monkeypatch, "examples_without_main")

    # Then: no dataset construction or network-capable method is attempted.
    assert constructors == []
    assert calls == []


def _import_examples(monkeypatch, name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _load_examples(monkeypatch):
    calls = []

    class RecordingDataset:
        def __init__(self, *args, **kwargs):
            pass

        def _record(self, name, *args, **kwargs):
            calls.append((name, args, kwargs))

        def nearest(self, *args, **kwargs):
            self._record("nearest", *args, **kwargs)
            return Panorama("pano-under-test", 56.1629, 10.2039)

        def country(self, *args, **kwargs):
            self._record("country", *args, **kwargs)

        def radius(self, *args, **kwargs):
            self._record("radius", *args, **kwargs)

        def bbox(self, *args, **kwargs):
            self._record("bbox", *args, **kwargs)

        def download_view(self, *args, **kwargs):
            self._record("download_view", *args, **kwargs)

        def download_half_panorama(self, *args, **kwargs):
            self._record("download_half_panorama", *args, **kwargs)

        def download_panorama(self, *args, **kwargs):
            self._record("download_panorama", *args, **kwargs)

    monkeypatch.setattr("streetview_dataset.StreetViewDataset", RecordingDataset)
    module = _import_examples(monkeypatch, "examples_under_test")
    calls.clear()
    return module, calls


def test_default_radius_workflow_uses_configured_constants(monkeypatch):
    module, calls = _load_examples(monkeypatch)
    # Given: the default workflow is selected and constants are configured.
    module.COUNT = 3
    module.DOWNLOAD_MODE = "none"

    # When: the examples entry point is run.
    module.main()

    # Then: radius receives the configured values.
    assert module.WORKFLOW == "radius"
    assert calls == [
        (
            "radius",
            (),
            {
                "lat": 56.1629,
                "lon": 10.2039,
                "radius_km": 10,
                "count": 3,
                "download": "none",
            },
        )
    ]


def test_bbox_workflow_dispatches_with_configured_count_and_download(monkeypatch):
    module, calls = _load_examples(monkeypatch)
    # Given: bbox is selected with configured collection constants.
    module.WORKFLOW = "bbox"
    module.COUNT = 4
    module.DOWNLOAD_MODE = "flat"

    # When: the examples entry point is run.
    module.main()

    # Then: bbox receives its configured bounds, count, and download mode.
    assert calls == [
        (
            "bbox",
            (),
            {
                "west": 9.0,
                "south": 55.0,
                "east": 11.0,
                "north": 57.0,
                "count": 4,
                "download": "flat",
            },
        )
    ]


@pytest.mark.parametrize(
    "point_workflow",
    [
        ("single_view", "download_view", "SINGLE_VIEW_PATH"),
        ("half_panorama", "download_half_panorama", "HALF_PANORAMA_PATH"),
        ("full_panorama", "download_panorama", "FULL_PANORAMA_PATH"),
    ],
)
def test_point_workflows_resolve_a_panorama_before_downloading(
    monkeypatch, point_workflow
):
    workflow, method, path_constant = point_workflow
    module, calls = _load_examples(monkeypatch)
    # Given: one point workflow is selected.
    module.WORKFLOW = workflow

    # When: the examples entry point is run.
    module.main()

    # Then: nearest resolves the Panorama passed to the selected downloader.
    assert [call[0] for call in calls] == ["nearest", method]
    assert calls[0][1] == (56.1629, 10.2039)
    assert isinstance(calls[1][1][0], Panorama)
    assert calls[1][1][1] == getattr(module, path_constant)


def test_nearest_workflow_resolves_the_configured_point(monkeypatch):
    module, calls = _load_examples(monkeypatch)
    # Given: nearest is selected.
    module.WORKFLOW = "nearest"

    # When: the examples entry point is run.
    module.main()

    # Then: only the nearest lookup is dispatched.
    assert calls == [("nearest", (56.1629, 10.2039), {})]


def test_country_workflow_dispatches_to_country_collection(monkeypatch):
    module, calls = _load_examples(monkeypatch)
    # Given: country collection is selected with configured constants.
    module.WORKFLOW = "country"
    module.COUNT = 2
    module.DOWNLOAD_MODE = "none"

    # When: the examples entry point is run.
    module.main()

    # Then: country receives the configured count and download mode.
    assert calls == [("country", ("Denmark",), {"count": 2, "download": "none"})]


def test_invalid_workflow_raises_value_error(monkeypatch):
    module, calls = _load_examples(monkeypatch)
    # Given: an unsupported workflow is selected.
    module.WORKFLOW = "not-a-workflow"

    # When: the examples entry point is run.
    with pytest.raises(ValueError):
        module.main()

    # Then: no dataset operation is attempted.
    assert calls == []
