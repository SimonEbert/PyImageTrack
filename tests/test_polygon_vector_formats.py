import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from PyImageTrack.run_pipeline import _read_polygon_vector


@pytest.mark.parametrize("extension", [".geojson", ".gpkg"])
def test_polygon_vector_reader_supports_geojson_and_geopackage(tmp_path, extension):
    polygons = gpd.GeoDataFrame(
        {"moving_id": ["area-1"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:4326",
    )
    path = tmp_path / f"moving_area{extension}"
    polygons.to_file(path, driver="GeoJSON" if extension == ".geojson" else "GPKG")

    loaded = _read_polygon_vector(path)

    assert len(loaded) == 1
    assert loaded.loc[0, "moving_id"] == "area-1"
    assert loaded.crs.to_epsg() == 4326


def test_polygon_vector_reader_rejects_unsupported_formats(tmp_path):
    path = tmp_path / "moving_area.txt"
    path.write_text("not a vector file", encoding="utf-8")

    with pytest.raises(ValueError, match="Supported formats"):
        _read_polygon_vector(path)
