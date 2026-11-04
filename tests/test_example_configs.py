import pytest

from PyImageTrack.run_pipeline import run_from_config


@pytest.mark.parametrize("config_path",
                          [
                          "configs/00_example_two_images.toml",
                          "configs/01_example_image_series.toml",
                          "configs/02_example_non_georeferenced.toml"])
def test_run_example_configs(config_path: str):
    run_from_config(config_path)