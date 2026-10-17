from yolo.calib import fit_ground_plane
import pytest


def test_ground_plane_maps_image_points_to_world_points() -> None:
    plane = fit_ground_plane(
        [(0, 0), (10, 0), (10, 20), (0, 20)],
        [(0, 0), (2, 0), (2, 4), (0, 4)],
    )
    assert plane.project((5, 10)) == pytest.approx((1.0, 2.0))
