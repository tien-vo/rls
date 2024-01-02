import pytest

from ..magnetized_plasma import MagnetizedPlasmaUnits

units = MagnetizedPlasmaUnits()


@pytest.mark.parametrize("scaling", ["ion", "electron"])
def test_consistency(scaling):
    units.set_scale(scaling)
    units.check_consistency()
