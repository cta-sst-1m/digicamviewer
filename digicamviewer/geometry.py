from ctapipe.instrument.camera import CameraGeometry
from ctapipe.instrument.camera import _find_neighbor_pixels
import numpy as np
import astropy.units as u


def generate_geometry(camera):
    """
    Generate the SST-1M geometry from the CTS configuration
    :param cts: a CTS instance
    :param available_board:  which board per sector are available (dict)
    :return: the geometry for visualisation and the list of "good" pixels
    """
    pix_x = []
    pix_y = []
    pix_id = []

    for pix in camera.Pixels:
        pix_x.append(pix.center[0])
        pix_y.append(pix.center[1])
        pix_id.append(pix.ID)

    neighbors_pix = _find_neighbor_pixels(pix_x, pix_y, 30.)
    geom = CameraGeometry(cam_id=0, pix_id=pix_id, pix_x=pix_x * u.mm, pix_y=pix_y * u.mm,
                          pix_area=np.ones(1296) * 482.41, neighbors=neighbors_pix, pix_type='hexagonal')

    return geom, pix_id

