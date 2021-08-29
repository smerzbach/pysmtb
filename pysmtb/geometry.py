import numpy as np


def spherical_to_cartesian(inclination_radians: np.ndarray, azimuth_radians: np.ndarray, radius: np.ndarray = None):
    """convert spherical to 3D cartesian coordinates

    spherical coordinates have to be specified as separate ndarrays of inclination angle (0 radians in zenith, pi
    radians in nadir) and optionally a radial coordinate (which is otherwise implicitly assumed to be 1 for unit
    vectors);

    input array dimensions are arbitrary but must match between the different arrays;
    returns tuple of cartesian coordinates in separate x, y and z arrays"""

    if radius is None:
        radius = 1

    sin_theta = np.sin(inclination_radians)
    x = radius * sin_theta * np.cos(azimuth_radians)
    y = radius * sin_theta * np.sin(azimuth_radians)
    z = radius * np.cos(inclination_radians)
    return x, y, z


def cartesian_to_spherical(x: np.ndarray, y: np.ndarray, z: np.ndarray, normalized: bool = False, with_radius: bool = False):
    """convert (optionally normalized) 3D cartesian to spherical coordinates

    spherical coordinates follow the inclination-azimuth convention: inclination angle theta goes from 0 radians in the
    zenith (north pole) to pi radians at nadir (south pole), azimuth is circular from 0 to 2 pi radians, with 0 in x
    direction

    inputs are numpy ndarrays with arbitrary but matching dimensions; if normalized == True, inputs are assumed to be of
    unit length; with_radius determines whether only inclination and azimuth, or if also the radial coordinate should be
    returned in case the inputs are not normalized
    returns spherical coordinates in separate arrays: inclination, azimuth [and radius, when with_radius == True]
    """

    if not normalized:
        radius = np.sqrt(x**2 + y**2 + z**2)
        inv_radius = 1 / radius
        x *= inv_radius
        y *= inv_radius
        z *= inv_radius
    else:
        radius = np.ones_like(x)

    inclination = np.arccos(z)
    azimuth = np.arctan2(y, x)

    if with_radius:
        return inclination, azimuth, radius
    else:
        return inclination, azimuth


def cartesian_to_plane_stereographic(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """convert 3D cartesian vectors on the unit sphere to 2D coordinates by transforming as:

    x_2d = x / (1 + z)
    y_2dr = y / (1 + z)

    this corresponds to a stereographic projection from the lower pole (nadir), which maps the upper hemisphere (z > 0)
    inside the unit circle in the x-y-plane

    inputs numpy ndarrays with arbitrary but matching dimensions;
    returns tuple x_par, y_par with same dimensions as inputs"""

    inv_denom = 1. / (1 + z)
    return x * inv_denom, y * inv_denom


def plane_to_cartesian_stereographic(x_2d: np.ndarray, y_2d: np.ndarray):
    """transform 2D coordinates from the x-y-plane to 3D cartesian coordinates on the unit sphere by means of a
    stereographic projection from the lower pole, which maps 2D points from within the x-y unit circle to the upper
    hemisphere

    inputs: x_2d, y_2d: np.ndarrays with arbitrary but matching dimensions
    returns 3-tuple of np.ndarrays x, y, z, where each array has the same shape as the inputs"""

    len_sqr = x_2d ** 2 + y_2d ** 2
    x = 2 * x_2d / (len_sqr + 1)
    y = 2 * y_2d / (len_sqr + 1)
    z = (1 - len_sqr) / (1 + len_sqr)

    return x, y, z
