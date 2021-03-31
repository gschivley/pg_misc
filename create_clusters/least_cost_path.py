from typing import Dict, List, NamedTuple, Union
import numpy as np
import rasterio
import rasterio.features
from affine import Affine
from shapely.geometry import shape, box
from skimage.graph import MCP_Geometric
from skimage.graph import _mcp
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from IPython import embed as IP
import pickle


def convert_coordinates(
    points: gpd.GeoDataFrame, dataset_avoid: rasterio.io.DatasetReader
) -> gpd.GeoDataFrame:

    # transform dataset
    pixel_width, _, originX, _, pixel_height, originY, *args = dataset_avoid.transform

    print("**Converting the X coordinates**")
    # Convert start and end to coordinate pixel
    points["X"] = points.geometry.x
    points["xcoord"] = xcoord2pixelOffset(points.X, originX, pixel_width).round(0)

    print("**Converting the Y coordinates**")
    points["Y"] = points.geometry.y
    points["ycoord"] = ycoord2pixelOffset(points.Y, originY, pixel_height).round(0)

    return points


def xcoord2pixelOffset(x, originX, pixelWidth):

    return (x - originX) / pixelWidth


def ycoord2pixelOffset(y, originY, pixelHeight):

    return (y - originY) / pixelHeight


def create_offsets():

    print("**Creating offsets**")
    # Create tuples listing 8 directions in terms of -1,0,1
    offsets = _mcp.make_offsets(2, True)
    # Destinations are marked with -1 in traceback, so add (0, 0) to the end of offsets
    offsets.append(np.array([0, 0]))
    offsets_arr = np.array(offsets)

    # Calculate distance for moving in each direction. up, down or sideways = 1 and diagonal = sqrt(2)
    tb_distance_dict = {
        idx: (1 if any(a == 0) else np.sqrt(2)) for idx, a in enumerate(offsets_arr)
    }
    tb_distance_dict[-1] = 1  # need to include the -1 destination values

    return offsets_arr, tb_distance_dict


def calc_path_cost(
    cost_surface_list: List[np.ndarray],
    # cost_surface2: np.ndarray,
    traceback_arr: np.ndarray,
    tb_dist_dict: dict,
    route: List[tuple],
    pixel_size: Union[int, float] = 1,
):

    """Determine the cost of a pre-calculated path using the cost surface accounting
    for travel on the diagonal.

    Parameters
    ----------
    cost_surface_list : List[np.ndarray]
            List of NxM arrays of cost values for each pixel
    traceback_arr : np.ndarray
            NxM array indicating the adjacent pixel with lowest cumulative cost. Points
            in the least-cost direction to the nearest destination point. Destination
            points have a value of -1, 0-7 are for adjacent pixels.
    tb_dist_dict : dict
            Keys are -1 through 7, corresponding to values in the `traceback_arr`. Values
            indicate unit path distance to the next pixel (1 or sqrt(2)).
    route : List[tuple]
            A list of coordinates in the `cost_surface` array that the path travels from
            start to end.
    pixel_size : Union[int, float]
            Assuming a square pixel, this is the height or width, and it is used to scale
            the cost per pixel to total path cost. Default value is 1 (assumes costs are
            already scaled).

    Returns
    -------
    float
            Total path cost
    """
    c = 0
    _p = None
    for p in route:
        c += sum(cs[p] for cs in cost_surface_list) * tb_dist_dict[traceback_arr[p]] / 2
        if _p:
            c += (
                sum(cs[_p] for cs in cost_surface_list)
                * tb_dist_dict[traceback_arr[_p]]
                / 2
            )
        _p = p

    return c * pixel_size


class CostRoute(NamedTuple):
    cost: float
    route: List[tuple]
    dest_id: Union[str, int]


def cost_function(
    dataset_avoid: rasterio.io.DatasetReader,
    actual_cost_surface_list: List[np.ndarray],
    start_points: gpd.GeoDataFrame,
    start_point_id_col: str,
    end_points: gpd.GeoDataFrame,
    end_point_id_col: str,
) -> Dict[Union[int, str], CostRoute]:

    cost_surface_avoid = dataset_avoid.read(1)

    # transform the dataset
    pixel_width, _, originX, _, pixel_height, originY, *args = dataset_avoid.transform

    # Create offsets
    offsets_arr, tb_distance_dict = create_offsets()

    print("**Initialize MCP_G**")
    # Initialize MCP_G to get the cost and path from each pixel on the rastor image to the start points
    mcp_g = MCP_Geometric(cost_surface_avoid)
    costs, traceback = mcp_g.find_costs(
        zip(end_points.ycoord.values, end_points.xcoord.values)
    )
    end_point_lookup = {}
    for idx, row in end_points.iterrows():
        end_point_lookup[(row["ycoord"], row["xcoord"])] = row[end_point_id_col]

    print("**Saving the list of routes, their actual cost, and **")
    # Save route list with association to CPA (and substations to MSA)
    cost_route = {}
    for idx, row in start_points.iterrows():
        # Use traceback to create a route ( it shows cell locations in (x,y))
        _route = mcp_g.traceback((row["ycoord"], row["xcoord"]))
        end_point_coords = _route[0]

        # Calculate cost of the route
        cost = calc_path_cost(
            actual_cost_surface_list, traceback, tb_distance_dict, _route, pixel_width
        )
        cost_route[row[start_point_id_col]] = CostRoute(
            cost, _route, end_point_lookup[end_point_coords]
        )

    return cost_route
