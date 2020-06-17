"""
This script creates voronoi polygons around major metro centers in the US, with
modifications of the NYC and Long Island areas to keep them as distinct IPM regions.

To add additional metro areas for a new region, use the --extra-metro-cbsa-ids flag,
once for each additional cbsa_id to include:

python create_voronoi_polygons.py --extra-metro-cbsa-ids 12100 --extra-metro-cbsa-ids 41540
"""

from typing import List

import pandas as pd
import geopandas as gpd
import shapely.ops
from shapely.ops import cascaded_union
from geovoronoi import voronoi_regions_from_coords
import typer

from site_interconnection_costs import (
    load_ipm_shapefile,
    load_metro_areas_shapefile,
)

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    # More extensive test-like formatter...
    "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s",
    # This is the datetime format string.
    "%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def load_us_outline():
    "Load a gdf of US states and return the outline of lower-48"
    us_states = gpd.read_file(
        "https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_5m.json"
    )
    drop_states = ["Puerto Rico", "Alaska", "Hawaii"]
    us_states = us_states.loc[~(us_states["NAME"].isin(drop_states)), :]

    us_outline = shapely.ops.unary_union(us_states["geometry"])

    return us_outline


def find_largest_cities(
    metro_areas_gdf,
    ipm_gdf,
    min_population=750000,
    max_cities_per_region=None,
    extra_metro_cbsa_ids=[],
):
    _metro_areas_gdf = metro_areas_gdf.copy()
    _metro_areas_gdf["geometry"] = _metro_areas_gdf["center"]
    #     metro_ipm_gdf = gpd.sjoin(ipm_gdf, _metro_areas_gdf, how="left", op="intersects")
    metro_ipm_gdf = gpd.sjoin(ipm_gdf, _metro_areas_gdf, how="left", op="contains")

    df_list = []
    # nw_areas["latitude"] = 0
    # nw_areas["longitude"] = 0
    grouped = metro_ipm_gdf.groupby("IPM_Region", as_index=False)
    for _, _df in grouped:
        #     n_df = _df.nlargest(5, "population")
        n_df = _df.loc[
            (_df["population"] >= min_population)
            | (_df["cbsa_id"].isin(extra_metro_cbsa_ids)),
            :,
        ]
        if max_cities_per_region:
            n_df = n_df.nlargest(max_cities_per_region, "population")
        # If there aren't any city that meet population criteria keep the largest city
        if n_df.empty:
            n_df = _df.nlargest(1, "population")
        df_list.append(n_df)
    largest_cities = pd.concat(df_list, ignore_index=True)

    lats = [center.y for center in largest_cities.center]
    lons = [center.x for center in largest_cities.center]

    largest_cities["latitude"] = lats
    largest_cities["longitude"] = lons

    return largest_cities


def main(fn: str = "large_metro_voronoi.geojson", extra_metro_cbsa_ids: List[str] = []):

    logger.info("Loading files")
    us_outline = load_us_outline()
    ipm_gdf = load_ipm_shapefile()
    ipm_gdf["convex_hull"] = ipm_gdf.convex_hull
    # site_locations = load_site_locations()
    metro_gdf = load_metro_areas_shapefile()

    logger.info("Finding largest metros")
    if extra_metro_cbsa_ids:
        logger.info(f"The extra metros {extra_metro_cbsa_ids} will be included")
    largest_metros = find_largest_cities(
        metro_areas_gdf=metro_gdf,
        ipm_gdf=ipm_gdf,
        min_population=750000,
        extra_metro_cbsa_ids=extra_metro_cbsa_ids,
    )

    logger.info("Making voronoi polygons")
    poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(
        largest_metros[["longitude", "latitude"]].values, us_outline
    )

    metro_voronoi = largest_metros.iloc[[x[0] for x in poly_to_pt_assignments], :]
    metro_voronoi["metro_id"] = metro_voronoi["cbsa_id"]
    metro_voronoi.geometry = poly_shapes

    logger.info("Fixing NYC/Long Island")
    ny_z_j_poly = ipm_gdf.loc[ipm_gdf["IPM_Region"] == "NY_Z_J", "convex_hull"].values[
        0
    ]
    ny_z_k_poly = ipm_gdf.loc[ipm_gdf["IPM_Region"] == "NY_Z_K", "convex_hull"].values[
        0
    ]
    ny_z_j_k_poly = cascaded_union([ny_z_j_poly, ny_z_k_poly])

    for cbsa_id in metro_voronoi.query(
        "IPM_Region.isin(['NENG_CT', 'PJM_EMAC']).values"
    )["cbsa_id"].to_list():
        # print(cbsa_id)
        metro_voronoi.loc[
            metro_voronoi["cbsa_id"] == cbsa_id, "geometry"
        ] = metro_voronoi.loc[
            metro_voronoi["cbsa_id"] == cbsa_id, "geometry"
        ].difference(
            ny_z_j_k_poly
        )

    # Need the unary_union to make geometries valid
    ny_z_j_ipm = shapely.ops.unary_union(
        ipm_gdf.loc[ipm_gdf["IPM_Region"] == "NY_Z_J", "geometry"].values[0]
    )
    ny_z_k_ipm = shapely.ops.unary_union(
        ipm_gdf.loc[ipm_gdf["IPM_Region"] == "NY_Z_K", "geometry"].values[0]
    )

    # Get a simplified outline of Long Island
    # Start with the zone K convex hull, remove the overlap with zone J IPM region,
    # then take the intersection with the US outline.
    ny_z_k_ipm = ny_z_k_poly.difference(ny_z_j_ipm).intersection(us_outline)

    # Same with NYC, zone J. Remove the bordering regions (zone K, other IPM regions)
    # from the convex hull, then take intersection with US outline.
    ny_z_j_ipm = (
        ny_z_j_poly.difference(ny_z_k_ipm)
        .difference(
            shapely.ops.unary_union(
                ipm_gdf.query("IPM_Region=='PJM_EMAC'")["geometry"].values[0]
            )
        )
        .difference(
            shapely.ops.unary_union(
                ipm_gdf.query("IPM_Region=='NY_Z_G-I'")["geometry"].values[0])
            )
        .intersection(us_outline)
    )

    data_dict = {
        "IPM_Region": ["NY_Z_J", "NY_Z_K"],
        "state": ["NY", "NY"],
        "metro_id": ["NY_Z_J", "NY_Z_K"],
        "latitude": [ny_z_j_ipm.centroid.y, ny_z_k_ipm.centroid.y],
        "longitude": [ny_z_j_ipm.centroid.x, ny_z_k_ipm.centroid.x],
    }

    ny_z_j_k_df = gpd.GeoDataFrame(
        data=data_dict, geometry=[ny_z_j_ipm, ny_z_k_ipm], crs=metro_voronoi.crs
    )

    final_metro_voronoi = pd.concat(
        [metro_voronoi, ny_z_j_k_df], ignore_index=True, sort=False
    )

    logger.info("Writing polygons to file")
    cols = ["IPM_Region", "geometry", "latitude", "longitude", "metro_id"]
    final_metro_voronoi[cols].to_file(fn, driver="GeoJSON")


if __name__ == "__main__":
    typer.run(main)
