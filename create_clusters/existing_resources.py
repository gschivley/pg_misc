import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
import shapely.vectorized
from scipy.spatial import cKDTree
import netCDF4
from pathlib import Path
import logging

from powergenome.transmission import haversine

logger = logging.getLogger(__file__)
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

CWD = Path.cwd()
VCE_DATA_PATH = Path("/Volumes/Macintosh HD 1/Updated_Princeton_Data")
VCE_WIND_PATH = VCE_DATA_PATH / "PRINCETON-Wind-Data-2012"
VCE_SOLAR_PATH = VCE_DATA_PATH / "PRINCETON-Solar-Data-2012"
EIA_860M_PATH = CWD / "march_generator2020.xlsx"
EIA_860ER_PATH = CWD / "eia8602019ER"
METRO_VORONOI_PATH = CWD / "large_metro_voronoi.geojson"


def load_vce_grid_points():
    print("Loading VCE grid points")
    df = pd.read_csv(VCE_DATA_PATH / "PRINCETON-MetaData" / "RUC_LatLonSites.csv")
    df["Site"] = df["Site"].astype(str).str.zfill(6)

    return df


def load_us_states_gdf():

    us_states = gpd.read_file(
        "https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_5m.json"
    )
    drop_states = ["Puerto Rico", "Alaska", "Hawaii"]
    us_states = us_states.loc[~(us_states["NAME"].isin(drop_states)), :]
    us_states = us_states.reset_index(drop=True)

    return us_states


def label_site_region(gdf, id_col, lat, lon):

    mask = pd.Series(index=range(len(lat)))

    for n in gdf.index:
        dataGeom = gdf["geometry"][n]
        data_mask = shapely.vectorized.contains(dataGeom, lon, lat)
        mask[data_mask] = gdf[id_col][n]

    return mask


def label_plant_region(plant_locations, model_gdf, id_col):

    mask = np.array([np.nan] * len(plant_locations))

    for idx, row in model_gdf.iterrows():
        region_geom = row["geometry"]
        region_mask = shapely.vectorized.contains(
            region_geom, plant_locations["longitude"], plant_locations["latitude"]
        )
        mask[region_mask] = idx

    plant_locations[id_col] = mask
    region_map = {idx: row[id_col] for idx, row in model_gdf.iterrows()}
    plant_locations[id_col] = plant_locations[id_col].map(region_map)
    plant_locations = plant_locations.dropna()

    return plant_locations


def load_860m():
    logger.info("Loading 860m")
    cols = [
        "Plant Code",
        "State",
        "Generator ID",
        "Technology",
        "Nameplate Capacity (MW)",
        "Single-Axis Tracking?",
        "Dual-Axis Tracking?",
        "Fixed Tilt?",
        "Azimuth Angle",
        "Tilt Angle",
        "DC Net Capacity (MW)",
        "Operating Year",
    ]
    eia860m = pd.read_excel(EIA_860M_PATH, header=1, skip_footer=1)


def load_plant_locations():
    logger.info("Loading plant data")
    cols = ["Plant Code", "Latitude", "Longitude"]
    plants_df = pd.read_excel(
        EIA_860ER_PATH / "2___Plant_y2019_Early_Release.xlsx",
        header=2,
        usecols=cols,
        na_values=[" "],
    )
    plants_df.columns = plants_df.columns.str.lower().str.replace(" ", "_")

    return plants_df


def load_solar_file():
    print("Loading solar data")
    solar_cols = [
        "Plant Code",
        "State",
        "Generator ID",
        "Technology",
        "Prime Mover",
        "Nameplate Capacity (MW)",
        "Single-Axis Tracking?",
        "Dual-Axis Tracking?",
        "Fixed Tilt?",
        "Azimuth Angle",
        "Tilt Angle",
        "DC Net Capacity (MW)",
        "Operating Year",
    ]
    solar_df = pd.read_excel(
        EIA_860ER_PATH / "3_3_Solar_Y2019_Early_Release.xlsx",
        header=2,
        usecols=solar_cols,
        na_values=[" "],
    )
    solar_df.columns = (
        solar_df.columns.str.lower()
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("?", "")
    )

    def tracking_value(row):
        if row["single_axis_tracking"] == "Y":
            return 1
        elif row["dual_axis_tracking"] == "Y":
            return 2
        else:
            return 0

    solar_df["tracking"] = solar_df.apply(lambda row: tracking_value(row), axis=1)
    solar_df["azimuth_angle"].fillna(solar_df["azimuth_angle"].mode(), inplace=True)
    solar_df["tilt_angle"].fillna(solar_df["tilt_angle"].mode(), inplace=True)

    return solar_df


def load_wind_file():
    wind_cols = [
        "Plant Code",
        "State",
        "Generator ID",
        "Nameplate Capacity (MW)",
        "Operating Year",
        "Number of Turbines",
        "Predominant Turbine Manufacturer",
        "Predominant Turbine Model Number",
        "Design Wind Speed (mph)",
        "Wind Quality Class",
        "Turbine Hub Height (Feet)",
    ]
    wind_df = pd.read_excel(
        EIA_860ER_PATH / "3_2_Wind_Y2019_Early_Release.xlsx",
        header=2,
        usecols=wind_cols,
    )
    wind_df.columns = (
        wind_df.columns.str.lower()
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace(" ", "_")
    )
    wind_df["predominant_turbine_manufacturer"] = (
        wind_df["predominant_turbine_manufacturer"]
        .str.replace("-", " ")
        .str.replace("Micon", "NEG Micon")
        .str.replace("NEG NEG", "NEG")
    )
    wind_df["predominant_turbine_model_number"] = (
        wind_df["predominant_turbine_model_number"]
        .str.replace("-", " ")
        .str.replace("/", " ")
        .str.replace("DW", "DirectWind ")
        .str.replace("Vestas ", "")
    )
    wind_df["model_name"] = (
        wind_df["predominant_turbine_manufacturer"]
        + " "
        + wind_df["predominant_turbine_model_number"].astype(str)
    )
    wind_df["turbine_capacity_mw"] = (
        wind_df["nameplate_capacity_mw"] / wind_df["number_of_turbines"]
    )

    return wind_df


def renorm_solar(df):

    df = df * 1.3
    df = np.where(df <= 100, df, 100)

    return df


def ckdnearest(gdA, gdB):
    "https://gis.stackexchange.com/a/301935"
    nA = np.array(list(zip(gdA.Latitude, gdA.Longitude)))
    nB = np.array(list(zip(gdB["latitude"], gdB["longitude"])))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)

    gdB.rename(columns={"latitude": "lat2", "longitude": "lon2"}, inplace=True)

    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB.loc[idx, gdB.columns != "geometry"].reset_index(drop=True),
        ],
        axis=1,
    )
    gdf["dist_mile"] = gdf.apply(
        lambda row: haversine(
            row["Longitude"], row["Latitude"], row["lon2"], row["lat2"], units="mile"
        ),
        axis=1,
    )

    return gdf


SOLAR_PROFILE_DICT = {
    0: "Fixed_SolarPV_Lat",
    1: "Axis1_SolarPV_Lat",
    2: "Axis1_SolarPV_Lat",
}


def build_single_solar_profile(row):

    variable = SOLAR_PROFILE_DICT[row["tracking"]]

    fpath = f"Site_{row['Site']}_SolarPV.nc4"
    site_data = netCDF4.Dataset(VCE_SOLAR_PATH / fpath)
    gen_profile = np.array(site_data[variable])
    gen_profile = renorm_solar(gen_profile)
    plant_code = row["plant_code"]
    generator_id = row["generator_id"]
    plant_gen_id = f"{plant_code}_{generator_id}"

    return {plant_gen_id: gen_profile}


def build_all_solar_profiles(plant_locations=None, region_gdf=None):
    logger.info(f"Starting to build all solar profiles")
    solar_df = load_solar_file()

    site_locations = load_vce_grid_points()

    if plant_locations is None:
        plant_locations = load_plant_locations()

    solar_df = solar_df.loc[
        solar_df["plant_code"].isin(plant_locations["plant_code"]), :
    ]

    nearest_sites = ckdnearest(
        plant_locations.rename(
            columns={"latitude": "Latitude", "longitude": "Longitude"}
        ),
        site_locations.rename(
            columns={"Latitude": "latitude", "Longitude": "longitude"}
        ),
    )

    solar_df["Site"] = solar_df["plant_code"].map(
        nearest_sites.set_index("plant_code")["Site"]
    )
    solar_df["metro_id"] = solar_df["plant_code"].map(
        nearest_sites.set_index("plant_code")["metro_id"]
    )
    solar_df["IPM_Region"] = solar_df["plant_code"].map(
        nearest_sites.set_index("plant_code")["IPM_Region"]
    )

    existing_data_fn = Path("existing_solarpv_profiles.parquet")
    if existing_data_fn.exists():
        solar_profiles_df = pd.read_parquet(existing_data_fn).T
    else:
        solar_profiles = []
        for idx, row in solar_df.iterrows():
            solar_profiles.append(build_single_solar_profile(row))

        solar_profiles_dict = {k: v for d in solar_profiles for k, v in d.items()}
        solar_profiles_df = pd.DataFrame(solar_profiles_dict)
        solar_profiles_df.to_parquet(existing_data_fn)
        solar_profiles_df = solar_profiles_df.T

    solar_df["plant_gen_id"] = (
        solar_df["plant_code"].astype(str) + "_" + solar_df["generator_id"]
    )
    solar_df = solar_df.set_index("plant_gen_id")
    logger.info("Done building all solar profiles")
    return solar_profiles_df, solar_df


def region_weighted_avg_profile(df, plant_meta, region_col="IPM_Region"):

    meta_grouped = plant_meta.dropna(subset=[region_col]).groupby(region_col)
    region_wm = {}
    for _, _meta in meta_grouped:

        region_wm[_] = np.average(
            df.loc[_meta.index, :], weights=_meta["nameplate_capacity_mw"], axis=0
        )

    results = pd.DataFrame(region_wm)
    return results


def build_regional_profiles(plant_profiles, plant_meta, region_col="IPM_Region"):

    regional_weighted_profiles = region_weighted_avg_profile(
        plant_profiles, plant_meta, region_col=region_col
    )
    regional_weighted_profiles = (regional_weighted_profiles / 100).round(4)

    return regional_weighted_profiles


def build_single_wind_profile(row, variable="Wind_Power_80m"):

    fpath = f"Site_{row['Site']}_Wind.nc4"
    site_data = netCDF4.Dataset(VCE_WIND_PATH / fpath)
    gen_profile = np.array(site_data[variable])
    plant_code = row["plant_code"]
    generator_id = row["generator_id"]
    plant_gen_id = f"{plant_code}_{generator_id}"

    return {plant_gen_id: gen_profile}


def build_all_wind_profiles(plant_locations=None, region_gdf=None):
    logger.info(f"Starting to build all wind profiles")

    wind_df = load_wind_file()

    site_locations = load_vce_grid_points()

    if plant_locations is None:
        plant_locations = load_plant_locations()

    wind_df = wind_df.loc[wind_df["plant_code"].isin(plant_locations["plant_code"]), :]

    nearest_sites = ckdnearest(
        plant_locations.rename(
            columns={"latitude": "Latitude", "longitude": "Longitude"}
        ),
        site_locations.rename(
            columns={"Latitude": "latitude", "Longitude": "longitude"}
        ),
    )

    wind_df["Site"] = wind_df["plant_code"].map(
        nearest_sites.set_index("plant_code")["Site"]
    )
    wind_df["metro_id"] = wind_df["plant_code"].map(
        nearest_sites.set_index("plant_code")["metro_id"]
    )
    wind_df["IPM_Region"] = wind_df["plant_code"].map(
        nearest_sites.set_index("plant_code")["IPM_Region"]
    )

    existing_data_fn = Path("existing_wind_profiles.parquet")
    if existing_data_fn.exists():
        wind_profiles_df = pd.read_parquet(existing_data_fn).T
    else:
        wind_profiles = []
        for idx, row in wind_df.iterrows():
            wind_profiles.append(
                build_single_wind_profile(row, variable="Wind_Power_80m")
            )

        wind_profiles_dict = {k: v for d in wind_profiles for k, v in d.items()}
        wind_profiles_df = pd.DataFrame(wind_profiles_dict)
        wind_profiles_df.to_parquet(existing_data_fn)
        wind_profiles_df = wind_profiles_df.T

    wind_df["plant_gen_id"] = (
        wind_df["plant_code"].astype(str) + "_" + wind_df["generator_id"]
    )
    wind_df = wind_df.set_index("plant_gen_id")
    logger.info("Done building all wind profiles")

    return wind_profiles_df, wind_df


def make_regional_metadata(plant_meta, region_col="IPM_Region"):

    region_meta = plant_meta.groupby(region_col).agg(
        {"nameplate_capacity_mw": "sum", "plant_code": "count"}
    )
    region_meta = region_meta.rename(columns={"plant_code": "plant_count"})

    return region_meta


def main():
    output_path = CWD.parent / "existing_renewables"
    output_path.mkdir(exist_ok=True)
    region_gdf = gpd.read_file(METRO_VORONOI_PATH)

    plant_locations = load_plant_locations()
    plant_locations["metro_id"] = label_site_region(
        gdf=region_gdf,
        id_col="metro_id",
        lat=plant_locations.latitude,
        lon=plant_locations.longitude,
    )
    plant_locations["IPM_Region"] = plant_locations["metro_id"].map(
        region_gdf.set_index("metro_id")["IPM_Region"]
    )

    logger.info("Building wind/solarpv profiles")
    plant_wind_profiles, plant_wind_meta = build_all_wind_profiles(
        plant_locations, region_gdf
    )
    plant_solar_profiles, plant_solar_meta = build_all_solar_profiles(
        plant_locations, region_gdf
    )

    region_wind_meta = make_regional_metadata(plant_wind_meta)
    region_solar_meta = make_regional_metadata(plant_solar_meta)

    region_wind_meta.to_csv(output_path / "regional_existing_wind_metadata.csv")
    region_solar_meta.to_csv(output_path / "regional_existing_solarpv_metadata.csv")

    region_wind_profiles = build_regional_profiles(
        plant_wind_profiles, plant_wind_meta, region_col="IPM_Region"
    )
    region_solar_profiles = build_regional_profiles(
        plant_solar_profiles, plant_solar_meta, region_col="IPM_Region"
    )

    region_wind_profiles.to_csv(
        output_path / "regional_existing_wind_profiles.csv", index=False
    )
    region_solar_profiles.to_csv(
        output_path / "regional_existing_solarpv_profiles.csv", index=False
    )


if __name__ == "__main__":
    main()
