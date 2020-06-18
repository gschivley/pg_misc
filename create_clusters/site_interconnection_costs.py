import numpy as np
import netCDF4
import pandas as pd
import geopandas as gpd
import shapely.vectorized
from scipy.spatial import cKDTree
from pathlib import Path

# from joblib import Parallel, delayed
import typer
import shapely

# from pyproj import crs
# import pyproj

from powergenome.params import DATA_PATHS, IPM_SHAPEFILE_PATH, IPM_GEOJSON_PATH
from powergenome.transmission import haversine
from powergenome.nrelatb import investment_cost_calculator, fetch_atb_costs
from powergenome.util import reverse_dict_of_lists, init_pudl_connection
from powergenome.price_adjustment import inflation_price_adjustment

CWD = Path.cwd()
VCE_DATA_PATH = Path("/Volumes/Macintosh HD 1/Updated_Princeton_Data")
VCE_WIND_PATH = VCE_DATA_PATH / "PRINCETON-Wind-Data-2012"
VCE_SOLAR_PATH = VCE_DATA_PATH / "PRINCETON-Solar-Data-2012"

ATB_USD_YEAR = 2017

pudl_engine, pudl_out = init_pudl_connection()

cost_multiplier_region_map = {
    "TRE": ["ERC_PHDL", "ERC_REST", "ERC_WEST"],
    "FRCC": ["FRCC"],
    "MISW": ["MIS_WUMS", "MIS_MNWI", "MIS_IA"],
    "MISE": ["MIS_LMI"],
    "PJMC": ["PJM_COMD"],
    "MISC": ["MIS_IL", "MIS_MO", "S_D_AECI", "MIS_INKY"],
    "SPPN": ["MIS_MAPP", "SPP_WAUE", "SPP_NEBR", "MIS_MIDA"],
    "SPPC": ["SPP_N"],
    "SPPS": ["SPP_WEST", "SPP_SPS"],
    "MISS": ["MIS_AMSO", "MIS_WOTA", "MIS_LA", "MIS_AR", "MIS_D_MS"],
    "SRSE": ["S_SOU"],
    "SRCA": ["S_VACA"],
    "PJMD": ["PJM_Dom"],
    "PJMW": ["PJM_West", "PJM_AP", "PJM_ATSI"],
    "PJME": ["PJM_WMAC", "PJM_EMAC", "PJM_SMAC", "PJM_PENE"],
    "SRCE": ["S_C_TVA", "S_C_KY"],
    "NYUP": ["NY_Z_A", "NY_Z_B", "NY_Z_C&E", "NY_Z_D", "NY_Z_F", "NY_Z_G-I",],
    "NYCW": ["NY_Z_J", "NY_Z_K"],
    "ISNE": ["NENG_ME", "NENGREST", "NENG_CT"],
    "RMRG": ["WECC_CO"],
    "BASN": ["WECC_ID", "WECC_WY", "WECC_UT", "WECC_NNV"],
    "NWPP": ["WECC_PNW", "WECC_MT"],
    "CANO": ["WEC_CALN", "WEC_BANC"],
    "CASO": ["WECC_IID", "WECC_SCE", "WEC_LADW", "WEC_SDGE"],
    "SRSG": ["WECC_AZ", "WECC_NM", "WECC_SNV"],
}
rev_cost_mult_region_map = reverse_dict_of_lists(cost_multiplier_region_map)

tx_capex_region_map = {
    "wecc": [
        "WECC_AZ",
        "WECC_CO",
        "WECC_ID",
        "WECC_MT",
        "WECC_NM",
        "WECC_NNV",
        "WECC_PNW",
        "WECC_SNV",
        "WECC_UT",
        "WECC_WY",
    ],
    "ca": ["WEC_BANC", "WEC_CALN", "WEC_LADW", "WEC_SDGE", "WECC_IID", "WECC_SCE",],
    "tx": ["ERC_PHDL", "ERC_REST", "ERC_WEST",],
    "upper_midwest": [
        "MIS_MAPP",
        "SPP_WAUE",
        "MIS_MNWI",
        "MIS_MIDA",
        "MIS_IA",
        "MIS_IL",
        "MIS_INKY",
    ],
    "lower_midwest": ["SPP_N", "SPP_WEST", "SPP_SPS", "SPP_NEBR",],
    "miso_s": [
        "MIS_LA",
        "MIS_WOTA",
        "MIS_AMSO",
        "MIS_AR",
        "MIS_MO",
        "S_D_AECI",
        "MIS_D_MS",
    ],
    "great_lakes": ["MIS_WUMS", "MIS_LMI",],
    "pjm_s": ["PJM_AP", "PJM_ATSI", "PJM_COMD", "PJM_Dom", "PJM_West", "S_C_KY",],
    "pj_pa": ["PJM_PENE", "PJM_WMAC",],
    "pjm_md_nj": ["PJM_EMAC", "PJM_SMAC"],
    "ny": ["NY_Z_A", "NY_Z_B", "NY_Z_C&E", "NY_Z_D", "NY_Z_F", "NY_Z_G-I",],
    "tva": ["S_C_TVA",],
    "south": ["S_SOU",],
    "fl": ["FRCC"],
    "vaca": ["S_VACA"],
    "ne": ["NY_Z_J", "NY_Z_K", "NENG_CT", "NENG_ME", "NENGREST",],
}

rev_region_mapping = reverse_dict_of_lists(tx_capex_region_map)

spur_costs_2013 = {
    "wecc": 3900,
    "ca": 5200,
    "tx": 3900,
    "upper_midwest": 3900,
    "lower_midwest": 3800,
    "miso_s": 5200,
    "great_lakes": 4100,
    "pjm_s": 5200,
    "pj_pa": 5200,
    "pjm_md_nj": 5200,
    "ny": 5200,
    "tva": 3800,
    "south": 4950,
    "fl": 4100,
    "vaca": 3800,
    "ne": 5200,
}

spur_costs_2017 = {
    region: inflation_price_adjustment(cost, 2013, 2017)
    for region, cost in spur_costs_2013.items()
}

tx_costs_2013 = {
    "wecc": 1350,
    "ca": 2750,
    "tx": 1350,
    "upper_midwest": 900,
    "lower_midwest": 900,
    "miso_s": 1750,
    "great_lakes": 1050,
    "pjm_s": 1350,
    "pj_pa": 1750,
    "pjm_md_nj": 3500,
    "ny": 3500,
    "tva": 1050,
    "south": 1350,
    "fl": 1350,
    "vaca": 900,
    "ne": 3500,
}

tx_costs_2017 = {
    region: inflation_price_adjustment(cost, 2013, 2017)
    for region, cost in tx_costs_2013.items()
}

spur_line_wacc = 0.069
spur_line_investment_years = 60


def load_atb_capex_wacc():
    settings = {
        "atb_cap_recovery_years": 20,
        "atb_financial_case": "Market",
        "atb_cost_case": "Mid",
        "atb_usd_year": 2017,
        "target_usd_year": 2019,
        "pv_ac_dc_ratio": 1.3,
        "cost_multiplier_region_map": cost_multiplier_region_map,
    }

    atb_costs = fetch_atb_costs(pudl_engine, settings)

    solarpv_2030_capex = atb_costs.query(
        "technology=='UtilityPV' & cost_case=='Mid'"
        " & financial_case=='Market' & basis_year==2030  & tech_detail=='LosAngeles'"
    )["capex"].values[0]

    wind_2030_capex = atb_costs.query(
        "technology=='LandbasedWind' & cost_case=='Mid'"
        " & financial_case=='Market' & basis_year==2030  & tech_detail=='LTRG1'"
    )["capex"].values[0]

    solarpv_2030_wacc = atb_costs.query(
        "technology=='UtilityPV' & cost_case=='Mid'"
        " & financial_case=='Market' & basis_year==2030  & tech_detail=='LosAngeles'"
    )["waccnomtech"].values[0]

    wind_2030_wacc = atb_costs.query(
        "technology=='LandbasedWind' & cost_case=='Mid'"
        " & financial_case=='Market' & basis_year==2030  & tech_detail=='LTRG1'"
    )["waccnomtech"].values[0]

    financials_dict = {
        "capex": {"wind": wind_2030_capex, "solarpv": solarpv_2030_capex},
        "wacc": {"wind": wind_2030_wacc, "solarpv": solarpv_2030_wacc},
    }

    return financials_dict


def load_regional_cost_multipliers():

    regional_cost_multipliers = pd.read_csv(
        "AEO_2020_regional_cost_corrections.csv", index_col=0
    )
    regional_cost_multipliers = regional_cost_multipliers.fillna(1)

    return regional_cost_multipliers


def load_site_locations(folder=Path.cwd(), as_gdf=True):

    site_locations = pd.read_csv(folder / "RUC_LatLonSites.csv", dtype={"Site": str})
    site_locations["Site"] = site_locations["Site"].str.zfill(6)

    if as_gdf:
        site_locations = gpd.GeoDataFrame(
            site_locations,
            crs="EPSG:4326",
            geometry=gpd.points_from_xy(
                site_locations.Longitude, site_locations.Latitude,
            ),
        )

    return site_locations


def fix_geometries(gdf):

    region_polys = {}
    fixed_regions = {}
    for region in gdf.index:
        region_polys[region] = []
        try:
            for i in range(len(gdf.loc[region, "geometry"])):
                region_polys[region].append(
                    shapely.geometry.Polygon(gdf.loc[region, "geometry"][i].exterior)
                )
        except TypeError:
            region_polys[region].append(
                shapely.geometry.Polygon(gdf.loc[region, "geometry"].exterior)
            )
        fixed_regions[region] = shapely.geometry.MultiPolygon(region_polys[region])

    gdf.geometry = [x for x in fixed_regions.values()]

    return gdf


def load_substations(min_kv=161):
    substation_gdf = gpd.read_file(
        CWD / "Electric_Substations" / "Electric_Substations.shp"
    )
    # substation_gdf = substation_gdf.to_crs(epsg=4326)
    substation_gdf = substation_gdf.loc[
        (substation_gdf["TYPE"] == "SUBSTATION")
        & (substation_gdf["STATUS"].isin(["IN SERVICE", "UNDER CONST"]))
        & (substation_gdf["MAX_VOLT"] >= min_kv),
        ["ID", "MAX_VOLT", "MIN_VOLT", "geometry"],
    ]

    substation_gdf = substation_gdf.rename(columns={"ID": "substation_id"})

    substation_gdf["latitude"] = substation_gdf.geometry.y
    substation_gdf["longitude"] = substation_gdf.geometry.x

    return substation_gdf


def load_ipm_shapefile(filetype="shp"):
    """Load the IPM shapefile or geojson file.

    Parameters
    ----------
    filetype : str, optional
        Either "shp" or "geojson", by default "shp"

    Returns
    -------
    GeoDataFrame
        IPM_Region (region names) and geometry columns
    """
    print("loading IPM shapefile")

    if filetype.lower() == "shp":
        file_path = IPM_SHAPEFILE_PATH
    elif filetype.lower() == "geojson":
        file_path = IPM_GEOJSON_PATH
    else:
        raise ValueError(
            f"Parameter 'filetype' must be 'shp' or 'geojson', not {filetype}"
        )
    ipm_regions = gpd.read_file(file_path)
    ipm_regions = ipm_regions.to_crs(epsg=4326)

    ipm_regions = fix_geometries(ipm_regions)

    return ipm_regions


def load_metro_areas_shapefile():
    shpfile_path = (
        CWD / "USA_Core_Based_Statistical_Area" / "USA_Core_Based_Statistical_Area.shp"
    )
    metro_areas = gpd.read_file(shpfile_path)
    metro_areas = metro_areas.to_crs(epsg=4326)

    metro_areas["center"] = metro_areas.centroid

    keep_cols = ["CBSA_ID", "NAME", "CBSA_TYPE", "POPULATION", "center", "geometry"]
    # metro_areas["geometry"] = metro_areas["center"]
    metro_areas = metro_areas.loc[:, keep_cols]
    metro_areas.columns = metro_areas.columns.str.lower()
    metro_areas["state"] = metro_areas["name"].str.split(", ").str[-1]
    metro_areas = metro_areas.loc[~metro_areas.state.isin(["AK", "HI", "PR"]), :]

    return metro_areas


def load_us_states_gdf():

    us_states = gpd.read_file(
        "https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_5m.json"
    )
    drop_states = ["Puerto Rico", "Alaska", "Hawaii"]
    us_states = us_states.loc[~(us_states["NAME"].isin(drop_states)), :]
    us_states = us_states.reset_index(drop=True)

    return us_states


def load_cpa_gdf(filepath, target_crs, slope_filter=None, layer=None):

    if layer is not None:
        cpa_gdf = gpd.read_file(filepath, layer=layer)
    else:
        cpa_gdf = gpd.read_file(filepath)

    if slope_filter:
        cpa_gdf = cpa_gdf.loc[cpa_gdf["m_slope"] <= slope_filter, :]
        cpa_gdf = cpa_gdf.reset_index(drop=True)

    cpa_gdf = cpa_gdf.to_crs(target_crs)
    cpa_gdf["Latitude"] = cpa_gdf.centroid.y
    cpa_gdf["Longitude"] = cpa_gdf.centroid.x
    cpa_gdf["cpa_id"] = cpa_gdf.index

    return cpa_gdf


def load_gen_profiles(site_list, resource, variable):
    if resource.lower() == "wind":
        resource = "Wind"
        resource_path = VCE_WIND_PATH
    elif resource.lower() == "solarpv":
        resource = "SolarPV"
        resource_path = VCE_SOLAR_PATH

    site_profiles = {}
    for s in site_list:
        fpath = f"Site_{s}_{resource}.nc4"
        site_data = netCDF4.Dataset(resource_path / fpath)
        gen_profile = np.array(site_data[variable])
        site_profiles[s] = gen_profile

    df = pd.DataFrame(site_profiles)

    return df.T


def load_site_capacity_factors(site_substation_metro=None):

    site_wind_cf = pd.read_csv("RUC_LatLonSites_CF.csv", skiprows=2)
    site_wind_cf["Site"] = site_wind_cf["Site"].astype(str).str.zfill(6)
    site_wind_cf.columns = [
        col.replace(" \n", " ").replace("\n", " ") for col in site_wind_cf.columns
    ]
    site_wind_cf = site_wind_cf.set_index("Site")

    if Path("Site_SolarPV_CF.csv").exists():
        site_solarpv_cf = pd.read_csv("Site_SolarPV_CF.csv", index_col=1)
        site_solarpv_cf.index = site_solarpv_cf.index.astype(str).str.zfill(6)
    else:
        site_solarpv_cf = load_gen_profiles(
            site_substation_metro["Site"],
            resource="solarPV",
            variable="Axis1_SolarPV_Lat",
        )

    site_cf_dict = {"wind": site_wind_cf, "solarpv": site_solarpv_cf}

    return site_cf_dict


def find_largest_cities(
    metro_areas_gdf, ipm_gdf, min_population=750000, max_cities_per_region=None
):
    _metro_areas_gdf = metro_areas_gdf.copy()
    _metro_areas_gdf["geometry"] = _metro_areas_gdf["center"]
    #     metro_ipm_gdf = gpd.sjoin(ipm_gdf, _metro_areas_gdf, how="left", op="intersects")
    metro_ipm_gdf = gpd.sjoin(ipm_gdf, _metro_areas_gdf, how="left", op="contains")

    df_list = []
    grouped = metro_ipm_gdf.groupby("IPM_Region", as_index=False)
    for _, _df in grouped:

        n_df = _df.loc[_df["population"] >= min_population, :]
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


def wa_capex(nearest_df):

    wa = np.average(nearest_df["interconnect_capex"], weights=nearest_df["km2"])

    return wa


def label_site_region(gdf, id_col, lat, lon):

    mask = pd.Series(index=range(len(lat)))

    for n in gdf.index:
        dataGeom = gdf["geometry"][n]
        data_mask = shapely.vectorized.contains(dataGeom, lon, lat)
        mask[data_mask] = gdf[id_col][n]

    return mask


def calc_interconnect_distances(
    site_gdf, substation_gdf, metro_gdf, site_id_col="cpa_id"
):

    # Substation to nearest metro
    _substation_gdf = substation_gdf.rename(
        columns={"latitude": "Latitude", "longitude": "Longitude"}
    )
    nearest_substation_metro = ckdnearest(
        _substation_gdf, metro_gdf.reset_index(drop=True),
    )
    nearest_substation_metro = nearest_substation_metro.rename(
        columns={"dist_mile": "substation_metro_tx_miles"}
    )

    if "latitude" not in substation_gdf.columns:
        print("lowercase lat isn't in substation_gdf")
        substation_gdf = substation_gdf.rename(
            columns={"latitude": "Latitude", "longitude": "Longitude"}
        )
    # print(substation_gdf.head())
    # Site to nearest substation
    nearest_site_substation = ckdnearest(
        site_gdf.reset_index(drop=True),
        substation_gdf.reset_index(drop=True)
        # .rename(
        #     columns={"Latitude": "latitude", "Longitude": "longitude"}
        # )
    )
    nearest_site_substation = nearest_site_substation.rename(
        columns={"dist_mile": "site_substation_spur_miles"}
    )
    # print(metro_gdf.head())
    # Site to nearest metro (direct spur line)
    nearest_site_metro = ckdnearest(
        site_gdf.reset_index(drop=True),
        metro_gdf.drop(columns=["IPM_Region"]).reset_index(drop=True),
    )
    # nearest_site_metro = nearest_site_metro.drop_duplicates(
    #     subset=["cpa_id", "cbsa_id"]
    # )
    nearest_site_metro = nearest_site_metro.rename(
        columns={"dist_mile": "site_metro_spur_miles"}
    )

    # Combine all of the distances into a single dataframe
    # Mapping is probably slower (b/c of setting the index) than merges but it helps
    # ensure unique IDs for sites and substations
    site_substation_metro = nearest_site_substation.copy()
    site_substation_metro["substation_metro_tx_miles"] = site_substation_metro[
        "substation_id"
    ].map(
        nearest_substation_metro.set_index("substation_id")["substation_metro_tx_miles"]
    )

    site_substation_metro["site_metro_spur_miles"] = site_substation_metro[
        site_id_col
    ].map(nearest_site_metro.set_index(site_id_col)["site_metro_spur_miles"])

    nan_sites = site_substation_metro.loc[site_substation_metro["cbsa_id"].isnull(), :]
    num_nan_sites = len(nan_sites)
    if num_nan_sites > 0:
        egrid_col = "eGrid_to_E"
        nan_regions = list(nan_sites[egrid_col].unique())
        print(
            f"There are {num_nan_sites} CPAs with no associated metro.\n"
            "This is probably because the site centroid is outside US borders.\n"
            f"The sites are located in {nan_regions} eGRID region(s),"
            " and are being removed."
        )

        site_substation_metro = site_substation_metro.dropna(subset=["cbsa_id"])

    return site_substation_metro


def calc_interconnect_costs_lcoe(site_substation_metro, resource, cap_rec_years=20):

    financials_dict = load_atb_capex_wacc()
    regional_cost_multipliers = load_regional_cost_multipliers()
    site_substation_metro_lcoe = site_substation_metro.copy()

    # Calculate interconnection capex, min of direct to metro and through a substation.
    # Include the difference in spur line and high-voltage tx costs by region.
    ipm_spur_costs = {
        ipm_region: spur_costs_2017[agg_region]
        for ipm_region, agg_region in rev_region_mapping.items()
    }
    ipm_tx_costs = {
        ipm_region: tx_costs_2017[agg_region]
        for ipm_region, agg_region in rev_region_mapping.items()
    }

    site_substation_metro_lcoe.loc[
        :, "spur_capex_mw_mile"
    ] = site_substation_metro_lcoe["IPM_Region"].map(ipm_spur_costs)
    site_substation_metro_lcoe.loc[:, "metro_direct_capex"] = (
        site_substation_metro_lcoe.loc[:, "spur_capex_mw_mile"]
        * site_substation_metro_lcoe.loc[:, "site_metro_spur_miles"]
    )
    site_substation_metro_lcoe.loc[:, "site_substation_capex"] = (
        site_substation_metro_lcoe.loc[:, "spur_capex_mw_mile"]
        * site_substation_metro_lcoe.loc[:, "site_substation_spur_miles"]
    )

    site_substation_metro_lcoe.loc[:, "tx_capex_mw_mile"] = site_substation_metro_lcoe[
        "IPM_Region"
    ].map(ipm_tx_costs)
    site_substation_metro_lcoe.loc[:, "substation_metro_capex"] = (
        site_substation_metro_lcoe.loc[:, "tx_capex_mw_mile"]
        * site_substation_metro_lcoe.loc[:, "substation_metro_tx_miles"]
    )
    site_substation_metro_lcoe.loc[:, "site_substation_metro_capex"] = (
        site_substation_metro_lcoe.loc[:, "site_substation_capex"]
        + site_substation_metro_lcoe.loc[:, "substation_metro_capex"]
    )

    site_substation_metro_lcoe.loc[
        :, "interconnect_capex"
    ] = site_substation_metro_lcoe[
        ["site_substation_metro_capex", "metro_direct_capex"]
    ].min(
        axis=1
    )

    # Calc site capex, including regional cost multipliers
    capex_lambda = (
        lambda x: regional_cost_multipliers.loc[rev_cost_mult_region_map[x], "Wind"]
        * financials_dict["capex"][resource]
    )
    # wind_capex_lambda = (
    #     lambda x: regional_cost_multipliers.loc[rev_cost_mult_region_map[x], "Wind"]
    #     * financials_dict["capex"]["wind"]
    # )
    # solarpv_capex_lambda = (
    #     lambda x: regional_cost_multipliers.loc[
    #         rev_cost_mult_region_map[x], "Solar PV—tracking"
    #     ]
    #     * financials_dict["capex"]["solarpv"]
    # )

    capex_map = {
        region: capex_lambda(region) for region in rev_cost_mult_region_map.keys()
    }
    # wind_capex_map = {
    #     region: wind_capex_lambda(region) for region in rev_cost_mult_region_map.keys()
    # }
    # solarpv_capex_map = {
    #     region: solarpv_capex_lambda(region)
    #     for region in rev_cost_mult_region_map.keys()
    # }

    print(f"Assigning {resource} capex values")
    site_substation_metro_lcoe.loc[:, "capex"] = site_substation_metro_lcoe[
        "IPM_Region"
    ].map(capex_map)
    # site_substation_metro_lcoe.loc[:, "solarpv_capex"] = site_substation_metro_lcoe[
    #     "IPM_Region"
    # ].map(solarpv_capex_map)

    # site_substation_metro_lcoe.loc[:, "wind_capex"] = site_substation_metro_lcoe[
    #     "IPM_Region"
    # ].map(wind_capex_map)

    # Calculate site, interconnect, and total annuities
    print(f"Calculating {resource} annuities")
    site_substation_metro_lcoe["resource_annuity"] = investment_cost_calculator(
        capex=site_substation_metro_lcoe["capex"],
        wacc=financials_dict["wacc"][resource],
        cap_rec_years=cap_rec_years,
    )
    # site_substation_metro_lcoe["solarpv_annuity"] = investment_cost_calculator(
    #     capex=site_substation_metro_lcoe["solarpv_capex"],
    #     wacc=financials_dict["wacc"]["solarpv"],
    #     cap_rec_years=cap_rec_years,
    # )
    # print("Calculating wind annuities")
    # site_substation_metro_lcoe["wind_annuity"] = investment_cost_calculator(
    #     capex=site_substation_metro_lcoe["wind_capex"],
    #     wacc=financials_dict["wacc"]["wind"],
    #     cap_rec_years=cap_rec_years,
    # )
    print("Calculating interconnect annuities")
    site_substation_metro_lcoe.loc[
        :, "interconnect_annuity"
    ] = investment_cost_calculator(
        capex=site_substation_metro_lcoe["interconnect_capex"],
        wacc=spur_line_wacc,
        cap_rec_years=spur_line_investment_years,
    )

    site_substation_metro_lcoe.loc[:, "total_site_annuity"] = (
        site_substation_metro_lcoe.loc[:, "resource_annuity"]
        + site_substation_metro_lcoe.loc[:, "interconnect_annuity"]
    )
    # site_substation_metro_lcoe.loc[:, "total_wind_site_annuity"] = (
    #     site_substation_metro_lcoe.loc[:, "wind_annuity"]
    #     + site_substation_metro_lcoe.loc[:, "interconnect_annuity"]
    # )
    # site_substation_metro_lcoe.loc[:, "total_solarpv_site_annuity"] = (
    #     site_substation_metro_lcoe.loc[:, "solarpv_annuity"]
    #     + site_substation_metro_lcoe.loc[:, "interconnect_annuity"]
    # )

    # Use site capacity factor to calculate LCOE
    # The column "Site" identifies the VCE site.
    site_cf_dict = load_site_capacity_factors(site_substation_metro_lcoe)

    variable = {
        "wind": "2012 100m Average Capacity Factor",
        "solarpv": "Axis1_SolarPV_Lat_CF",
    }

    site_substation_metro_lcoe.loc[:, f"{resource}_cf"] = (
        site_substation_metro_lcoe["Site"].map(
            site_cf_dict[resource][variable[resource]]
        )
        / 100
    )
    # site_substation_metro_lcoe.loc[:, "solarpv_tracking_cf"] = (
    #     site_substation_metro_lcoe["Site"].map(
    #         site_cf_dict["solarpv"]["Axis1_SolarPV_Lat_CF"]
    #     )
    #     / 100
    # )
    # site_substation_metro_lcoe.loc[:, "wind_100m_cf"] = (
    #     site_substation_metro_lcoe["Site"].map(
    #         site_cf_dict["wind"]["2012 100m Average Capacity Factor"]
    #     )
    #     / 100
    # )

    site_substation_metro_lcoe.loc[:, "lcoe"] = site_substation_metro_lcoe.loc[
        :, "total_site_annuity"
    ] / (site_substation_metro_lcoe.loc[:, f"{resource}_cf"] * 8760)

    # site_substation_metro_lcoe.loc[:, "wind_lcoe"] = site_substation_metro_lcoe.loc[
    #     :, "total_wind_site_annuity"
    # ] / (site_substation_metro_lcoe.loc[:, "wind_100m_cf"] * 8760)

    # site_substation_metro_lcoe.loc[:, "solarpv_lcoe"] = site_substation_metro_lcoe.loc[
    #     :, "total_solarpv_site_annuity"
    # ] / (site_substation_metro_lcoe.loc[:, "solarpv_tracking_cf"] * 8760)

    return site_substation_metro_lcoe


def main(resource="solarpv", scenario="base"):

    print("Loading states, voronoi, and CPAs")
    us_states = load_us_states_gdf()

    metro_voronoi_gdf = gpd.read_file("large_metro_voronoi.geojson")

    cpa_files = {
        "wind": "2020-05-19-OnshoreWind-Base-upto30deg_shp",
        "solarpv": "2020-05-28-SolarBase15deg_CPAs_shapefile",
    }
    cpa_slope_filter = {"wind": 19, "solarpv": 10}
    cpa_gdf = load_cpa_gdf(cpa_files[resource], target_crs=us_states.crs)
    if "m_slope" in cpa_gdf.columns:
        cpa_gdf = cpa_gdf.loc[cpa_gdf["m_slope"] <= cpa_slope_filter[resource], :]
    cpa_gdf = cpa_gdf.reset_index(drop=True)

    cpa_gdf["state"] = label_site_region(
        gdf=us_states, id_col="NAME", lat=cpa_gdf.Latitude, lon=cpa_gdf.Longitude
    )

    cpa_gdf["cbsa_id"] = label_site_region(
        gdf=metro_voronoi_gdf,
        id_col="cbsa_id",
        lat=cpa_gdf.Latitude,
        lon=cpa_gdf.Longitude,
    )

    cpa_gdf["city"] = cpa_gdf["cbsa_id"].map(
        metro_voronoi_gdf.set_index("cbsa_id")["name"]
    )
    cpa_gdf["IPM_Region"] = cpa_gdf["cbsa_id"].map(
        metro_voronoi_gdf.set_index("cbsa_id")["IPM_Region"]
    )

    site_locations = load_site_locations()
    site_locations = site_locations.rename(
        columns={"Latitude": "latitude", "Longitude": "longitude"}
    )
    cpa_vce_site = ckdnearest(cpa_gdf.copy(), site_locations.copy())
    cpa_vce_site = cpa_vce_site.drop(columns=["lat2", "lon2"])

    print("Loading other data")
    substation_gdf = load_substations(min_kv=161)
    # substation_gdf = substation_gdf.rename(
    #     columns={"latitude": "Latitude", "longitude": "Longitude"}
    # )

    ipm_gdf = load_ipm_shapefile()

    metro_gdf = load_metro_areas_shapefile()
    largest_metros = find_largest_cities(
        metro_areas_gdf=metro_gdf, ipm_gdf=ipm_gdf, min_population=750000
    )

    print("Calculating interconnect distances")
    cpa_substation_metro = calc_interconnect_distances(
        site_gdf=cpa_vce_site,
        substation_gdf=substation_gdf,
        metro_gdf=largest_metros,
        site_id_col="cpa_id",
    )

    print("Calculating interconnect costs")
    cpa_substation_metro_lcoe = calc_interconnect_costs_lcoe(
        cpa_substation_metro, resource=resource
    )

    print("Writing results to file")
    cpa_substation_metro_lcoe.to_csv(f"{scenario}_{resource}_lcoe.csv", index=False)


# def load_cluster_assignments(path):
#     df = pd.read_csv(path)
#     df = df.rename(columns={"Unnamed: 0": "Site"})
#     df["Site"] = df["Site"].astype(str).str.zfill(6)

#     return df


# def calc_cluster_site_spur_line_capex(cluster_assignments, site_locations, metro_areas, substations, spur_cost=3901, high_v_cost=1351):

#     cluster_assignments = pd.merge(
#         cluster_assignments,
#         site_locations,
#         on="Site"
#     )
#     _metro_areas = metro_areas.copy()
#     nearest_metro = ckdnearest(cluster_assignments, _metro_areas)
#     nearest_metro = nearest_metro.rename(columns={"dist_mile": "total_mile"})

#     _substations = substations.copy()
#     nearest_substation = ckdnearest(nearest_metro.drop(columns=["lat2", "lon2"]), _substations)
#     nearest_substation = nearest_substation.rename(columns={"dist_mile": "spur_mile"})

#     nearest_substation["high_v_distance"] = nearest_substation["total_mile"] - nearest_substation["spur_mile"]
#     nearest_substation.loc[
#         nearest_substation["high_v_distance"] < 0,
#         "high_v_distance"
#     ] = 0

#     nearest_substation.loc[
#         nearest_substation["spur_mile"] > nearest_substation["total_mile"],
#         "spur_mile"
#     ] = nearest_substation["total_mile"]

#     nearest_substation["spur_capex"] = nearest_substation["spur_mile"] * spur_cost
#     nearest_substation["high_v_capex"] = nearest_substation["high_v_distance"] * high_v_cost
#     nearest_substation["interconnect_capex"] = nearest_substation["spur_capex"] + nearest_substation["high_v_capex"]

#     assert (nearest_substation[["spur_capex", "high_v_capex"]] < 0).sum().sum() == 0

#     return nearest_substation


# site_locations = load_site_locations()
# ipm_gdf = load_ipm_shapefile()
# metro_gdf = load_metro_areas_shapefile()
# largest_metro_gdf = find_largest_cities(metro_gdf, ipm_gdf, min_population=750000)

# test_assignments = load_cluster_assignments("/Users/greg/Documents/CATF/VCE generation profiles/cluster_assignments/cluster_assignments_wind_base3k_WECC_MT.csv")

# substation_gdf = load_substations()
# substation_gdf

# substation_ipm_gdf = gpd.sjoin(ipm_gdf, substation_gdf, how="left", op="contains").reset_index(drop=True)

# nearest_metros = calc_cluster_site_spur_line_capex(
#     test_assignments,
#     site_locations,
#     largest_metro_gdf,
#     substation_ipm_gdf
# )
# nearest_metros.to_clipboard()


# nearest_substation = calc_cluster_site_spur_line_capex(test_assignments, site_locations, substation_ipm_gdf)

# cluster_metro_distance = nearest_metros.groupby("cluster").apply(wa_distance)
# cluster_metro_distance

# nearest_metros["dist_mile"].describe()

# CA_REGIONS = ["WECC_SCE", "WECC_IID", "WEC_LADW", "WEC_CALN", "WEC_BANC", "WEC_SDGE"]

# WECC_TX_CAPEX = {
#     "spur": 4181,
#     "high_v": 1448
# }
# CA_TX_CAPEX = {
#     "spur": 5574,
#     "high_v": 2948
# }


# def main():

#     site_locations = load_site_locations()
#     ipm_gdf = load_ipm_shapefile()
#     metro_gdf = load_metro_areas_shapefile()
#     largest_metro_gdf = find_largest_cities(metro_gdf, ipm_gdf, min_population=750000)

#     substation_gdf = load_substations()
#     substation_ipm_gdf = gpd.sjoin(ipm_gdf, substation_gdf, how="left", op="contains").reset_index(drop=True)

#     cluster_assn_files = list((CWD / "cluster_assignments").glob("*WEC*"))

#     solar_base_files = [fn for fn in cluster_assn_files if "solarPV" in fn.name and "base" in fn.name]
#     wind_base_files = [fn for fn in cluster_assn_files if "wind" in fn.name and "base" in fn.name]

#     interconnect_annuity_folder = CWD / "interconnect_annuity"
#     interconnect_annuity_folder.mkdir(exist_ok=True)

#     keep_cols = [
#         "Site",
#         "cf",
#         "km2",
#         "IPM_Region",
#         "total_mile",
#         "spur_mile",
#         "high_v_distance",
#         "spur_capex",
#         "high_v_capex",
#         "interconnect_capex",
#         "interconnect_annuity"
#     ]

#     # solar
#     print("calculating solar")
#     solar_list = []
#     for fn in solar_base_files:
#         name = fn.name
#         region = name.split(".")[0].split("base_")[-1]
#         if region in CA_REGIONS:
#             spur_capex = CA_TX_CAPEX["spur"]
#             high_v_capex = CA_TX_CAPEX["high_v"]
#         else:
#             spur_capex = WECC_TX_CAPEX["spur"]
#             high_v_capex = WECC_TX_CAPEX["high_v"]

#         assignments = load_cluster_assignments(fn)

#         site_interconnect_capex = calc_cluster_site_spur_line_capex(
#             assignments,
#             site_locations,
#             largest_metro_gdf,
#             substation_ipm_gdf,
#             spur_cost=spur_capex,
#             high_v_cost=high_v_capex
#         )

#         # cluster_interconnect_capex = site_interconnect_capex.groupby(["IPM_Region", "cluster"], as_index=False).apply(wa_capex)

#         site_interconnect_capex["region"] = region

#         solar_list.append(site_interconnect_capex)

#     solar_cluster_interconnect_df = pd.concat(solar_list)
#     solar_cluster_interconnect_df["interconnect_annuity"] = investment_cost_calculator(
#         solar_cluster_interconnect_df["interconnect_capex"],
#         wacc=0.069,
#         cap_rec_years=60
#     ).round(-3)
#     # return solar_cluster_interconnect_df
#     # cluster_interconnect_capex = solar_cluster_interconnect_df.groupby(["region", "cluster"]).apply(wa_capex)
#     solar_cluster_interconnect_df[keep_cols].to_csv(
#         CWD / "interconnect_annuity" / f"solar_interconnect_annuity_WECC.csv",
#         index=False
#     )
#     # solar_cluster_interconnect_df.to_csv("regional_solar_interconnect_capex.csv")

#     # wind
#     print("calculating wind")
#     wind_list = []
#     for fn in wind_base_files:
#         name = fn.name
#         region = name.split(".")[0].split("base3k_")[-1]
#         if region in CA_REGIONS:
#             spur_capex = CA_TX_CAPEX["spur"]
#             high_v_capex = CA_TX_CAPEX["high_v"]
#         else:
#             spur_capex = WECC_TX_CAPEX["spur"]
#             high_v_capex = WECC_TX_CAPEX["high_v"]

#         assignments = load_cluster_assignments(fn)

#         site_interconnect_capex = calc_cluster_site_spur_line_capex(
#             assignments,
#             site_locations,
#             largest_metro_gdf,
#             substation_ipm_gdf,
#             spur_cost=spur_capex,
#             high_v_cost=high_v_capex
#         )

#         # cluster_interconnect_capex = site_interconnect_capex.groupby(["IPM_Region", "cluster"], as_index=False).apply(wa_capex)

#         site_interconnect_capex["region"] = region

#         wind_list.append(site_interconnect_capex)

#     wind_cluster_interconnect_df = pd.concat(wind_list)
#     wind_cluster_interconnect_df["interconnect_annuity"] = investment_cost_calculator(
#         wind_cluster_interconnect_df["interconnect_capex"],
#         wacc=0.069,
#         cap_rec_years=60
#     ).round(-3)

#     wind_cluster_interconnect_df[keep_cols].to_csv(
#         CWD / "interconnect_annuity" / f"wind_interconnect_annuity_WECC.csv",
#         index=False
#     )
#     # cluster_interconnect_capex = wind_cluster_interconnect_df.groupby(["region", "cluster"]).apply(wa_capex)
#     # cluster_interconnect_capex.to_csv("regional_wind_interconnect_capex.csv")


if __name__ == "__main__":
    typer.run(main)
