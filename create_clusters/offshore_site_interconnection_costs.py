import numpy as np
import netCDF4
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from pathlib import Path
import logging
import typer

from powergenome.params import DATA_PATHS, IPM_SHAPEFILE_PATH, IPM_GEOJSON_PATH
from powergenome.transmission import haversine
from powergenome.nrelatb import investment_cost_calculator, fetch_atb_costs
from powergenome.util import reverse_dict_of_lists, init_pudl_connection, find_centroid
from powergenome.price_adjustment import inflation_price_adjustment

from site_interconnection_costs import ckdnearest

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
    "ny": ["NY_Z_A", "NY_Z_B", "NY_Z_C&E", "NY_Z_D", "NY_Z_F", "NY_Z_G-I", "NY_Z_J",],
    "tva": ["S_C_TVA",],
    "south": ["S_SOU",],
    "fl": ["FRCC"],
    "vaca": ["S_VACA"],
    "ne": ["NY_Z_K", "NENG_CT", "NENG_ME", "NENGREST",],
}

rev_region_mapping = reverse_dict_of_lists(tx_capex_region_map)

spur_costs_2013 = {
    "wecc": 3900,
    "ca": 3900 * 2.25,  # According to Reeds docs, CA is 2.25x the rest of WECC
    "tx": 3900,
    "upper_midwest": 3900,
    "lower_midwest": 3800,
    "miso_s": 3900 * 2.25,
    "great_lakes": 4100,
    "pjm_s": 3900 * 2.25,
    "pj_pa": 3900 * 2.25,
    "pjm_md_nj": 3900 * 2.25,
    "ny": 3900 * 2.25,
    "tva": 3800,
    "south": 4950,
    "fl": 4100,
    "vaca": 3800,
    "ne": 3900 * 2.25,
}

spur_costs_2017 = {
    region: inflation_price_adjustment(cost, 2013, 2017)
    for region, cost in spur_costs_2013.items()
}

tx_costs_2013 = {
    "wecc": 1350,
    "ca": 1350 * 2.25,  # According to Reeds docs, CA is 2.25x the rest of WECC
    "tx": 1350,
    "upper_midwest": 900,
    "lower_midwest": 900,
    "miso_s": 1750,
    "great_lakes": 1050,
    "pjm_s": 1350,
    "pj_pa": 1750,
    "pjm_md_nj": 4250,  # Bins are $1500 wide - assume max bin is $750 above max
    "ny": 2750,
    "tva": 1050,
    "south": 1350,
    "fl": 1350,
    "vaca": 900,
    "ne": 4250,  # Bins are $1500 wide - assume max bin is $750 above max
}

tx_costs_2017 = {
    region: inflation_price_adjustment(cost, 2013, 2017)
    for region, cost in tx_costs_2013.items()
}

spur_line_wacc = 0.069
spur_line_investment_years = 60


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
    centroid = find_centroid(cpa_gdf)
    cpa_gdf["Latitude"] = centroid.y
    cpa_gdf["Longitude"] = centroid.x
    cpa_gdf["cpa_id"] = cpa_gdf.index

    cpa_gdf["prefSite"] = cpa_gdf["prefSite"].fillna(0)

    dist_cols = ["d_coast_sub_161kVplus", "d_coast", "d_sub_load_metro_750k_center"]
    for col in dist_cols:
        mile_col = f"{col}_miles"
        cpa_gdf[mile_col] = cpa_gdf[col] * 1.60934

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
    offshore_spur_costs = pd.read_csv("atb_offshore_spur_costs.csv", index_col=0)
    offshore_spur_costs = offshore_spur_costs * 1000
    offshore_spur_costs.columns = [str(x) for x in offshore_spur_costs.columns]

    # Include finance factor of 1.032 from ATB spreadsheet
    offshore_fixed_2030_spur = offshore_spur_costs.loc["TRG 3 - Mid", "2030"] * 1.032
    offshore_floating_2030_spur = (
        offshore_spur_costs.loc["TRG 10 - Mid", "2030"] * 1.032
    )

    offshore_fixed_spur_mw_mile = offshore_fixed_2030_spur / 30 * 1.60934
    offshore_floating_spur_mw_mile = offshore_floating_2030_spur / 30 * 1.60934

    offshorewind_fixed_2030_capex = (
        atb_costs.query(
            "technology=='OffShoreWind' & cost_case=='Mid'"
            " & financial_case=='Market' & basis_year==2030  & tech_detail=='OTRG3'"
        )["capex"].values[0]
        - offshore_fixed_2030_spur
    )

    offshorewind_floating_2030_capex = (
        atb_costs.query(
            "technology=='OffShoreWind' & cost_case=='Mid'"
            " & financial_case=='Market' & basis_year==2030  & tech_detail=='OTRG10'"
        )["capex"].values[0]
        - offshore_floating_2030_spur
    )

    offshorewind_fixed_2030_wacc = atb_costs.query(
        "technology=='OffShoreWind' & cost_case=='Mid'"
        " & financial_case=='Market' & basis_year==2030  & tech_detail=='OTRG3'"
    )["waccnomtech"].values[0]

    offshorewind_floating_2030_wacc = atb_costs.query(
        "technology=='OffShoreWind' & cost_case=='Mid'"
        " & financial_case=='Market' & basis_year==2030  & tech_detail=='OTRG10'"
    )["waccnomtech"].values[0]

    financials_dict = {
        "capex": {
            "fixed": offshorewind_fixed_2030_capex,
            "floating": offshorewind_floating_2030_capex,
        },
        "offshore_trg_spur_capex_mw_mile": {
            "fixed": offshore_fixed_spur_mw_mile,
            "floating": offshore_floating_spur_mw_mile,
        },
        "wacc": {
            "fixed": offshorewind_fixed_2030_wacc,
            "floating": offshorewind_floating_2030_wacc,
        },
    }

    return financials_dict


def load_site_capacity_factors(site_substation_metro=None):

    site_wind_cf = pd.read_csv("RUC_LatLonSites_CF.csv", skiprows=2)
    site_wind_cf["Site"] = site_wind_cf["Site"].astype(str).str.zfill(6)
    site_wind_cf.columns = [
        col.replace(" \n", " ").replace("\n", " ") for col in site_wind_cf.columns
    ]
    site_wind_cf = site_wind_cf.set_index("Site")

    return site_wind_cf


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


def load_regional_cost_multipliers():

    regional_cost_multipliers = pd.read_csv(
        "AEO_2020_regional_cost_corrections.csv", index_col=0
    )
    regional_cost_multipliers = regional_cost_multipliers.fillna(1)

    return regional_cost_multipliers


# def ckdnearest(gdA, gdB):
#     "https://gis.stackexchange.com/a/301935"
#     nA = np.array(list(zip(gdA.Latitude, gdA.Longitude)))
#     nB = np.array(list(zip(gdB["latitude"], gdB["longitude"])))
#     btree = cKDTree(nB)
#     dist, idx = btree.query(nA, k=1)

#     gdB.rename(columns={"latitude": "lat2", "longitude": "lon2"}, inplace=True)

#     gdf = pd.concat(
#         [
#             gdA.reset_index(drop=True),
#             gdB.loc[idx, gdB.columns != "geometry"].reset_index(drop=True),
#         ],
#         axis=1,
#     )
#     gdf["dist_mile"] = gdf.apply(
#         lambda row: haversine(
#             row["Longitude"], row["Latitude"], row["lon2"], row["lat2"], units="mile"
#         ),
#         axis=1,
#     )

#     return gdf


def calc_interconnect_costs_lcoe(cpa_gdf, cap_rec_years=20):

    financials_dict = load_atb_capex_wacc()
    regional_cost_multipliers = load_regional_cost_multipliers()
    cpa_gdf_lcoe = cpa_gdf.copy()

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

    cpa_gdf_lcoe.loc[:, "spur_capex_mw_mile"] = cpa_gdf_lcoe["IPM_Region"].map(
        ipm_spur_costs
    )
    cpa_gdf_lcoe.loc[:, "land_substation_capex"] = (
        cpa_gdf_lcoe.loc[:, "spur_capex_mw_mile"]
        * cpa_gdf_lcoe.loc[:, "d_coast_sub_161kVplus_miles"]
    )

    cpa_gdf_lcoe.loc[:, "tx_capex_mw_mile"] = cpa_gdf_lcoe["IPM_Region"].map(
        ipm_tx_costs
    )
    cpa_gdf_lcoe.loc[:, "substation_metro_capex"] = (
        cpa_gdf_lcoe.loc[:, "tx_capex_mw_mile"]
        * cpa_gdf_lcoe.loc[:, "d_sub_load_metro_750k_center_miles"]
    )

    cpa_gdf_lcoe.loc[:, "offshore_spur_capex_mw_mile"] = cpa_gdf_lcoe[
        "turbineType"
    ].map(financials_dict["offshore_trg_spur_capex_mw_mile"])
    cpa_gdf_lcoe.loc[:, "offshore_spur_capex"] = (
        cpa_gdf_lcoe.loc[:, "offshore_spur_capex_mw_mile"]
        * cpa_gdf_lcoe.loc[:, "d_coast_miles"]
    )

    cpa_gdf_lcoe.loc[:, "interconnect_capex"] = (
        cpa_gdf_lcoe.loc[:, "land_substation_capex"]
        + cpa_gdf_lcoe.loc[:, "substation_metro_capex"]
        + cpa_gdf_lcoe.loc[:, "offshore_spur_capex"]
    )

    # Calc site capex, including regional cost multipliers
    fixed_capex_lambda = (
        lambda x: regional_cost_multipliers.loc[
            rev_cost_mult_region_map[x], "Wind offshore"
        ]
        * financials_dict["capex"]["fixed"]
    )
    floating_capex_lambda = (
        lambda x: regional_cost_multipliers.loc[
            rev_cost_mult_region_map[x], "Wind offshore"
        ]
        * financials_dict["capex"]["floating"]
    )

    fixed_capex_map = {
        region: fixed_capex_lambda(region) for region in rev_cost_mult_region_map.keys()
    }
    floating_capex_map = {
        region: floating_capex_lambda(region)
        for region in rev_cost_mult_region_map.keys()
    }

    logger.info(f"Assigning capex values")
    cpa_gdf_lcoe.loc[cpa_gdf_lcoe["turbineType"] == "fixed", "capex"] = cpa_gdf_lcoe[
        "IPM_Region"
    ].map(fixed_capex_map)
    cpa_gdf_lcoe.loc[cpa_gdf_lcoe["turbineType"] == "floating", "capex"] = cpa_gdf_lcoe[
        "IPM_Region"
    ].map(floating_capex_map)

    # Calculate site, interconnect, and total annuities
    logger.info(f"Calculating resource annuities")
    cpa_gdf_lcoe["resource_annuity"] = investment_cost_calculator(
        capex=cpa_gdf_lcoe["capex"],
        wacc=financials_dict["wacc"]["fixed"],  # fixed/floating have same wacc
        cap_rec_years=cap_rec_years,
    )
    logger.info("Calculating interconnect annuities")
    cpa_gdf_lcoe.loc[:, "interconnect_annuity"] = investment_cost_calculator(
        capex=cpa_gdf_lcoe["interconnect_capex"],
        wacc=spur_line_wacc,
        cap_rec_years=spur_line_investment_years,
    )

    cpa_gdf_lcoe.loc[:, "total_site_annuity"] = (
        cpa_gdf_lcoe.loc[:, "resource_annuity"]
        + cpa_gdf_lcoe.loc[:, "interconnect_annuity"]
    )

    # Use site capacity factor to calculate LCOE
    # The column "Site" identifies the VCE site.
    site_cf = load_site_capacity_factors(cpa_gdf_lcoe)

    cpa_gdf_lcoe.loc[:, "offshore_wind_cf"] = (
        cpa_gdf_lcoe["Site"].map(site_cf["2012 160m Average Capacity Factor"]) / 100
    )

    cpa_gdf_lcoe.loc[:, "lcoe"] = cpa_gdf_lcoe.loc[:, "total_site_annuity"] / (
        cpa_gdf_lcoe.loc[:, "offshore_wind_cf"] * 8760
    )

    return cpa_gdf_lcoe


def main(
    voronoi_gdf_fn: str = "large_metro_voronoi.geojson", fn_prefix: str = "",
):
    logger.info("Loading states, voronoi, and CPAs")
    us_states = load_us_states_gdf()

    metro_voronoi_gdf = gpd.read_file("large_metro_voronoi.geojson")
    cpa_gdf = load_cpa_gdf(
        "20200612_combined_wind_0_01_offshore_supp_CPA_wAtt_US_LCOE.gdb",
        target_crs=us_states.crs,
        layer="combined_wind_0_01_offshore_supp_CPA_wAtt_US_LCOE",
    )

    # Specify fixed (OTRG3) and floating (OTRG10), with cutoff at 50m
    # cpa_gdf["TRG"] = "OTRG10"
    # cpa_gdf.loc[cpa_gdf["m_seafloor"] >= -50, "TRG"] = "OTRG3"

    logger.info("Finding nearest MSA to assign IPM Region and cbsa_id")
    cpa_metro = ckdnearest(
        cpa_gdf.reset_index(drop=True), metro_voronoi_gdf.reset_index(drop=True)
    )
    cpa_metro = cpa_metro.drop(columns=["lat2", "lon2"])
    nnv_filter = cpa_metro.loc[cpa_metro.IPM_Region == "WECC_NNV", :].index
    cpa_metro.loc[nnv_filter, "IPM_Region"] = "WEC_CALN"
    cpa_metro.loc[nnv_filter, "metro_id"] = "41860"

    logger.info("Matching CPAs with VCE sites")
    site_locations = load_site_locations()
    site_locations = site_locations.rename(
        columns={"Latitude": "latitude", "Longitude": "longitude"}
    )
    cpa_vce_site = ckdnearest(cpa_metro.copy(), site_locations.copy())
    cpa_vce_site = cpa_vce_site.drop(columns=["lat2", "lon2"])

    cpa_vce_lcoe = calc_interconnect_costs_lcoe(cpa_vce_site)

    logger.info("Writing results to file")

    # cpa_vce_lcoe.drop(columns=["geometry"]).to_csv("base_offshorewind_lcoe.csv", index=False, float_format='%.5f')

    geodata_cols = [
        "cpa_id",
        # "state",
        "Site",
        "metro_id",
        "IPM_Region",
        "interconnect_annuity",
        "lcoe",
        "geometry",
    ]
    cpa_vce_lcoe.drop(columns=["geometry"]).to_csv(
        f"{fn_prefix}base_offshorewind_lcoe.csv"
    )


if __name__ == "__main__":
    typer.run(main)
