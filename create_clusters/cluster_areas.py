import numpy as np
import netCDF4
import pandas as pd
import scipy.cluster.hierarchy as hac
from pathlib import Path
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import typer
import shapely
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

CWD = Path.cwd()
VCE_DATA_PATH = Path("/Volumes/Macintosh HD 1/Updated_Princeton_Data")
VCE_WIND_PATH = VCE_DATA_PATH / "PRINCETON-Wind-Data-2012"
VCE_SOLAR_PATH = VCE_DATA_PATH / "PRINCETON-Solar-Data-2012"


def load_site_locations(folder=Path.cwd()):

    site_locations = pd.read_csv(folder / "RUC_LatLonSites.csv", dtype={"Site": str})
    site_locations["Site"] = site_locations["Site"].str.zfill(6)

    return site_locations


def add_cluster_labels(df, clusters=range(1, 51), lcoe_col="lcoe"):
    Z = hac.linkage(df[lcoe_col].values.reshape(-1, 1), method="ward")

    for n in clusters:
        df[f"cluster_{n}"] = hac.fcluster(Z, n, criterion="maxclust")

    return df


def cluster_weighted_avg_profile(df, site_profiles, cluster_index, weight_col="Area"):
    """Calculate the average CF in each hour, weighted by land area of each site

    Parameters
    ----------
    df : DataFrame
        Each row is a CPA with an associated "Site" identifier, which must match an
        index value in the "site_profiles" dataframe.
    site_profiles : DataFrame
        Hourly capacity factor profiles. The index identifies a "Site" and the columns
        are numbered hours.
    weight_col : str, optional
        The column in "df" to use for weighting the average, by default "Area"

    Returns
    -------
    DataFrame
        A single row dataframe with hourly capacity factors from 0-1.
    """
    cluster_wm = np.average(
        site_profiles.loc[df["Site"], :], weights=df[weight_col], axis=0
    )

    # Convert from 0-100 CF values down to 0-1 CF
    # if results.mean(axis=1) > 1:
    cluster_wm = cluster_wm / 100

    assert np.std(cluster_wm) != 0

    return {cluster_index: cluster_wm}


def load_gen_profiles(site_list, resource_type, variable, scenario):
    if resource_type.lower() == "wind":
        resource = "Wind"
        resource_path = VCE_WIND_PATH
    elif resource_type.lower() == "offshorewind":
        resource = "Wind"
        resource_path = VCE_WIND_PATH
    elif resource_type.lower() == "solarpv":
        resource = "SolarPV"
        resource_path = VCE_SOLAR_PATH

    fn = f"{scenario}_{resource_type}_site_profiles.parquet"
    if Path(fn).exists():
        logger.info("Profiles already saved as parquet file")
        df = pd.read_parquet(fn)

    else:
        logger.info("Loading all profiles from .nc4")
        site_profiles = {}
        for s in site_list:
            fpath = f"Site_{s}_{resource}.nc4"
            site_data = netCDF4.Dataset(resource_path / fpath)
            gen_profile = np.array(site_data[variable])
            site_profiles[s] = gen_profile

        df = pd.DataFrame(site_profiles)

        logger.info("Saving profiles to parquet")
        df.to_parquet(fn)

    return df.T


def renorm_solar(df):
    "Account for DC capacity higher than AC capacity, with values over 100 clipped."
    df = df * 1.3
    df = df.where(df <= 100, 100)

    return df


def wa_column(df, value_col, weight_col="Area"):

    wa = np.average(df[value_col], axis=0, weights=df[weight_col])

    return pd.Series(wa)


def wa_rmse(df, lcoe_col, weight_col="Area"):

    wa = mean_squared_error(
        df[lcoe_col],
        [wa_column(df, lcoe_col, weight_col)] * len(df),
        sample_weight=df[weight_col],
    )

    return pd.Series(wa)


def load_lcoe_data(path):
    cpa_lcoe = pd.read_csv(path, dtype={"Site": str})

    return cpa_lcoe


def make_clusters_tidy(cluster_df, additional_cluster_cols=[]):

    cluster_cols = [col for col in cluster_df.columns if "cluster_" in col]
    keep_cols = [
        "site_substation_spur_miles",
        "substation_metro_tx_miles",
        "site_metro_spur_miles",
        "offshore_spur_miles",
        "spur_miles",
        "tx_miles",
        "interconnect_annuity",
        "m_popden",
        "GW",
    ]
    keep_cols = [col for col in keep_cols if col in cluster_df.columns]
    id_vars = (
        additional_cluster_cols
        + ["IPM_Region", "cbsa_id", "cpa_id", "Site", "lcoe", "Area",]
        + keep_cols
    )

    tidy_clustered = pd.melt(
        cluster_df,
        id_vars=id_vars,
        value_vars=cluster_cols,
        var_name="cluster_level",
        value_name="cluster",
    )

    tidy_clustered["cluster_level"] = (
        tidy_clustered["cluster_level"].str.split("_").str[-1].astype(int)
    )

    return tidy_clustered


def make_cluster_metadata(
    tidy_clustered,
    additional_group_cols=[],
    # resource_type,
    # resource_mw_km2,
    relative_rmse_filter=0.025,
    gw_filter=0.5,
):
    group_cols = [
        "IPM_Region",
        "cbsa_id",
        "cluster_level",
        "cluster",
    ] + additional_group_cols
    logger.info("weighted lcoe")
    clustered_meta = tidy_clustered.groupby(group_cols).apply(
        wa_column, value_col="lcoe", weight_col="Area"
    )
    clustered_meta.columns = ["lcoe"]

    sum_cols = ["Area", "GW"]
    clustered_meta[sum_cols] = tidy_clustered.groupby(group_cols)[sum_cols].sum()

    # if resource_type == "offshorewind":
    #     clustered_meta
    # clustered_meta["GW"] = clustered_meta["Area"] * resource_mw_km2 / 1000

    avg_std_capacity = (
        clustered_meta.reset_index()
        .groupby(["cbsa_id", "cluster_level"], as_index=False)["GW"]
        .sum()
        .groupby("cbsa_id")["GW"]
        .std()
        .mean()
    )

    # Test to make sure the capacity in each cluster level is consistant.
    assert np.allclose(avg_std_capacity, 0)

    logger.info("weighted lcoe error")
    clustered_meta["rmse"] = tidy_clustered.groupby(group_cols).apply(
        wa_rmse, lcoe_col="lcoe"
    )

    clustered_meta["relative_rmse"] = clustered_meta["rmse"] / clustered_meta["lcoe"]

    other_cols = [
        "site_substation_spur_miles",
        "substation_metro_tx_miles",
        "site_metro_spur_miles",
        "offshore_spur_miles",
        "spur_miles",
        "tx_miles",
        "interconnect_annuity",
        "m_popden",
    ]
    other_cols = [col for col in other_cols if col in tidy_clustered.columns]
    logger.info("weighted other cols")
    clustered_meta[other_cols] = tidy_clustered.groupby(group_cols).apply(
        wa_column, value_col=other_cols
    )

    clustered_meta["meets_criteria"] = False
    clustered_meta.loc[
        (clustered_meta["relative_rmse"] <= relative_rmse_filter)
        | (clustered_meta["GW"] <= gw_filter),
        "meets_criteria",
    ] = True

    logger.info("Filtering metadata clusters")
    df_list = []
    for _, _df in clustered_meta.groupby(
        ["IPM_Region", "cbsa_id", "cluster_level"] + additional_group_cols
    ):
        if len(_df) > _df["meets_criteria"].sum():
            df_list.append(_df)

    filtered_clustered_meta = pd.concat(df_list)

    return filtered_clustered_meta


def make_weighted_profiles(
    tidy_clustered, cluster_meta, site_profiles, additional_group_cols=[], n_jobs=-2
):
    group_cols = [
        "IPM_Region",
        "cbsa_id",
        "cluster_level",
        "cluster",
    ] + additional_group_cols
    logger.info("Create weighted profiles")
    wa_profililes_list = Parallel(n_jobs=n_jobs, verbose=8)(
        delayed(cluster_weighted_avg_profile)(
            df=_df, site_profiles=site_profiles, cluster_index=idx
        )
        for idx, _df in tidy_clustered.groupby(group_cols)
    )

    # return wa_profililes_list
    from collections import ChainMap

    combined_dict = dict(ChainMap(*wa_profililes_list))
    wa_profiles = pd.DataFrame(data=combined_dict.values(), index=combined_dict.keys())
    wa_profiles.index = wa_profiles.index.rename(group_cols)

    filtered_wa_profiles = wa_profiles  # .loc[cluster_meta.index, :]
    logger.info("Making IPM Region a category")
    filtered_wa_profiles = filtered_wa_profiles.reset_index()
    filtered_wa_profiles["IPM_Region"] = filtered_wa_profiles["IPM_Region"].astype(
        "category"
    )

    logger.info("Make profiles tidy")
    tidy_wa_profiles = filtered_wa_profiles.melt(
        id_vars=group_cols,
        value_vars=list(wa_profiles.columns),
        var_name="hour",
        value_name="capacity_factor",
    )

    logger.info("Sort tidy profiles")
    sort_cols = additional_group_cols + [
        "IPM_Region",
        "cbsa_id",
        "cluster_level",
        "cluster",
        "hour",
    ]
    tidy_wa_profiles = tidy_wa_profiles.sort_values(by=sort_cols).reset_index(drop=True)

    return tidy_wa_profiles


def set_final_spur_columns(cpa_lcoe, resource_type):

    if resource_type != "offshorewind":
        cpa_lcoe["spur_miles"] = cpa_lcoe["site_substation_spur_miles"]
        cpa_lcoe["tx_miles"] = cpa_lcoe["substation_metro_tx_miles"]
        direct_capex = cpa_lcoe["metro_direct_capex"]
        indirect_capex = cpa_lcoe[
            ["site_substation_capex", "substation_metro_capex"]
        ].sum(axis=1)
        cpa_lcoe.loc[direct_capex < indirect_capex, "spur_miles"] = cpa_lcoe[
            "site_metro_spur_miles"
        ]
        cpa_lcoe.loc[direct_capex < indirect_capex, "tx_miles"] = 0
    else:
        cpa_lcoe["spur_miles"] = cpa_lcoe["d_coast_sub_161kVplus_miles"]
        cpa_lcoe["offshore_spur_miles"] = cpa_lcoe["d_coast_miles"]
        cpa_lcoe["tx_miles"] = cpa_lcoe["d_sub_load_metro_750k_center_miles"]

    return cpa_lcoe


def set_cpa_capacity(cpa_lcoe, resource_type, resource_density):

    if resource_type == "offshorewind":
        cpa_lcoe["GW"] = (
            cpa_lcoe["Area"] * cpa_lcoe["turbineType"].map(resource_density) / 1000
        )
    else:
        cpa_lcoe["GW"] = cpa_lcoe["Area"] * resource_density[resource_type] / 1000

    return cpa_lcoe


def main(
    lcoe_path,
    resource_type="solarpv",
    scenario="base",
    relative_rmse_filter: float = 0.025,
    gw_filter: float = 0.5,
    create_profiles: bool = True,
    n_jobs: int = -2,
    max_cluster_levels: int = 50,
):
    if isinstance(relative_rmse_filter, str):
        logger.info("Converting 'relative_rmse_filter' from string to float")
        relative_rmse_filter = float(relative_rmse_filter)

    resource_density = {"solarpv": 45, "wind": 2.7, "fixed": 5, "floating": 8}
    resource_variable = {
        "solarpv": "Axis1_SolarPV_Lat",
        "wind": "Wind_Power_100m",
        "offshorewind": "Wind_Power_160m",
    }

    if resource_type == "offshorewind":
        additional_group_cols = ["turbineType", "prefSite"]
    else:
        additional_group_cols = []

    if resource_type == "solarpv":
        gw_filter = 2.5

    cpa_lcoe = (
        load_lcoe_data(lcoe_path)
        .pipe(set_final_spur_columns, resource_type=resource_type)
        .pipe(
            set_cpa_capacity,
            resource_type=resource_type,
            resource_density=resource_density,
        )
    )

    logger.info("LCOE loaded, calculating cluster labels")
    cpa_lcoe_cluster_labels = cpa_lcoe.groupby(
        ["IPM_Region", "cbsa_id"] + additional_group_cols
    ).apply(
        add_cluster_labels, clusters=range(1, max_cluster_levels + 1), lcoe_col="lcoe"
    )

    logger.info("Making clusters tidy")

    tidy_clustered = make_clusters_tidy(cpa_lcoe_cluster_labels, additional_group_cols)

    logger.info("Making cluster metadata")

    cluster_meta = make_cluster_metadata(
        tidy_clustered=tidy_clustered,
        additional_group_cols=additional_group_cols,
        relative_rmse_filter=relative_rmse_filter,
        gw_filter=gw_filter,
    )

    logger.info("Writing metadata.")
    cluster_meta.to_csv(
        f"{resource_type}_{scenario}_cluster_metadata.csv", float_format="%.4f"
    )

    if create_profiles:
        logger.info("Loading site profiles")
        site_profiles = load_gen_profiles(
            tidy_clustered["Site"].unique(),
            resource_type=resource_type,
            variable=resource_variable[resource_type],
            scenario=scenario,
        )

        if resource_type == "solarpv":
            logger.info("Renormalizing solar profiles")
            site_profiles = renorm_solar(site_profiles)

        logger.info("Making cluster weighted profiles")
        cluster_profiles = make_weighted_profiles(
            tidy_clustered.set_index(
                ["IPM_Region", "cbsa_id", "cluster_level", "cluster"]
                + additional_group_cols
            )
            .loc[cluster_meta.index]
            .reset_index(),
            cluster_meta,
            site_profiles,
            additional_group_cols=additional_group_cols,
            n_jobs=n_jobs,
        )

        logger.info("Writing profiles.")
        if resource_type == "wind":
            cluster_profiles.round(4).to_parquet(
                f"{resource_type}_{scenario}_cluster_profiles.parquet",
                partition_cols=["IPM_Region"],
            )
        else:
            cluster_profiles.round(4).to_parquet(
                f"{resource_type}_{scenario}_cluster_profiles.parquet"
            )


# def write_weighted_avg_profiles(
#     name, variable, available_sites=None, source_folder="site_area",
#     max_area=10000, wind_density=2.7, solar_density=45*0.2
# ):
#     resource_density = {
#         "solarPV": 45*0.2,
#         "wind": 2.7
#     }
#     region = name.split(".")[0].split("_IPM_")[-1]
#     resource, scenario = name.split("_")[2:4]

#     area_clusters = assign_cluster_labels(
#         name, variable, keep_sites=available_sites, folder=source_folder, soft_max_area=max_area
#     )

#     try:
#         fn = f"cluster_assignments_{resource}_{scenario}_{region}.csv"
#         area_clusters[["cf", "km2", "cluster"]].to_csv(CWD / "cluster_assignments" / fn)

#         # hour_cols = [col for col in area_clusters.columns if isinstance(col, int)]
#         # cluster_cov = area_clusters[hour_cols].cov()

#         cluster_weighted_profiles = cluster_weighted_avg_profile(area_clusters)

#         fn = f"cluster_profiles_{resource}_{scenario}_{region}.csv" # name.replace(".geojson", ".csv")
#         cluster_weighted_profiles.to_csv(CWD / "cluster_profiles" / fn, index=False)

#         cluster_cf = pd.DataFrame(cluster_weighted_profiles.mean(), columns=["cf"])
#         cluster_cf["km2"] = area_clusters.groupby("cluster")["km2"].sum()
#         cluster_cf["GW"] = cluster_cf["km2"] * resource_density[resource] / 1000
#         cluster_cf["avg_std"] = area_clusters.groupby("cluster").apply(cluster_avg_std)
#         cluster_cf = cluster_cf.sort_values(by="cf", ascending=False)
#         cluster_cf["cumulative_GW"] = cluster_cf["GW"].cumsum()

#         fn = f"cluster_overview_{resource}_{scenario}_{region}.csv"
#         cluster_cf.to_csv(CWD / "cluster_overview" / fn)
#     except TypeError:
#         print(f"No {variable} clusters in {region}")


# def main(
#     wind_density: float = 2.7,
#     solar_density: float = 45 * 0.2,
#     max_wind_capacity_gw: float = 60,
#     max_solar_capacity_gw: float = 40,
#     run_wind: bool = True,
#     run_solar: bool = True
# ):
#     solar_interconnect_annuities = load_interconnect_annuities("solar")
#     solar_lcoe = calc_site_lcoe(SOLAR_ANNUITY, solar_interconnect_annuities)
#     solar_lcoe.to_csv("Solar_LCOE.csv", index=False)

#     solar_keep_list = []
#     for meta_region, gw_cap in SOLAR_META_CAP_GW.items():
#         _solar_keep = filter_sites_within_region(meta_region, gw_cap * 1000, solar_density, solar_lcoe)
#         solar_keep_list.append(_solar_keep)
#     keep_solar = pd.concat(solar_keep_list)

#     wind_interconnect_annuities = load_interconnect_annuities("wind")
#     wind_lcoe = calc_site_lcoe(WIND_ANNUITY, wind_interconnect_annuities)
#     wind_lcoe.to_csv("Wind_LCOE.csv", index=False)

#     wind_keep_list = []
#     for meta_region, gw_cap in WIND_META_CAP_GW.items():
#         _wind_keep = filter_sites_within_region(meta_region, gw_cap * 1000, wind_density, wind_lcoe)
#         wind_keep_list.append(_wind_keep)
#     keep_wind = pd.concat(wind_keep_list)

#     solar_files = (CWD / "site_area").glob("*solar*")
#     wind_files = (CWD / "site_area").glob("*wind*")

#     solar_area = max_solar_capacity_gw * 1000 / solar_density
#     wind_area = max_wind_capacity_gw * 1000 / wind_density

#     if run_solar:
#         Parallel(n_jobs=-2)(delayed(write_weighted_avg_profiles)
#             (
#                 name=path.name,
#                 variable="Axis1_SolarPV_Lat",
#                 available_sites=keep_solar["Site"],
#                 max_area=solar_area,

#             )
#             for path in solar_files if "WEC" in path.name
#         )

#     if run_wind:
#         Parallel(n_jobs=-2)(delayed(write_weighted_avg_profiles)
#             (
#                 name=path.name,
#                 variable="Wind_Power_100m",
#                 available_sites=keep_wind["Site"],
#                 max_area=wind_area,

#             )
#             for path in wind_files if "WEC" in path.name
#         )


# def plot_region_cluster_profile(region, cluster, resource, scenario, hours=24*14):
#     variable_dict = {
#         "wind": "Wind_Power_100m",
#         "solarPV": "Axis1_SolarPV_Lat"
#     }
#     cluster_assn_files = (CWD / "cluster_assignments").glob(f"*{region}*")
#     cluster_assn_fn = [fn for fn in cluster_assn_files if resource in fn.name and scenario in fn.name][0]

#     cluster_assignments = pd.read_csv(CWD / "cluster_assignments" / cluster_assn_fn, index_col=0)
#     sites = cluster_assignments.query("cluster==@cluster").index.to_list()
#     sites = [f"{str(s).zfill(6)}" for s in sites]

#     profiles = load_gen_profiles(sites, resource=resource, variable=variable_dict[resource])
#     # return profiles
#     profiles["Site"] = profiles.index
#     profiles_tidy = profiles.melt(id_vars="Site", var_name="hour", value_name="cf")

#     # profiles.T.loc[:hours, :].plot()
#     ax = sns.lineplot(x="hour", y="cf", data=profiles_tidy.query("hour<=@hours"), estimator=None, color="lightblue", lw=.25)
#     sns.lineplot(x="hour", y="cf", data=profiles_tidy.query("hour<=@hours"), estimator="mean", ci=None, color="k", ax=ax)
#     # ax = sns.lineplot(x="hour", y="cf", data=profiles_tidy.query("hour<=@hours"), ci="sd")

# cluster_overview_wecc_files = [fn for fn in (CWD / "cluster_overview").glob("*wind_base3k_WEC*")]
# for fn in cluster_overview_wecc_files:
#     try:
#         df = pd.read_csv(fn)
#         region = fn.name.split(".")[0].split("3k_")[-1]
#         total_gw = df["cumulative_GW"].max()
#         top_gw = df.query("cumulative_GW<=@total_gw/10").copy()
#         wm_cf = np.average(top_gw["cf"], weights=top_gw["GW"])
#         # wm_avg_std = np.average(top_gw["avg_std"], weights=top_gw["GW"])
#         print(f"{region}: {total_gw / 10:.1f} GW, {wm_cf:.1f}% CF, {len(top_gw)} clusters")
#     except ZeroDivisionError:
#         pass

# cluster_overview_wecc_files = [fn for fn in (CWD / "cluster_overview").glob("*solarPV_base_WEC*")]
# for fn in cluster_overview_wecc_files:
#     try:
#         df = pd.read_csv(fn)
#         region = fn.name.split(".")[0].split("base_")[-1]
#         total_gw = df["cumulative_GW"].max()
#         top_gw = df.query("cumulative_GW<=@total_gw/20").copy()
#         wm_cf = np.average(top_gw["cf"], weights=top_gw["GW"])
#         # wm_avg_std = np.average(top_gw["avg_std"], weights=top_gw["GW"])
#         print(f"{region}: {total_gw / 10:.1f} GW, {wm_cf:.1f}% CF, {len(top_gw)} clusters")
#     except ZeroDivisionError:
#         pass

# cluster_overview_wecc_files = [fn for fn in (CWD / "cluster_overview").glob("*wind_base3k_WEC*")]
# for fn in cluster_overview_wecc_files:
#     try:
#         df = pd.read_csv(fn)
#         region = fn.name.split(".")[0].split("3k_")[-1]
#         total_gw = df["cumulative_GW"].max()
#         top_gw = df.query("cumulative_GW<=@total_gw/10").copy()
#         wm_cf = np.average(top_gw["cf"], weights=top_gw["GW"])
#         # wm_avg_std = np.average(top_gw["avg_std"], weights=top_gw["GW"])
#         print(f"{region}: {total_gw / 10:.1f} GW, {wm_cf:.1f}% CF, {len(top_gw)} clusters")
#     except ZeroDivisionError:
#         pass

# cluster_overview_wecc_files = [fn for fn in (CWD / "cluster_overview").glob("*solarPV_base_WEC*")]
# for fn in cluster_overview_wecc_files:
#     try:
#         df = pd.read_csv(fn)
#         region = fn.name.split(".")[0].split("base_")[-1]
#         total_gw = df["cumulative_GW"].max()
#         # top_gw = df.query("cumulative_GW<=@total_gw/20").copy()
#         wm_cf = np.average(df["cf"], weights=df["GW"])
#         # wm_avg_std = np.average(top_gw["avg_std"], weights=top_gw["GW"])
#         print(f"{region}: {total_gw:.1f} GW, {wm_cf:.1f}% CF, {len(df)} clusters")
#     except ZeroDivisionError:
#         print(region)

# cluster_overview_wecc_files = [fn for fn in (CWD / "cluster_overview").glob("*wind_base3k_WEC*")]
# for fn in cluster_overview_wecc_files:
#     try:
#         df = pd.read_csv(fn)
#         region = fn.name.split(".")[0].split("base3k_")[-1]
#         total_gw = df["cumulative_GW"].max()
#         # top_gw = df.query("cumulative_GW<=@total_gw/20").copy()
#         wm_cf = np.average(df["cf"], weights=df["GW"])
#         # wm_avg_std = np.average(top_gw["avg_std"], weights=top_gw["GW"])
#         print(f"{region}: {total_gw:.1f} GW, {wm_cf:.1f}% CF, {len(df)} clusters")
#     except ZeroDivisionError:
#         pass


# def aggregate_results(resource, scenario, fn_filter=None):
#     profile_fns = (CWD / "cluster_profiles").glob(f"*{resource}_{scenario}*")
#     overview_fns = (CWD / "cluster_overview").glob(f"*{resource}_{scenario}*")
#     if fn_filter is not None:
#         profile_fns = [fn for fn in profile_fns if fn_filter in fn.name]
#         overview_fns = [fn for fn in overview_fns if fn_filter in fn.name]

#     profile_list = []
#     for fn in profile_fns:
#         region = fn.name.split(".")[0].split(f"{scenario}_")[-1]
#         df = pd.read_csv(fn)
#         dft = df.T
#         dft["cluster"] = dft.index
#         df_tidy = dft.melt(
#             id_vars="cluster", var_name="hour", value_name="cf"
#         ).sort_values(["cluster", "hour"])
#         df_tidy["region"] = region

#         profile_list.append(df_tidy)

#     profile_df = pd.concat(profile_list)
#     profile_df["cf"] = (profile_df["cf"] / 100).round(4)

#     profile_df.to_csv(f"wecc_{resource}_{scenario}_profiles.csv", index=False)

#     # return profile_df

#     overview_list = []
#     for fn in overview_fns:
#         region = fn.name.split(".")[0].split(f"{scenario}_")[-1]
#         df = pd.read_csv(fn)
#         df = df.rename(columns={"Unnamed: 0": "cluster"})
#         df["region"] = region

#         overview_list.append(df)

#     overview_df = pd.concat(overview_list)
#     overview_df["cf"] = (overview_df["cf"] / 100).round(4)
#     overview_df["km2"] = overview_df["km2"].round(3)
#     overview_df["GW"] = overview_df["GW"].round(3)
#     overview_df["avg_std"] = overview_df["avg_std"].round(3)
#     overview_df["cumulative_GW"] = overview_df["cumulative_GW"].round(3)

#     overview_df.to_csv(f"wecc_{resource}_{scenario}_overview.csv", index=False)


# # aggregate_results("solarPV", "base", fn_filter="WEC")
# # aggregate_results("wind", "base3k", fn_filter="WEC")


if __name__ == "__main__":
    typer.run(main)
#     aggregate_results("solarPV", "base", fn_filter="WEC")
#     aggregate_results("wind", "base3k", fn_filter="WEC")
