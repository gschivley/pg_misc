import numpy as np
import netCDF4
import pandas as pd
import geopandas as gpd
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


def load_gen_profiles(site_list, resource_type, variable, scenario, region_filter=None):
    if region_filter:
        region_filter = region_filter + "_"
    else:
        region_filter = ""

    if resource_type.lower() == "wind":
        resource = "Wind"
        resource_path = VCE_WIND_PATH
    elif resource_type.lower() == "offshorewind":
        resource = "Wind"
        resource_path = VCE_WIND_PATH
    elif resource_type.lower() == "solarpv":
        resource = "SolarPV"
        resource_path = VCE_SOLAR_PATH

    fn = f"{region_filter}{scenario}_{resource_type}_site_profiles.parquet"
    if Path(fn).exists():
        logger.info("Profiles already saved as parquet file")
        df = pd.read_parquet(fn)
        logger.info("Downcasting data")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        logger.info("Downcasting complete")

    else:
        logger.info("Loading all profiles from .nc4")
        site_profiles = {}
        for s in site_list:
            fpath = f"Site_{s}_{resource}.nc4"
            site_data = netCDF4.Dataset(resource_path / fpath)
            gen_profile = np.array(site_data[variable]).astype(np.float32)
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
    cpa_lcoe = pd.read_csv(path, dtype={"Site": str, "metro_id": str})
    # cpa_lcoe = gpd.read_parquet(path)
    # cpa_lcoe["Site"] = cpa_lcoe["Site"].astype(str).str.zfill(6)

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
        "MW",
        "state",
        "id",
    ]
    keep_cols = [col for col in keep_cols if col in cluster_df.columns]
    id_vars = (
        additional_cluster_cols
        + ["IPM_Region", "metro_id", "cpa_id", "Site", "lcoe", "Area",]
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
    tidy_clustered, additional_group_cols=[], relative_rmse_filter=0.025, mw_filter=500,
):
    group_cols = [
        "IPM_Region",
        "metro_id",
        "cluster_level",
        "cluster",
    ] + additional_group_cols
    logger.info("weighted lcoe")
    clustered_meta = tidy_clustered.groupby(group_cols).apply(
        wa_column, value_col="lcoe", weight_col="Area"
    )
    clustered_meta.columns = ["lcoe"]

    sum_cols = ["Area", "MW"]
    clustered_meta[sum_cols] = tidy_clustered.groupby(group_cols)[sum_cols].sum()

    avg_std_capacity = (
        clustered_meta.reset_index()
        .groupby(["metro_id", "cluster_level"], as_index=False)["MW"]
        .sum()
        .groupby("metro_id")["MW"]
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
        | (clustered_meta["MW"] <= mw_filter),
        "meets_criteria",
    ] = True

    logger.info("Filtering metadata clusters")
    df_list = []
    for _, _df in clustered_meta.groupby(
        ["IPM_Region", "metro_id", "cluster_level"] + additional_group_cols
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
        "metro_id",
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
        "metro_id",
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
        cpa_lcoe["MW"] = cpa_lcoe["Area"] * cpa_lcoe["turbineType"].map(
            resource_density
        )
    else:
        cpa_lcoe["MW"] = cpa_lcoe["Area"] * resource_density[resource_type]

    return cpa_lcoe


def format_metadata(df, by="lcoe"):
    # Initialize sequential unique cluster id
    cluster_id = 1
    all_clusters = []
    for metro in df["metro_id"].unique():
        sdf = df[df["metro_id"] == metro]
        levels = sdf["cluster_level"].drop_duplicates().sort_values(ascending=False)
        # Start from base clusters
        clusters = sdf[sdf["cluster_level"] == levels.max()].to_dict(orient="records")
        for i, x in enumerate(clusters, start=cluster_id):
            x["id"] = i
        for level in levels[1:]:
            parent_id = cluster_id + len(clusters)
            new = sdf[sdf["cluster_level"] == level]
            # Old cluster is a child if:
            # - not already assigned to a parent
            # - capacity not duplicated in current cluster level
            children = [
                x
                for x in clusters
                if not x.get("parent_id") and (x[by] != new[by]).all()
            ]
            if len(children) != 2:
                raise ValueError(
                    f"Found {len(children)} children for level {level} in metro_id {metro}"
                )
            for x in children:
                x["parent_id"] = parent_id
            # New cluster is a parent if:
            # - capacity not present in previous cluster level
            is_parent = ~new[by].isin(sdf[sdf["cluster_level"] == level + 1][by])
            if sum(is_parent) == 1:
                parent = new[is_parent].iloc[0].to_dict()
                parent["id"] = parent_id
            else:
                raise ValueError(
                    f"Found {sum(is_parent)} parents at level {level} in metro_id {metro}"
                )
            clusters.append(parent)
        all_clusters.extend(clusters)
        cluster_id += len(clusters)
    return pd.DataFrame(all_clusters)


def main(
    lcoe_path,
    resource_type="solarpv",
    scenario="base",
    relative_rmse_filter: float = 0.025,
    mw_filter: float = 500,
    create_profiles: bool = True,
    write_site_labels: bool = True,
    n_jobs: int = -2,
    max_cluster_levels: int = 50,
    region_filter=None,
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
        mw_filter = 2500

    cpa_lcoe = (
        load_lcoe_data(lcoe_path)
        .pipe(set_final_spur_columns, resource_type=resource_type)
        .pipe(
            set_cpa_capacity,
            resource_type=resource_type,
            resource_density=resource_density,
        )
    )
    if region_filter:
        logger.info(f"Regional filter {region_filter} applied.")
        cpa_lcoe = cpa_lcoe.loc[cpa_lcoe.IPM_Region.str.contains(region_filter), :]
        region_filter = region_filter + "_"
    else:
        region_filter = ""

    if resource_type == "offshorewind":
        cpa_lcoe = cpa_lcoe.loc[
            (cpa_lcoe.turbineType == "fixed") & (cpa_lcoe.prefSite == 0), :
        ]

    logger.info("LCOE loaded, calculating cluster labels")
    cpa_lcoe_cluster_labels = cpa_lcoe.groupby(
        ["IPM_Region", "metro_id"] + additional_group_cols
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
        mw_filter=mw_filter,
    )
    cols = cluster_meta.index.names
    cluster_meta = cluster_meta.reset_index().pipe(format_metadata)

    logger.info("Writing metadata.")
    cluster_folder = CWD.parent / "cluster_data"
    cluster_folder.mkdir(exist_ok=True)
    cluster_meta.to_csv(
        cluster_folder
        / f"{region_filter}{resource_type}_{scenario}_cluster_metadata.csv",
        float_format="%.4f",
    )

    # HACK: Use clusters.format_metadata to drop duplicate clusters
    cluster_meta = cluster_meta.set_index(cols)
    tidy_clustered = (
        tidy_clustered.set_index(cols).loc[cluster_meta.index].reset_index()
    )
    tidy_clustered = tidy_clustered.merge(
        cluster_meta.reset_index()[["cluster_level", "cluster", "id"]],
        on=["cluster_level", "cluster"],
    )
    tidy_clustered.to_csv(
        cluster_folder / f"{region_filter}{resource_type}_{scenario}_site_metadata.csv",
        float_format="%.4f",
        index=False,
    )

    if create_profiles:
        logger.info("Loading site profiles")
        site_profiles = load_gen_profiles(
            tidy_clustered["Site"].unique(),
            resource_type=resource_type,
            variable=resource_variable[resource_type],
            scenario=scenario,
            region_filter=region_filter
        )

        if resource_type == "solarpv":
            logger.info("Renormalizing solar profiles")
            site_profiles = renorm_solar(site_profiles)

        logger.info("Making cluster weighted profiles")
        cluster_profiles = make_weighted_profiles(
            tidy_clustered.set_index(
                ["IPM_Region", "metro_id", "cluster_level", "cluster"]
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
        print(cluster_profiles.head())
        if resource_type == "wind":
            cluster_profiles.round(4).to_parquet(
                cluster_folder
                / f"{region_filter}{resource_type}_{scenario}_cluster_profiles.parquet",
                partition_cols=["IPM_Region"],
            )
        else:
            cluster_profiles.round(4).to_parquet(
                cluster_folder
                / f"{region_filter}{resource_type}_{scenario}_cluster_profiles.parquet"
            )


if __name__ == "__main__":
    typer.run(main)
