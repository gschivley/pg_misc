import copy
import re
import json
import logging
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from powergenome.resource_clusters import MERGE, build_tree, group_rows, prune_tree
import typer

RESOURCE_DENSITY = {"utilitypv": 45, "landbasedwind": 2.7, "fixed": 5, "floating": 8}
# Fraction of capacity to retain for each CPA based on population density. Based
# on data used in Princeton Net Zero study.
RESOURCE_POP_DEN_RETAIN = {
    "utilitypv": {
        "pop_den_(0-4]": 1,
        "pop_den_(4-12]": 1,
        "pop_den_(12-25]": 1 - 0.8,
        "pop_den_25+": 1 - 0.8,
    },
    "landbasedwind": {
        "pop_den_(0-4]": 1,
        "pop_den_(4-12]": 1 - 0.9,
        "pop_den_(12-25]": 1 - 0.9,
        "pop_den_25+": 1 - 0.9,
    },
}
GROUP_COLS = {"offshorewind": ["turbine_type", "pref_site"]}
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
formatter = logging.Formatter(
    # More extensive test-like formatter...
    "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s",
    # This is the datetime format string.
    "%Y-%m-%d %H:%M:%S",
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
LOGGER.addHandler(handler)


def snake(s: str) -> str:
    """
    Convert variable name to snake case.
    Examples
    --------
    >>> snake('Area')
    'area'
    >>> snake('turbineType')
    'turbine_type'
    >>> snake('MW')
    'mw'
    >>> snake('IPM_Region')
    'ipm_region'
    >>> snake('d_coast_sub_161kVplus_miles')
    'd_coast_sub_161kvplus_miles'
    """
    if "_" in s:
        return s.lower()
    return re.sub(r"([a-z])([A-Z])", r"\1_\2", s).lower()


def snake_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names to snake case.
    Examples
    --------
    >>> df = pd.DataFrame([{'Area': 1, 'turbineType': True}])
    >>> snake_columns(df)
       area  turbine_type
    0     1          True
    """
    return df.rename(columns={name: snake(name) for name in df.columns})


def set_transmission(df: pd.DataFrame, technology: str) -> pd.DataFrame:
    if technology == "offshorewind":
        df["spur_miles"] = df["d_coast_sub_161kvplus_miles"]
        df["offshore_spur_miles"] = df["d_coast_miles"]
        df["tx_miles"] = df["d_sub_load_metro_750k_center_miles"]
        return df
    df["spur_miles"] = df["site_substation_spur_miles"]
    df["tx_miles"] = df["substation_metro_tx_miles"]
    indirect_capex = df[["site_substation_capex", "substation_metro_capex"]].sum(axis=1)
    mask = df["metro_direct_capex"] < indirect_capex
    df.loc[mask, "spur_miles"] = df["site_metro_spur_miles"]
    df.loc[mask, "tx_miles"] = 0
    return df


def set_capacity(df: pd.DataFrame, technology: str) -> pd.DataFrame:
    if technology == "offshorewind":
        density = df["turbine_type"].map(RESOURCE_DENSITY)
    else:
        density = RESOURCE_DENSITY[technology]
        retain_map = RESOURCE_POP_DEN_RETAIN[technology]

    df["mw"] = df["area"] * density

    if "m_popden" in df.columns:
        pop_den_bins = [0, 4, 12, 25, 200]
        pop_bin_names = [
            "pop_den_(0-4]",
            "pop_den_(4-12]",
            "pop_den_(12-25]",
            "pop_den_25+",
        ]
        df["pop_density_bin"] = pd.cut(
            df["m_popden"], pop_den_bins, include_lowest=True, labels=pop_bin_names
        )
        df["mw"] *= df["pop_density_bin"].astype(str).map(retain_map)
    return df


def load_cpa_data(path: str, technology: str) -> pd.DataFrame:
    cpa_data = (
        pd.read_csv(path, dtype={"Site": str, "metro_id": str, "prefSite": bool})
        .pipe(snake_columns)
        .pipe(set_capacity, technology)
        .pipe(set_transmission, technology)
    )

    cpa_data["site"] = cpa_data["site"].str.zfill(6)

    if technology != "offshorewind":
        cpa_data = cpa_data.rename(
            columns={
                "contains_existingwind": "contains_existingfacil",
                "contains_plannedwind": "contains_plannedfacil",
            }
        )
        cpa_data = cpa_data.query(
            "contains_existingfacil == 0 & contains_plannedfacil == 0"
        )
    return cpa_data


def load_profiles(path: str, technology: str) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="fastparquet")
    if technology == "utilitypv":
        # DC capacity is higher than AC capacity. 1.34 value from NREL ATB 2020
        df *= 1.34
        # Clip values over 100
        df = df.where(df <= 100, 100)
    # Scale capacity factors from 0-100 to 0-1
    return (df * 0.01).round(4)


def build_additional_metadata(
    cpa_data: pd.DataFrame, site_cluster: pd.DataFrame
) -> pd.DataFrame:
    pop_den_bins = [0, 4, 12, 25, 200]
    pop_bin_names = [
        "pop_den_(0-4]",
        "pop_den_(4-12]",
        "pop_den_(12-25]",
        "pop_den_25+",
    ]
    cpa_data["pop_density_bin"] = pd.cut(
        cpa_data["m_popden"], pop_den_bins, include_lowest=True, labels=pop_bin_names
    )
    pop_den_cats = cpa_data.pop_density_bin.unique().categories

    # landcover_labels = [f"lc_{i}" for i in sorted(metadata.landcover.unique())]

    pop_den_groups_df = pd.DataFrame(
        index=site_cluster.index.to_list(), columns=pop_den_cats, dtype=np.float32
    )
    prime_farmland_df = pd.DataFrame(
        index=site_cluster.index.to_list(), columns=["prime_farmland"], dtype=np.float32
    )
    landcover_df = pd.DataFrame(
        index=site_cluster.index.to_list(),
        columns=sorted(cpa_data.landcover.unique()),
        dtype=np.float32,
    )
    hmi_list = []
    for row in site_cluster.itertuples():
        cluster_id = row.Index
        cluster_cpas = row.cpa_id
        cluster_cpa_data = cpa_data.query("cpa_id in @cluster_cpas")
        cluster_area = cluster_cpa_data["area"].sum()

        cluster_cpa_popden_grouped = cluster_cpa_data.groupby("pop_density_bin")
        pop_density_area = cluster_cpa_popden_grouped["area"].sum() / cluster_area
        pop_den_groups_df.loc[cluster_id, :] = pop_density_area

        cluster_prime_farm_area = (
            cluster_cpa_data.query("contains_prime_farmland == 1")["area"].sum()
            / cluster_area
        )
        prime_farmland_df.loc[cluster_id, "prime_farmland"] = cluster_prime_farm_area

        landcover_area = (
            cluster_cpa_data.groupby("landcover")["area"].sum() / cluster_area
        )
        landcover_df.loc[cluster_id, :] = landcover_area

        hmi_list.append(
            (cluster_cpa_data["hmi"] * cluster_cpa_data["area"]).sum() / cluster_area
        )

    landcover_df.columns = [f"lc_{i}" for i in landcover_df.columns]
    add_meta_df = pd.concat(
        [pop_den_groups_df, prime_farmland_df, landcover_df], axis=1
    )
    add_meta_df["hmi"] = hmi_list

    return add_meta_df


def main(
    technology: str,
    cpa_data_path: str,
    profiles_path: str = None,
    existing: bool = False,
    min_relative_rmse: float = 0.025,
    min_mw: float = 100,
    max_level: int = 50,
    fn_prefix: str = "",
):
    """Cluster candidate project areas (CPAs) within each metro area using hierarchical
    clustering. Multiple levels of clusters (e.g. from 1 cluster to N clusters) are
    calculated, and only unique clusters are retained. Because heirarchical clustering
    is used, a deterministic parent-child relationship can be established.

    Clustering is done based on LCOE, which is used as an input to this script. The
    cluster metadata, cluster hourly profiles, and sites within each cluster are all
    saved.

    The capacity (`mw`) for each technology is calculated by multiplying the area by
    values from `RESOURCE_DENSITY`. IMPORTANT: `utilitypv` capacity assumes 100%
    coverage by solar panels, and these capacity values should be discounted by the
    user.

    Parameters
    ----------
    technology : str
        Name of the technology. Should be one of `utilitypv`, `landbasedwind`, or
        `offshorewind`. Names match the technology types from NREL ATB.
    metadata_path : str
        Path the the csv file with LCOE and spur line costs for each CPA. Also includes
        a site ID for each CPA, which is used to assign an hourly profile.
    profiles_path : str, optional
        Path to the parquet files with hourly profiles for each site, by default None.
    existing : bool, optional
        If the resource is new or already existing, by default False
    min_relative_rmse : float, optional
        A filter to determine the maximum acceptable relative RMSE in LCOE, by default
        0.025. If all clusters within level `n` have RMSE lower than this value, then
        levels `n+1` and higher are discarded. Since solar will always have zero
        generation at night, this value should probably be lower for solar than wind.
    min_mw : float, optional
        An additional filter based on the capacity of each cluster, by default 100. If
        all clusters within level `n` have capacity lower than this value, then levels
        `n+1` and higher are discarded.
    max_level : int, optional
        The number of cluster levels to calculate, by default 50. Level `n` will have
        `n` possible clusters in it.
    fn_prefix : str, optional
        The prefix to use for filenames when saving results, by default "".
    """
    # Load resource metadata
    LOGGER.info("Loading resource project area data")
    cpa_data = load_cpa_data(cpa_data_path, technology=technology)
    cpa_data.index = cpa_data["cpa_id"]
    # Prepare merge parameters
    merge = copy.deepcopy(MERGE)
    if profiles_path is not None:
        LOGGER.info("Loading resource profiles")
        profiles = load_profiles(profiles_path, technology=technology)
        merge["means"].append("profile")
    # Split resources into resource groups
    if GROUP_COLS.get(technology):
        group_cols = GROUP_COLS[technology]
        groups = []
        for idx, df in cpa_data.groupby(group_cols, as_index=False):
            group = {key: value for key, value in zip(group_cols, idx)}
            groups.append((group, df.drop(columns=group_cols)))
    else:
        groups = [({}, cpa_data)]
    # Process each group in series
    for group, df in groups:
        LOGGER.info(f"Processing resource group: {group}")
        metros = df.groupby(["metro_id"])
        i = 0
        trees = []
        for metro_id, rows in metros:
            LOGGER.info(f"Building resource tree for metro_id: {metro_id}")
            if profiles_path is not None:
                # Load profiles into dataframe
                rows["profile"] = list(profiles[rows["site"]].values.T)
            tree = build_tree(rows, by=rows[["lcoe"]], max_level=max_level, **merge)
            # Prune base levels with no clusters meeting the criteria
            grouped_rows = group_rows(rows[["lcoe", "mw"]], tree.index)
            relative_rmse = []
            for lcoe, (_, cluster) in zip(tree["lcoe"], grouped_rows):
                weights = cluster["mw"] / cluster["mw"].sum()
                # NOTE: Original code used MSE, not RMSE
                rmse = np.sqrt(np.sum(weights * (cluster["lcoe"] - lcoe) ** 2))
                relative_rmse.append(rmse / lcoe)
            keep = np.greater(relative_rmse, min_relative_rmse) & (tree["mw"] > min_mw)
            # from IPython import embed
            # embed()
            # tree["rmse"] = relative_rmse[keep]
            if len(tree) > 1:
                tree = prune_tree(tree, level=tree["level"][keep].max())
            else:
                tree["id"] = 0
                tree["parent_id"] = np.nan
                tree["level"] = 1
            # Ensure ids are unique across resource group
            tree[["id", "parent_id"]] += i
            i += len(tree)
            trees.append(tree)
        LOGGER.info(f"Writing output for resource group: {group}")
        # Join trees together
        clusters = pd.concat(trees, ignore_index=True)
        # Create a mapping of CPA IDs to cluster IDs
        site_cluster = (
            pd.concat(trees)
            .reset_index()
            .rename(columns={"index": "cpa_id"})[["cpa_id", "id"]]
            .set_index("id")
        )
        df_list = []
        for row in site_cluster.itertuples():
            _df = pd.DataFrame(
                {"cpa_id": row.cpa_id, "id": [row.Index] * len(row.cpa_id)}
            )
            #     _df["id"] = idx
            df_list.append(_df)
        cluster_assignments = pd.concat(df_list, ignore_index=True, sort=False)

        if technology != "offshorewind":
            extra_metadata = build_additional_metadata(cpa_data, site_cluster)
            clusters = pd.concat([clusters, extra_metadata], axis=1)

        # Build group metadata
        group = {
            "technology": technology,
            "tree": "metro_id",
            "existing": existing,
            **group,
        }
        basename = fn_prefix + "_".join([str(v) for v in group.values()])
        group_path = basename + "_group.json"
        group["metadata"] = basename + "_metadata.csv"
        group["site_cluster"] = basename + "_site_cluster.csv"
        if profiles_path is not None:
            group["profiles"] = basename + "_profiles.parquet"
        # Write clustered resource metadata
        if profiles_path:
            # clusters = clusters.drop(columns="profile")
            clusters.drop(columns="profile").to_csv(
                group["metadata"],
                float_format="%.4f",
                index=False,
            )
        else:
            clusters.to_csv(
                group["metadata"],
                float_format="%.4f",
                index=False,
            )
        # Write site/cluster mappings
        cluster_assignments.to_csv(group["site_cluster"], index=False)
        # Write clustered resource profiles
        if profiles_path is not None:
            pd.DataFrame(
                np.column_stack(clusters["profile"]),
                # to_parquet requires string column names
                columns=pd.RangeIndex(len(clusters)).astype(str),
            ).to_parquet(group["profiles"])
        # Write group metadata
        with open(group_path, "w") as fp:
            json.dump(group, fp, indent=4)


if __name__ == "__main__":
    typer.run(main)
