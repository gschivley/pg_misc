
import itertools
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy
import logging

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

WEIGHT = "mw"
MEANS = [
    "lcoe",
    "interconnect_annuity",
    "offshore_spur_miles",
    "spur_miles",
    "tx_miles",
    "site_substation_spur_miles",
    "substation_metro_tx_miles",
    "site_metro_spur_miles",
]
SUMS = ["area", "mw"]
PROFILE_KEYS = ["metro_id", "cluster_level", "cluster"]
HOURS_IN_YEAR = 8784


def format_metadata(df, cap_multiplier=None, by="mw"):
    # Convert all column names to lowercase
    df.columns = [name.lower() for name in df.columns]
    if cap_multiplier:
        df["mw"] = df["mw"] * cap_multiplier
    # Initialize sequential unique cluster id
    # cluster_id = 1
    # all_clusters = []
    # for metro in df["metro_id"].unique():
    #     sdf = df[df["metro_id"] == metro]
    #     levels = sdf["cluster_level"].drop_duplicates().sort_values(ascending=False)
    #     # Start from base clusters
    #     clusters = sdf[sdf["cluster_level"] == levels.max()].to_dict(orient="records")
    #     for i, x in enumerate(clusters, start=cluster_id):
    #         x["id"] = i
    #     for level in levels[1:]:
    #         parent_id = cluster_id + len(clusters)
    #         new = sdf[sdf["cluster_level"] == level]
    #         # Old cluster is a child if:
    #         # - not already assigned to a parent
    #         # - capacity not duplicated in current cluster level
    #         children = [
    #             x
    #             for x in clusters
    #             if not x.get("parent_id") and (x[by] != new[by]).all()
    #         ]
    #         if len(children) != 2:
    #             raise ValueError(
    #                 f"Found {len(children)} children for level {level} in metro_id {metro}"
    #             )
    #         for x in children:
    #             x["parent_id"] = parent_id
    #         # New cluster is a parent if:
    #         # - capacity not present in previous cluster level
    #         is_parent = ~new[by].isin(sdf[sdf["cluster_level"] == level + 1][by])
    #         if sum(is_parent) == 1:
    #             parent = new[is_parent].iloc[0].to_dict()
    #             parent["id"] = parent_id
    #         else:
    #             raise ValueError(
    #                 f"Found {sum(is_parent)} parents at level {level} in metro_id {metro}"
    #             )
    #         clusters.append(parent)
    #     all_clusters.extend(clusters)
    #     cluster_id += len(clusters)
    # return pd.DataFrame(all_clusters).set_index("id", drop=False)
    return df.set_index("id", drop=False)


def format_profiles_inplace(df):
    # Convert column names to lowercase
    df.columns = [name.lower() for name in df.columns]
    # Prepare index
    # NOTE: Assumes already sorted by hour (ascending)
    df.set_index(PROFILE_KEYS, inplace=True)


def build_clusters(metadata, ipm_regions, min_capacity=None, max_clusters=np.inf):
    if max_clusters < 1:
        raise ValueError("Max clusters must be greater than zero")
    df = metadata
    cdf = _get_base_clusters(df, ipm_regions).sort_values("lcoe")
    if min_capacity:
        # Drop clusters with highest LCOE until min_capacity reached
        end = cdf["mw"].cumsum().searchsorted(min_capacity) + 1
        if end > len(cdf):
            logger.warning(
                f"Capacity in {ipm_regions} ({cdf['mw'].sum()} MW) less than minimum "
                f"({min_capacity} MW). Using all available capacity instead."
            )
        else:
            cdf = cdf[:end]
    # Track ids of base clusters through aggregation
    cdf["ids"] = [[x] for x in cdf["id"]]
    # Aggregate clusters within each metro area (metro_id)
    while len(cdf) > max_clusters:
        # Sort parents by lowest LCOE distance of children
        diff = lambda x: abs(x.max() - x.min())
        parents = (
            cdf.groupby("parent_id", sort=False)
            .agg(child_ids=("id", list), n=("id", "count"), lcoe=("lcoe", diff))
            .sort_values(["n", "lcoe"], ascending=[False, True])
        )
        if parents.empty:
            break
        if parents["n"].iloc[0] == 2:
            # Choose parent with lowest LCOE
            best = parents.iloc[0]
            # Compute parent
            parent = pd.Series(
                _merge_children(
                    cdf.loc[best["child_ids"]],
                    ids=_flat(*cdf.loc[best["child_ids"], "ids"]),
                    **df.loc[best.name],
                )
            )
            # Add parent
            cdf.loc[best.name] = parent
            # Drop children
            cdf.drop(best["child_ids"], inplace=True)
        else:
            # Promote child with deepest parent
            parent_id = df.loc[parents.index, "cluster_level"].idxmax()
            parent = df.loc[parent_id]
            child_id = parents.loc[parent_id, "child_ids"][0]
            # Update child
            columns = ["id", "parent_id", "cluster_level"]
            cdf.loc[child_id, columns] = parent[columns]
            # Update index
            cdf.rename(index={child_id: parent_id}, inplace=True)
    # Keep only computed columns
    columns = _flat(MEANS, SUMS, "ids")
    columns = [col for col in columns if col in cdf.columns]
    cdf = cdf[columns]
    cdf.reset_index(inplace=True, drop=True)
    if len(cdf) > max_clusters:
        # Aggregate singleton metro area clusters
        Z = scipy.cluster.hierarchy.linkage(cdf[["lcoe"]].values, method="ward")
        cdf["_keep"] = True
        for child_idx in Z[:, 0:2].astype(int):
            cdf.loc[child_idx, "_keep"] = False
            parent = _merge_children(
                cdf.loc[child_idx], _keep=True, ids=_flat(*cdf.loc[child_idx, "ids"])
            )
            cdf.loc[len(cdf)] = parent
            if not cdf["_keep"].sum() > max_clusters:
                break
        cdf = cdf[cdf["_keep"]]
    return cdf[columns]


def build_cluster_profiles(clusters, metadata, profiles):
    results = np.zeros((HOURS_IN_YEAR, len(clusters)), dtype=float)
    for i, ids in enumerate(clusters["ids"]):
        idx = [tuple(x) for x in metadata.loc[ids, profiles.index.names].values]
        capacities = profiles.loc[idx, "capacity_factor"].values.reshape(
            HOURS_IN_YEAR, -1, order="F"
        )
        weights = metadata.loc[ids, "mw"].values
        weights /= weights.sum()
        results[:, i] = (capacities * weights).sum(axis=1)
    return results


def clusters_to_capacity_transmission_table(clusters, region, technology):
    columns = [
        "mw",
        "area",
        "lcoe",
        "interconnect_annuity",
        "spur_miles",
        "tx_miles",
        "site_metro_spur_miles",
        "site_substation_spur_miles",
        "substation_metro_tx_miles",
        "ids",
    ]
    columns = [col for col in columns if col in clusters.columns]
    return (
        clusters[columns]
        .assign(region=region, technology=technology, cluster=clusters.index)
        .reset_index(drop=True)
        .rename(columns={"mw": "max_capacity"})
        .rename(columns={"area": "area_km2"})
    )


def clusters_to_variable_resource_profiles_table(
    clusters, cluster_profiles, region, technology
):
    n_hours, n_clusters = cluster_profiles.shape
    cols = {}
    cols[("region", "Resource", "cluster")] = np.arange(n_hours) + 1
    for i in range(n_clusters):
        cols[(region, technology, clusters.index[i])] = cluster_profiles[:, i]
    return pd.DataFrame(cols)


def _get_base_clusters(df, ipm_regions):
    return (
        df[df["ipm_region"].isin(ipm_regions)]
        .groupby("metro_id")
        .apply(lambda g: g[g["cluster_level"] == g["cluster_level"].max()])
        .reset_index(level=["metro_id"], drop=True)
    )


def _merge_children(df, **kwargs):
    parent = kwargs
    for key in SUMS:
        parent[key] = df[key].sum()
    for key in [k for k in MEANS if k in df.columns]:
        parent[key] = (df[key] * df[WEIGHT]).sum() / df[WEIGHT].sum()
    return parent


def _flat(*args):
    args = [x if np.iterable(x) and not isinstance(x, str) else [x] for x in args]
    return list(itertools.chain(*args))
