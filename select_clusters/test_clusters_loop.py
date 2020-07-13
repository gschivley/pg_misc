import logging
import pandas as pd
from joblib import Parallel, delayed
from IPython import embed as IP
from clusters import (
    format_metadata,
    format_profiles_inplace,
    build_clusters,
    build_cluster_profiles,
    clusters_to_capacity_transmission_table,
    clusters_to_variable_resource_profiles_table,
)

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

# ---- Constants ----

UTC_OFFSET = 8

METADATA_PATHS = {
    "OnshoreWind": "../cluster_data/WECC_wind_base_cluster_metadata.csv",
    "UtilityPV": "../cluster_data/WECC_solarpv_base_cluster_metadata.csv",
    "OffShoreWind": "../cluster_data/WECC_offshorewind_base_cluster_metadata.csv",
}

PROFILE_PATHS = {
    "OnshoreWind": "../cluster_data/WECC_wind_base_cluster_profiles.parquet",
    "UtilityPV": "../cluster_data/WECC_solarpv_base_cluster_profiles.parquet",
    "OffShoreWind": "../cluster_data/WECC_offshorewind_base_cluster_profiles.parquet",
}

CAPACITY_MULTIPLIER = {
    "OnshoreWind": 1,
    "UtilityPV": 0.2,
    "OffShoreWind": 1,
}

SCENARIOS = {
    "UtilityPV": {
        "CA_N": {
            "ipm_regions": ["WEC_CALN", "WECC_BANC"],
            "max_clusters": 4,
            "min_capacity": 200000,
        },
        "CA_S": {
            "ipm_regions": ["WEC_SCE", "WEC_LADW", "WECC_SCE", "WEC_SDGE", "WECC_IID"],
            "max_clusters": 5,
            "min_capacity": 150000,
        },
        "WECC_N": {
            "ipm_regions": ["WECC_ID", "WECC_MT", "WECC_NNV", "WECC_UT"],
            "max_clusters": 20,
            "min_capacity": 250000,
        },
        "WECC_NMAZ": {
            "ipm_regions": ["WECC_AZ", "WECC_NM", "WECC_SNV",],
            "max_clusters": 15,
            "min_capacity": 300000,
        },
        # "WECC_PNW": {
        #     "ipm_regions": [],
        #     "max_clusters": 1,
        #     "min_capacity": 4.5,
        # },
        "WECC_WYCO": {
            "ipm_regions": ["WECC_WY", "WECC_CO"],
            "max_clusters": 15,
            "min_capacity": 180000,
        },
    },
    "OnshoreWind": {
        "CA_N": {
            "ipm_regions": ["WEC_CALN", "WECC_BANC"],
            "max_clusters": 1,
            "min_capacity": 22000,
        },
        "CA_S": {
            "ipm_regions": ["WEC_SCE", "WEC_LADW", "WECC_SCE", "WEC_SDGE", "WECC_IID"],
            "max_clusters": 10,
            "min_capacity": 45000,
        },
        "WECC_N": {
            "ipm_regions": ["WECC_ID", "WECC_MT", "WECC_NNV", "WECC_UT"],
            "max_clusters": 15,
            "min_capacity": 100000,
        },
        "WECC_NMAZ": {
            "ipm_regions": ["WECC_AZ", "WECC_NM", "WECC_SNV"],
            "max_clusters": 15,
            "min_capacity": 120000,
        },
        # "WECC_PNW": {
        #     "ipm_regions": [],
        #     "max_clusters": 1,
        #     "min_capacity": 6,
        # },
        "WECC_WYCO": {
            "ipm_regions": ["WECC_WY", "WECC_CO"],
            "max_clusters": 20,
            "min_capacity": 200000,
        },
    },
    "OffShoreWind": {
        "CA_N": {
            "ipm_regions": ["WEC_CALN", "WECC_BANC"],
            "max_clusters": 3,
            "min_capacity": 35000,
        },
        "WECC_PNW": {
            "ipm_regions": ["WECC_PNW"],
            "max_clusters": 3,
            "min_capacity": 26000,
        },
    },
}


def make_region_data(technology, region):

    print(f"{technology}, {region}")

    # Prepare resource metadata and profiles
    if tech == "OffShoreWind":
        metadata = (
            pd.read_csv(METADATA_PATHS[technology], dtype={"metro_id": str})
            .query("prefSite==1 & turbineType=='floating'")
            .pipe(
                format_metadata,
                cap_multiplier=CAPACITY_MULTIPLIER.get(technology),
                by="lcoe",
            )
        )
    else:
        metadata = pd.read_csv(METADATA_PATHS[technology], dtype={"metro_id": str}).pipe(
            format_metadata,
            cap_multiplier=CAPACITY_MULTIPLIER.get(technology),
            by="lcoe",
        )
    if tech == "OnshoreWind":
        filters = [
            [("IPM_Region", "=", r)]
            for r in SCENARIOS[technology][region]["ipm_regions"]
        ]
        profiles = pd.read_parquet(
            PROFILE_PATHS[technology], use_pandas_metadata=True, filters=filters
        )
    elif tech == "OffShoreWind":
        profiles = pd.read_parquet(PROFILE_PATHS[technology]).query(
            "prefSite==1 & turbineType=='floating'"
        )
    else:
        profiles = pd.read_parquet(PROFILE_PATHS[technology])
    format_profiles_inplace(profiles)

    # Build clusters
    scenario = SCENARIOS[technology][region]
    clusters = build_clusters(metadata, **scenario)

    # Build cluster profiles
    cluster_profiles = build_cluster_profiles(clusters, metadata, profiles)

    # Export results as PowerGenome input
    tab1 = clusters_to_capacity_transmission_table(clusters, region, technology)
    # tab1["max_capacity"] = tab1["max_capacity"] * CAPACITY_MULTIPLIER[technology]
    # spur_line = pd.concat([spur_line, tab1], ignore_index=True)

    tab2 = clusters_to_variable_resource_profiles_table(
        clusters, cluster_profiles, region, technology
    )
    # col = tab2[("region", "Resource", "cluster")]
    tab2 = tab2.drop([("region", "Resource", "cluster")], axis=1)
    # resource_variability = pd.concat([resource_variability, tab2], axis=1)

    return tab1, tab2


def remove_feb_29(df, offset):
    df.index = pd.date_range(start="2012-01-01", freq="H", periods=8784)

    df = df.loc[~((df.index.month == 2) & (df.index.day == 29)), :]

    wrap_rows = df.iloc[:offset, :]

    shifted_wrapped_df = pd.concat([df.iloc[offset:, :], wrap_rows], ignore_index=True)

    return shifted_wrapped_df.reset_index(drop=True)


PARALLEL = False
USE_IP = False
# ---- Processing ----
spur_line = pd.DataFrame()
resource_variability = pd.DataFrame()
cap_tx = []
profiles = []

for tech in SCENARIOS:
    technology = tech  # "UtilityPV"

    if PARALLEL:
        _tech_results = Parallel(n_jobs=-2)(
            delayed(make_region_data)(technology, region) for region in SCENARIOS[tech]
        )
        _cap_tx, _profiles = zip(*_tech_results)

        cap_tx.extend(_cap_tx)
        profiles.extend(_profiles)
    else:
        for reg in SCENARIOS[tech]:
            region = reg
            logger.info(f"{technology}, {region}")

            # Prepare resource metadata and profiles
            # IP()
            if tech == "OffShoreWind":
                metadata = (
                    pd.read_csv(METADATA_PATHS[technology], dtype={"metro_id": str})
                    .query("prefSite==1 & turbineType=='floating'")
                    .pipe(
                        format_metadata,
                        cap_multiplier=CAPACITY_MULTIPLIER.get(technology),
                        by="lcoe",
                    )
                )
            else:
                metadata = pd.read_csv(METADATA_PATHS[technology], dtype={"metro_id": str}).pipe(
                    format_metadata,
                    cap_multiplier=CAPACITY_MULTIPLIER.get(technology),
                    by="lcoe",
                )
            if tech == "OnshoreWind":
                filters = [
                    [("IPM_Region", "=", r)]
                    for r in SCENARIOS[technology][region]["ipm_regions"]
                ]
                profiles = pd.read_parquet(
                    PROFILE_PATHS[technology], use_pandas_metadata=True, filters=filters
                )
            elif tech == "OffShoreWind":
                profiles = pd.read_parquet(PROFILE_PATHS[technology]).query(
                    "prefSite==1 & turbineType=='floating'"
                )
            else:
                profiles = pd.read_parquet(PROFILE_PATHS[technology])
            format_profiles_inplace(profiles)

            # Build clusters
            scenario = SCENARIOS[technology][region]
            clusters = build_clusters(metadata, **scenario)

            # Build cluster profiles
            cluster_profiles = build_cluster_profiles(clusters, metadata, profiles)

            # Export results as PowerGenome input
            tab1 = clusters_to_capacity_transmission_table(clusters, region, technology)
            spur_line = pd.concat([spur_line, tab1], ignore_index=True)

            tab2 = clusters_to_variable_resource_profiles_table(
                clusters, cluster_profiles, region, technology
            )
            col = tab2[("region", "Resource", "cluster")]
            tab2 = tab2.drop([("region", "Resource", "cluster")], axis=1)
            resource_variability = pd.concat([resource_variability, tab2], axis=1)
            if USE_IP:
                IP()

if PARALLEL:
    resource_variability = pd.concat(profiles, axis=1)
    spur_line = pd.concat(cap_tx)
else:
    resource_variability = pd.concat([col, resource_variability], axis=1)

resource_variability = remove_feb_29(resource_variability, offset=UTC_OFFSET)

spur_line.to_csv(
    "test_offshore_resource_capacity_spur_line.csv", index=False, float_format="%.3f"
)
resource_variability.to_csv(
    "test_offshore_variable_resource_profiles.csv", index=False, float_format="%.4f"
)
