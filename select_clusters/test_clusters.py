import pandas as pd
from clusters import (
    format_metadata,
    format_profiles_inplace,
    build_clusters,
    build_cluster_profiles,
    clusters_to_capacity_transmission_table,
    clusters_to_variable_resource_profiles_table,
)
from pathlib import Path

# ---- Constants ----

CWD = Path.cwd()

METADATA_PATHS = {
    "OnshoreWind": CWD.parent / "wind_base_cluster_metadata.csv",
    # "OnshoreWind": CWD.parent / "preliminary_base_wind_metadata.csv",
    "UtilityPV": CWD.parent / "solarpv_base_cluster_metadata.csv",
}

PROFILE_PATHS = {
    "OnshoreWind": CWD.parent / "wind_base_cluster_profiles.parquet",
    # "OnshoreWind": CWD.parent / "preliminary_base_wind_cluster_profiles.parquet",
    "UtilityPV": CWD.parent / "solarpv_base_cluster_profiles.parquet",
}

CAPACITY_MULTIPLIER = {
    "OnshoreWind": 1,
    "UtilityPV": 0.2,
}

SCENARIOS = {
    "UtilityPV": {
        "CA_N": {
            "ipm_regions": ["WEC_CALN", "WECC_BANC"],
            "max_clusters": 4,
            "min_capacity": 200,
        },
        "CA_S": {
            "ipm_regions": ["WEC_SCE", "WEC_LADW", "WECC_SCE", "WEC_SDGE", "WECC_IID"],
            "max_clusters": 10,  # 5
            "min_capacity": 150,
        },
        "WECC_N": {
            "ipm_regions": ["WECC_ID", "WECC_MT", "WECC_NNV", "WECC_SNV", "WECC_UT"],
            "max_clusters": 20,
            "min_capacity": 250,
        },
        "WECC_NMAZ": {
            "ipm_regions": ["WECC_AZ", "WECC_NM"],
            "max_clusters": 15,
            "min_capacity": 300,
        },
        # "WECC_PNW": {
        #     "ipm_regions": [],
        #     "max_clusters": 1,
        #     "min_capacity": 4.5,
        # },
        "WECC_WYCO": {
            "ipm_regions": ["WECC_WY", "WECC_CO"],
            "max_clusters": 15,
            "min_capacity": 180,
        },
    },
    "OnshoreWind": {
        "CA_N": {
            "ipm_regions": ["WEC_CALN", "WECC_BANC"],
            "max_clusters": 1,
            "min_capacity": 22,
        },
        "CA_S": {
            "ipm_regions": ["WEC_SCE", "WEC_LADW", "WECC_SCE", "WEC_SDGE", "WECC_IID"],
            "max_clusters": 3,  # 10
            "min_capacity": 45,
        },
        "WECC_N": {
            "ipm_regions": ["WECC_ID", "WECC_MT", "WECC_NNV", "WECC_SNV", "WECC_UT"],
            "max_clusters": 15,
            "min_capacity": 100,
        },
        "WECC_NMAZ": {
            "ipm_regions": ["WECC_AZ", "WECC_NM"],
            "max_clusters": 15,
            "min_capacity": 120,
        },
        # "WECC_PNW": {
        #     "ipm_regions": [],
        #     "max_clusters": 1,
        #     "min_capacity": 6,
        # },
        "WECC_WYCO": {
            "ipm_regions": ["WECC_WY", "WECC_CO"],
            "max_clusters": 20,
            "min_capacity": 200,
        },
    },
}

# ---- Processing ----

technology = "UtilityPV"
region = "CA_N"
scenario = SCENARIOS[technology][region]

# Prepare resource metadata and profiles
metadata = pd.read_csv(METADATA_PATHS[technology]).pipe(
    format_metadata, cap_multiplier=CAPACITY_MULTIPLIER[technology], by="lcoe"
)
profiles = pd.read_parquet(
    PROFILE_PATHS[technology],
    engine="pyarrow",
    filters=[[("IPM_Region", "=", region)] for region in scenario["ipm_regions"]],
    # NOTE: Needed for unpartitioned datasets
    # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html
    use_legacy_dataset=False
)
format_profiles_inplace(profiles)

# Build clusters
clusters = build_clusters(metadata, **scenario)

# Build cluster profiles
cluster_profiles = build_cluster_profiles(clusters, metadata, profiles)

# Export results as PowerGenome input
tab1 = clusters_to_capacity_transmission_table(clusters, region, technology)
tab1.to_csv(
    f"{technology}_{region}_test_resource_capacity_spur_line.csv",
    index=False,
    float_format="%.3f",
)
tab2 = clusters_to_variable_resource_profiles_table(
    clusters, cluster_profiles, region, technology
)
tab2.to_csv(
    f"{technology}_{region}_test_variable_resource_profiles.csv",
    index=False,
    float_format="%.4f",
)
