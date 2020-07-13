from clusters import ClusterBuilder, load_metadata, read_parquet

PATH = "."
SCENARIOS = [
    {
        "region": "CA_N",
        "ipm_regions": ["WEC_CALN", "WECC_BANC"],
        "technology": "wind",
        # "turbine_type": "fixed",
        # "pref_site": 1,
        "min_capacity": 22000,
        "max_clusters": 1,
        "cap_multiplier": 1,
    },
    {
        "region": "CA_S",
        "ipm_regions": ["WEC_SCE", "WEC_LADW", "WECC_SCE", "WEC_SDGE", "WECC_IID"],
        "technology": "wind",
        # "turbine_type": "fixed",
        # "pref_site": 0,
        "min_capacity": 45000,
        "max_clusters": 10,
    },
    {
        "region": "CA_N",
        "ipm_regions": ["WEC_CALN", "WECC_BANC"],
        "technology": "solarpv",
        # "turbine_type": "fixed",
        # "pref_site": 1,
        "min_capacity": 200000,
        "max_clusters": 4,
        "cap_multiplier": 0.2,
    },
]

builder = ClusterBuilder(PATH)
builder.build_clusters(**SCENARIOS[0])
builder.build_clusters(**SCENARIOS[1])
metadata = builder.get_cluster_metadata()
profiles = builder.get_cluster_profiles()

metadata.to_csv("test_metadata.csv", index=False)
profiles.to_csv("test_profiles.csv", index=False)
