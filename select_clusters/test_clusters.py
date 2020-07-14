from clusters import ClusterBuilder

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
    {
        "region": "CA_S",
        "ipm_regions": ["WEC_SCE", "WEC_LADW", "WECC_SCE", "WEC_SDGE", "WECC_IID"],
        "technology": "solarpv",
        # "turbine_type": "fixed",
        # "pref_site": 1,
        "min_capacity": 150000,
        "max_clusters": 5,
        "cap_multiplier": 0.2,
    },
    {
        "region": "CA_N",
        "ipm_regions": ["WEC_CALN", "WECC_BANC"],
        "technology": "offshorewind",
        "turbine_type": "floating",
        "pref_site": 1,
        "min_capacity": 35000,
        "max_clusters": 3,
        # "cap_multiplier": 0.2,
    },
]

builder = ClusterBuilder(PATH)
for scenario in SCENARIOS:
    builder.build_clusters(**scenario)
metadata = builder.get_cluster_metadata()
profiles = builder.get_cluster_profiles().round(4)

metadata.to_csv("test_metadata.csv", index=False)
profiles.to_csv("test_profiles.csv", index=False)
