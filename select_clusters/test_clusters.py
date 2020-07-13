from clusters import ClusterBuilder

PATH = ".."
SCENARIOS = [
    {
        "region": "Region",
        "ipm_regions": ["NENGREST", "PJM_Dom"],
        "technology": "offshorewind",
        "turbine_type": "fixed",
        "pref_site": 1,
        "min_capacity": None,
        "max_clusters": 3,
    },
    {
        "region": "Region",
        "ipm_regions": ["NENGREST", "PJM_Dom"],
        "technology": "offshorewind",
        "turbine_type": "fixed",
        "pref_site": 0,
        "min_capacity": 10,
        "max_clusters": None,
    },
]

builder = ClusterBuilder(PATH)
for scenario in SCENARIOS:
    builder.build_clusters(**scenario)
builder.get_cluster_metadata()
builder.get_cluster_profiles()
