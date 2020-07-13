from clusters import ClusterBuilder, load_metadata, read_parquet

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
builder.build_clusters(**SCENARIOS[0])
builder.build_clusters(**SCENARIOS[1])
builder.get_cluster_metadata()
builder.get_cluster_profiles()
