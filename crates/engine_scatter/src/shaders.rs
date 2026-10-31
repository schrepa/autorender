engine_render::shader_modules! {
    "scatter::piece" in "scatter" = "piece.wgsl";
    "scatter::faces::occupancy" in "scatter" = "faces/occupancy.wgsl";
    "scatter::faces::placed" in "scatter" = "faces/placed.wgsl";
    "terrain::occupancy_maps" in "scatter" = "occupancy.wgsl";
    "scatter::cull" in "scatter" = "scatter_cull.wgsl";
    "scatter::snap" in "scatter" = "scatter_snap.wgsl";
}

engine_render::entry_points! {
    scatter_cull {
        compute: [cull_scatter @ "scatter::cull"],
    }
    scatter_snap {
        compute: [snap_scatter @ "scatter::snap"],
    }
    occupancy {
        vertex: [
            vs_splat @ "terrain::occupancy_maps",
            vs_clear @ "terrain::occupancy_maps",
            vs_means @ "terrain::occupancy_maps",
        ],
        fragment: [
            fs_splat @ "terrain::occupancy_maps",
            fs_clear @ "terrain::occupancy_maps",
            fs_means @ "terrain::occupancy_maps",
        ],
        compute: [select_covers @ "terrain::occupancy_maps"],
    }
}
