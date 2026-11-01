use serde_json::{Value, json};

pub(crate) fn terrain(rendering: &Value) -> Value {
    let held = &rendering["terrain_stats"];
    let shaded = &rendering["terrain_shading_stats"];
    let placed = &rendering["scatter_stats"];
    json!({
        "triangles": shaded["triangles"],
        "shallow_subdivisions": shaded["shallow_subdivisions"],
        "shadow_patches": shaded["shadow_patches"],
        "tiles": shaded["tiles"],
        "loading_tiles": held["loading_tiles"],
        "gpu_controls": shaded["gpu_controls"],
        "dropped_tiles": shaded["tiles_dropped"]["cut"],
        "fallback_tiles": held["fallback_tiles"],
        "horizons": shaded["horizons"],
        "sky_maps": shaded["sky_maps"],
        "lighting_uploads": shaded["lighting_uploads"],
        "bounce": {
            "relit": shaded["bounce_relit"],
            "probes": shaded["bounce_probes"],
            "levels": shaded["bounce_levels"],
            "dropped": shaded["bounce_dropped"],
        },
        "macro_maps": shaded["macro_maps"],
        "dropped_materials": shaded["dropped_materials"],
        "placed_objects": placed["placed_objects"],
        "placed_far_objects": placed["placed_far_objects"],
        "placed_impostors": placed["placed_impostors"],
        "dropped_placed_objects": placed["dropped_objects"],
        "white_water": rendering["water_stats"]["white_water"],
        "tile_cut": held["tile_cut"],
        "leaf_cut": shaded["leaves"],
    })
}
