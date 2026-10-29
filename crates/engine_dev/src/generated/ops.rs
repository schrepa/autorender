/// The op of every query, as requests name it.
pub const QUERY_OPS: [&str; 14] = ["status", "inspect", "assets", "digest", "distribution", "describe", "terrain_height", "terrain_route", "nav_path", "raycast", "overlap", "shape_cast", "collision_events", "terrain_sweep"];
/// The op of every control, as requests name it.
pub const CONTROL_OPS: [&str; 26] = ["present", "culling", "environment", "camera", "benchmark", "step", "frames", "input", "render", "pause", "resume", "capture", "game", "reload", "debug_panic", "debug_fault", "debug_device_loss", "debug_missing_asset", "probe", "terrain", "terrain_edit", "terrain_undo", "terrain_redo", "terrain_river", "world_changes", "nav_go"];
