/// The op of every query, as requests name it.
pub const QUERY_OPS: [&str; 14] = ["status", "inspect", "assets", "digest", "distribution", "describe", "raycast", "overlap", "shape_cast", "terrain_height", "nav_path", "terrain_sweep", "terrain_route", "collision_events"];
/// The op of every control, as requests name it.
pub const CONTROL_OPS: [&str; 26] = ["present", "culling", "environment", "terrain", "terrain_edit", "terrain_undo", "terrain_redo", "terrain_river", "nav_go", "world_changes", "camera", "benchmark", "step", "frames", "input", "render", "pause", "resume", "capture", "game", "reload", "debug_panic", "debug_fault", "debug_device_loss", "debug_missing_asset", "probe"];
