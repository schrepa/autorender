/// The op of every query, as requests name it.
pub const QUERY_OPS: [&str; 16] = ["status", "inspect", "assets", "digest", "distribution", "describe", "raycast", "overlap", "shape_cast", "terrain_height", "nav_path", "water_query", "deformation_query", "terrain_sweep", "terrain_route", "collision_events"];
/// The op of every control, as requests name it.
pub const CONTROL_OPS: [&str; 24] = ["present", "culling", "environment", "terrain", "terrain_edit", "terrain_undo", "terrain_redo", "deform", "nav_go", "animation", "world_changes", "camera", "benchmark", "step", "frames", "input", "render", "pause", "resume", "capture", "game", "reload", "debug_panic", "debug_fault"];
