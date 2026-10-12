/// The op of every query, as requests name it.
pub const QUERY_OPS: [&str; 15] = ["status", "inspect", "assets", "digest", "describe", "raycast", "overlap", "shape_cast", "terrain_height", "nav_path", "water_query", "deformation_query", "terrain_sweep", "terrain_route", "collision_events"];
/// The op of every control, as requests name it.
pub const CONTROL_OPS: [&str; 22] = ["present", "culling", "environment", "terrain", "terrain_edit", "terrain_undo", "terrain_redo", "deform", "nav_go", "animation", "world_changes", "camera", "benchmark", "step", "input", "render", "pause", "resume", "capture", "game", "reload", "debug_panic"];
