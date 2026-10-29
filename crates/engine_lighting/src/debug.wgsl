import package::meshes::debug::DEBUG_SHADOWED;

fn debug_shadowed(source: u32, visible: f32) -> u32 {
    return source | select(0u, DEBUG_SHADOWED, visible < 0.5);
}
