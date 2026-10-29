import package::generated::pieces::pieces;

fn reflects() -> bool {
    return pieces.reflections.surface.y != 0.0;
}

fn max_roughness() -> f32 {
    return pieces.reflections.surface.x;
}
