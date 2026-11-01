import package::generated::pieces::reflections::piece;

fn reflects() -> bool {
    return piece().surface.y != 0.0;
}

fn max_roughness() -> f32 {
    return piece().surface.x;
}
