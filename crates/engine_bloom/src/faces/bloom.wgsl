import package::engine::color::luminance;
import package::face::tonemap::exposure;
import package::generated::pieces::pieces;

fn glow_weight(color: vec3<f32>) -> f32 {
    let bright = max(luminance(color) * exposure() - pieces.bloom.threshold.x, 0.0);
    return bright / (1.0 + bright);
}
