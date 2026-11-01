import package::cover::faces::cover::cover_tint;
import package::place::far_cover::GroundAt;

fn far_color(color: vec3<f32>, at: GroundAt) -> vec3<f32> {
    return color * cover_tint(at.lush, at.vegetated);
}
