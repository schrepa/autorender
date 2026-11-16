import package::cover::faces::cover::cover_sunward;
import package::engine::shading::{Shaded, SurfacePoint};
import package::face::lit::lit as lit_leaf;

fn lit(point: SurfacePoint, pixel: vec2<f32>) -> Shaded {
    var leaf = point;
    leaf.direct_blocked = 1.0
        - (1.0 - point.direct_blocked) * pow(point.occlusion, cover_sunward());
    return lit_leaf(leaf, pixel);
}
