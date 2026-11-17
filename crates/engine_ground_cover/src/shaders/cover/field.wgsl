import package::cover::faces::cover::cover_sunward;
import package::engine::shading::SurfacePoint;

fn lighted(point: SurfacePoint) -> SurfacePoint {
    var leaf = point;
    leaf.direct_blocked = 1.0
        - (1.0 - point.direct_blocked) * pow(point.occlusion, cover_sunward());
    return leaf;
}
