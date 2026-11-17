import package::cover::faces::cover::cover_tint;
import package::place::far_cover::{FarGround, GroundAt};

fn far_cover(ground: FarGround, at: GroundAt) -> FarGround {
    var result = ground;
    result.color = ground.color * cover_tint(at.lush, at.vegetated);
    return result;
}
