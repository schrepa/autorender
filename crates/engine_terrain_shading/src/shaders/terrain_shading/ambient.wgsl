import package::place::ambient::{EnvironmentAt, EnvironmentLight};
import package::terrain_shading::faces::bounce::bounce_light;

fn environment(light: EnvironmentLight, at: EnvironmentAt) -> EnvironmentLight {
    if at.ground {
        return light;
    }
    let around = bounce_light(at.world, at.normal, false);
    var result = light;
    result.near = around.weight;
    result.near_open = around.sky;
    result.bounce = around.light * around.weight;
    return result;
}
