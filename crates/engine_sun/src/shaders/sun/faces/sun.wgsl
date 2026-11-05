import package::face::sky::sunlight;
import package::generated::pieces::sun::piece;

fn sun_direction() -> vec3<f32> {
    return piece().direction.xyz;
}

fn sun_color() -> vec3<f32> {
    return sunlight();
}
