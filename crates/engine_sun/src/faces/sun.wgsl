import package::face::sky::sunlight;
import package::generated::pieces::pieces;

fn sun_direction() -> vec3<f32> {
    return pieces.sun.direction.xyz;
}

fn sun_color() -> vec3<f32> {
    return sunlight(pieces.sun.color.rgb);
}
