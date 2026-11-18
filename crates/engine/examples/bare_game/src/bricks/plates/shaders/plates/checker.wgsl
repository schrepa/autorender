const PLATE_SQUARES: f32 = 8.0;
const PLATE_LIGHT: vec3<f32> = vec3<f32>(0.9, 0.8, 0.3);
const PLATE_DARK: vec3<f32> = vec3<f32>(0.2, 0.3, 0.6);

fn checker(uv: vec2<f32>) -> vec3<f32> {
    let square = floor(uv * PLATE_SQUARES);
    let odd = (i32(square.x) + i32(square.y)) & 1;
    return select(PLATE_DARK, PLATE_LIGHT, odd == 1);
}
