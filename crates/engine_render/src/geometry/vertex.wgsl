fn output_color(material: vec4<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16unorm(material.x), unpack2x16unorm(material.y));
}
