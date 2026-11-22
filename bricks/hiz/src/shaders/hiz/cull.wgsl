fn hiz_project_sphere(
    center_vs: vec3<f32>,
    radius: f32,
    p00: f32,
    p11: f32,
    near_z: f32,
) -> vec4<f32> {
    let r = max(radius, 0.0);
    let z = -center_vs.z;
    if z <= r + max(near_z, 1.0e-6) {
        return vec4<f32>(0.0, 0.0, 1.0, 1.0);
    }
    let cx = vec2<f32>(center_vs.x, z);
    let tx = sqrt(max(dot(cx, cx) - r * r, 0.0));
    let minx = vec2<f32>(tx * cx.x - r * cx.y, r * cx.x + tx * cx.y);
    let maxx = vec2<f32>(tx * cx.x + r * cx.y, -r * cx.x + tx * cx.y);

    let cy = vec2<f32>(center_vs.y, z);
    let ty = sqrt(max(dot(cy, cy) - r * r, 0.0));
    let miny = vec2<f32>(ty * cy.x - r * cy.y, r * cy.x + ty * cy.y);
    let maxy = vec2<f32>(ty * cy.x + r * cy.y, -r * cy.x + ty * cy.y);

    let sx = abs(p00);
    let sy = abs(p11);
    let ndc_min_x = (minx.x / max(minx.y, 1.0e-7)) * sx;
    let ndc_max_x = (maxx.x / max(maxx.y, 1.0e-7)) * sx;
    let ndc_min_y = (miny.x / max(miny.y, 1.0e-7)) * sy;
    let ndc_max_y = (maxy.x / max(maxy.y, 1.0e-7)) * sy;

    let u_min = ndc_min_x * 0.5 + 0.5;
    let u_max = ndc_max_x * 0.5 + 0.5;
    let v_min = 0.5 - ndc_max_y * 0.5;
    let v_max = 0.5 - ndc_min_y * 0.5;
    return clamp(
        vec4<f32>(u_min, v_min, u_max, v_max),
        vec4<f32>(0.0),
        vec4<f32>(1.0),
    );
}

fn hiz_test_rect(
    hiz_tex: texture_2d<f32>,
    uv_min: vec2<f32>,
    uv_max: vec2<f32>,
    nearest_ndc_z: f32,
    mip0_size: vec2<f32>,
    mip_count: u32,
) -> bool {
    if mip_count == 0u || nearest_ndc_z >= 1.0 {
        return false;
    }
    let uv0 = clamp(min(uv_min, uv_max), vec2<f32>(0.0), vec2<f32>(1.0));
    let uv1 = clamp(max(uv_min, uv_max), vec2<f32>(0.0), vec2<f32>(1.0));
    let mip0_i = max(vec2<i32>(mip0_size), vec2<i32>(1, 1));
    let max_mip0 = mip0_i - vec2<i32>(1, 1);
    let px_min = clamp(vec2<i32>(floor(uv0 * mip0_size)), vec2<i32>(0, 0), max_mip0);
    let px_max = clamp(vec2<i32>(floor(uv1 * mip0_size)), vec2<i32>(0, 0), max_mip0);

    let span = u32(max(px_max.x - px_min.x, px_max.y - px_min.y));
    var mip: u32 = 0u;
    if span > 1u {
        mip = 31u - countLeadingZeros(span);
    }
    let max_mip = mip_count - 1u;
    mip = min(mip, max_mip);
    var min_t = vec2<u32>(px_min) >> vec2<u32>(mip);
    var max_t = vec2<u32>(px_max) >> vec2<u32>(mip);
    if (max_t.x - min_t.x > 1u || max_t.y - min_t.y > 1u) && mip < max_mip {
        mip = mip + 1u;
        min_t = min_t >> vec2<u32>(1u);
        max_t = max_t >> vec2<u32>(1u);
    }
    let mip_w = max(u32(mip0_i.x) >> mip, 1u);
    let mip_h = max(u32(mip0_i.y) >> mip, 1u);
    let mip_max = vec2<u32>(mip_w - 1u, mip_h - 1u);
    min_t = min(min_t, mip_max);
    max_t = min(max_t, mip_max);

    let lod = i32(mip);
    let d00 = textureLoad(hiz_tex, vec2<i32>(i32(min_t.x), i32(min_t.y)), lod).x;
    let d10 = textureLoad(hiz_tex, vec2<i32>(i32(max_t.x), i32(min_t.y)), lod).x;
    let d01 = textureLoad(hiz_tex, vec2<i32>(i32(min_t.x), i32(max_t.y)), lod).x;
    let d11 = textureLoad(hiz_tex, vec2<i32>(i32(max_t.x), i32(max_t.y)), lod).x;
    let min_hiz_z = min(min(d00, d10), min(d01, d11));
    return nearest_ndc_z < min_hiz_z;
}
