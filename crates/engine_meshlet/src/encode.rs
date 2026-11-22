use engine_codec::{Error, ErrorCode, Expected};
use engine_math::{clamp, max};
use glam::{Vec2, Vec3};

use crate::format::{
    CLUSTER_TRI_WORDS, CLUSTER_VERTEX_WORDS, ClusterHeader, ClusterVertex, DecodedCluster,
    MAX_CLUSTER_TRIS, MAX_CLUSTER_VERTS, PAGE_SIZE_BYTES, PAGE_WORDS,
};

fn invalid(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidMesh, path)
}

pub fn encode_octahedral(normal: Vec3) -> [i16; 2] {
    let unit = if normal.is_finite() && normal.length_squared() > 1.0e-20 {
        normal.normalize()
    } else {
        Vec3::Y
    };
    let l1 = unit.x.abs() + unit.y.abs() + unit.z.abs();
    let (u, v) = if l1 > 1.0e-20 {
        (unit.x / l1, unit.y / l1)
    } else {
        (0.0, 1.0)
    };
    let sign = |value: f32| if value >= 0.0 { 1.0 } else { -1.0 };
    let (ox, oy) = if unit.z < 0.0 {
        ((1.0 - v.abs()) * sign(u), (1.0 - u.abs()) * sign(v))
    } else {
        (u, v)
    };
    [
        (clamp(ox, -1.0, 1.0) * 32767.0).round() as i16,
        (clamp(oy, -1.0, 1.0) * 32767.0).round() as i16,
    ]
}

pub fn decode_octahedral(encoded: [i16; 2]) -> Vec3 {
    let px = clamp(f32::from(encoded[0]) / 32767.0, -1.0, 1.0);
    let py = clamp(f32::from(encoded[1]) / 32767.0, -1.0, 1.0);
    let pz = 1.0 - px.abs() - py.abs();
    let fold = max(-pz, 0.0);
    let x = px + if px >= 0.0 { -fold } else { fold };
    let y = py + if py >= 0.0 { -fold } else { fold };
    let raw = Vec3::new(x, y, pz);
    if raw.length_squared() > 1.0e-20 {
        raw.normalize()
    } else {
        Vec3::Y
    }
}

fn quantize_bits(value: f32, min_v: f32, extent: f32, max_code: u32) -> u32 {
    if extent <= 1.0e-12 {
        return 0;
    }
    let t = clamp((value - min_v) / extent, 0.0, 1.0);
    (t * (max_code as f32)).round() as u32
}

fn wrap_uv(value: f32) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    if (0.0..=1.0).contains(&value) {
        value
    } else {
        value.rem_euclid(1.0)
    }
}

pub fn dequantize_position(qx: u16, qy: u16, qz: u16, aabb_min: Vec3, aabb_extent: Vec3) -> Vec3 {
    Vec3::new(
        aabb_min.x + (f32::from(qx) / 16383.0) * aabb_extent.x,
        aabb_min.y + (f32::from(qy) / 16383.0) * aabb_extent.y,
        aabb_min.z + (f32::from(qz) / 16383.0) * aabb_extent.z,
    )
}

pub fn encode_cluster_words(
    vertices: &[ClusterVertex],
    triangles: &[[u8; 3]],
    aabb_min: Vec3,
    aabb_extent: Vec3,
    out: &mut Vec<u32>,
) -> Result<usize, Error> {
    if vertices.is_empty() || vertices.len() > MAX_CLUSTER_VERTS {
        return Err(invalid("$.cluster.vertices")
            .range(1.0, MAX_CLUSTER_VERTS as f64)
            .got(vertices.len()));
    }
    if triangles.is_empty() || triangles.len() > MAX_CLUSTER_TRIS {
        return Err(invalid("$.cluster.triangles")
            .range(1.0, MAX_CLUSTER_TRIS as f64)
            .got(triangles.len()));
    }
    let start_len = out.len();
    out.reserve(vertices.len() * CLUSTER_VERTEX_WORDS + triangles.len() * CLUSTER_TRI_WORDS);
    for v in vertices {
        let gx = quantize_bits(v.position.x, aabb_min.x, aabb_extent.x, 16383);
        let gy = quantize_bits(v.position.y, aabb_min.y, aabb_extent.y, 16383);
        let gz = quantize_bits(v.position.z, aabb_min.z, aabb_extent.z, 16383);
        let r6 = (clamp(v.color[0], 0.0, 1.0).sqrt() * 63.0).round() as u32;
        let g6 = (clamp(v.color[1], 0.0, 1.0).sqrt() * 63.0).round() as u32;
        let b5 = (clamp(v.color[2], 0.0, 1.0).sqrt() * 31.0).round() as u32;
        let ao5 = (clamp(v.ao, 0.0, 1.0) * 31.0).round() as u32;
        let oct = encode_octahedral(v.normal);
        let uv_u = (clamp(wrap_uv(v.uv.x), 0.0, 1.0) * 65535.0).round() as u16;
        let uv_v = (clamp(wrap_uv(v.uv.y), 0.0, 1.0) * 65535.0).round() as u16;
        let w0 = (gx & 0x3FFF) | ((gy & 0x3FFF) << 14) | ((gz & 0xF) << 28);
        let w1 = ((gz >> 4) & 0x3FF)
            | ((r6 & 0x3F) << 10)
            | ((g6 & 0x3F) << 16)
            | ((b5 & 0x1F) << 22)
            | ((ao5 & 0x1F) << 27);
        let w2 = u32::from(oct[0] as u16) | (u32::from(oct[1] as u16) << 16);
        let w3 = u32::from(uv_u) | (u32::from(uv_v) << 16);
        out.extend_from_slice(&[w0, w1, w2, w3]);
    }
    let vert_limit = vertices.len() as u8;
    for (ti, &[i0, i1, i2]) in triangles.iter().enumerate() {
        if i0 >= vert_limit || i1 >= vert_limit || i2 >= vert_limit {
            return Err(invalid(format!("$.cluster.triangles[{ti}]"))
                .at_most(u64::from(vert_limit.saturating_sub(1)))
                .got([i0, i1, i2]));
        }
        let packed = u32::from(i0) | (u32::from(i1) << 8) | (u32::from(i2) << 16);
        out.push(packed);
    }
    Ok(out.len() - start_len)
}

pub fn decode_cluster_words(
    header: &ClusterHeader,
    page_words: &[u32],
) -> Result<DecodedCluster, Error> {
    let vert_count = header.vert_count() as usize;
    let tri_count = header.tri_count() as usize;
    let offset = header.page_word_offset() as usize;
    if vert_count == 0 || vert_count > MAX_CLUSTER_VERTS {
        return Err(invalid("$.header.vert_count")
            .range(1.0, MAX_CLUSTER_VERTS as f64)
            .got(vert_count));
    }
    if tri_count == 0 || tri_count > MAX_CLUSTER_TRIS {
        return Err(invalid("$.header.tri_count")
            .range(1.0, MAX_CLUSTER_TRIS as f64)
            .got(tri_count));
    }
    let needed = vert_count * CLUSTER_VERTEX_WORDS + tri_count * CLUSTER_TRI_WORDS;
    let end = offset.checked_add(needed).ok_or_else(|| {
        invalid("$.header.page_word_offset")
            .at_most(PAGE_WORDS as u64)
            .got(offset)
    })?;
    if end > page_words.len() || end > PAGE_WORDS {
        return Err(invalid("$.header.page_word_offset")
            .at_most(page_words.len() as u64)
            .got(end));
    }
    let aabb_min = header.aabb_min_vec3();
    let aabb_extent = header.aabb_extent_vec3();
    let mut vertices = Vec::with_capacity(vert_count);
    let mut cursor = offset;
    for _ in 0..vert_count {
        let w0 = page_words[cursor];
        let w1 = page_words[cursor + 1];
        let w2 = page_words[cursor + 2];
        let w3 = page_words[cursor + 3];
        cursor += CLUSTER_VERTEX_WORDS;
        let qx = (w0 & 0x3FFF) as u16;
        let qy = ((w0 >> 14) & 0x3FFF) as u16;
        let qz = (((w0 >> 28) & 0xF) | ((w1 & 0x3FF) << 4)) as u16;
        let r = f32::from(((w1 >> 10) & 0x3F) as u8) / 63.0;
        let g = f32::from(((w1 >> 16) & 0x3F) as u8) / 63.0;
        let b = f32::from(((w1 >> 22) & 0x1F) as u8) / 31.0;
        let cr = r * r;
        let cg = g * g;
        let cb = b * b;
        let ao = f32::from(((w1 >> 27) & 0x1F) as u8) / 31.0;
        let oct_x = (w2 & 0xFFFF) as u16 as i16;
        let oct_y = ((w2 >> 16) & 0xFFFF) as u16 as i16;
        let uv_u = f32::from((w3 & 0xFFFF) as u16) / 65535.0;
        let uv_v = f32::from(((w3 >> 16) & 0xFFFF) as u16) / 65535.0;
        vertices.push(ClusterVertex {
            position: dequantize_position(qx, qy, qz, aabb_min, aabb_extent),
            normal: decode_octahedral([oct_x, oct_y]),
            uv: Vec2::new(uv_u, uv_v),
            color: [cr, cg, cb],
            ao,
        });
    }
    let mut triangles = Vec::with_capacity(tri_count);
    let vert_limit = vert_count as u8;
    for ti in 0..tri_count {
        let word = page_words[cursor + ti];
        let i0 = (word & 0xFF) as u8;
        let i1 = ((word >> 8) & 0xFF) as u8;
        let i2 = ((word >> 16) & 0xFF) as u8;
        if i0 >= vert_limit || i1 >= vert_limit || i2 >= vert_limit {
            return Err(invalid(format!("$.cluster.triangles[{ti}]"))
                .at_most(u64::from(vert_limit.saturating_sub(1)))
                .got([i0, i1, i2]));
        }
        triangles.push([i0, i1, i2]);
    }
    Ok(DecodedCluster {
        vertices,
        triangles,
    })
}

pub fn words_to_le_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for &word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

pub fn page_bytes_as_words(bytes: &[u8]) -> Result<Vec<u32>, Error> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(PAGE_SIZE_BYTES) {
        return Err(invalid("$.pages")
            .expected(Expected::Multiple)
            .got(bytes.len()));
    }
    let mut words = Vec::with_capacity(bytes.len() / 4);
    for &chunk in bytes.as_chunks::<4>().0 {
        words.push(u32::from_le_bytes(chunk));
    }
    Ok(words)
}
