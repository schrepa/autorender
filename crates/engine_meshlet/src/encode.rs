use std::io::{Read as _, Write as _};

use engine_codec::{Error, ErrorCode, Expected};
use engine_math::{clamp, max};
use flate2::Compression;
use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use glam::{Vec2, Vec3};

use crate::format::{
    CLUSTER_HEADER_WORDS, CLUSTER_TRI_WORDS, CLUSTER_VERTEX_WORDS, ClusterHeader, ClusterVertex,
    DecodedCluster, MAX_CLUSTER_TRIS, MAX_CLUSTER_VERTS, PAGE_SIZE_BYTES, PAGE_WORDS, PageSpan,
};

const UV_FIXED_SCALE: f32 = 2048.0;
const UV_FIXED_MAX: f32 = 65535.0 / UV_FIXED_SCALE;

fn invalid(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidMesh, path)
}

pub fn compute_base_quant_step(bounds_min: Vec3, bounds_max: Vec3) -> f32 {
    let extent = (bounds_max - bounds_min).max(Vec3::ZERO);
    let max_ext = extent.max_element();
    let max_abs = max(bounds_min.abs().max_element(), bounds_max.abs().max_element());
    let target = max(max(max_ext / 262_144.0, max_abs / 4_194_304.0), 1.0 / 65_536.0);
    let exp = target.log2().ceil() as i32;
    2.0_f32.powi(exp.clamp(-16, 16))
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

fn clean_uv_coord(value: f32) -> f32 {
    if value.is_finite() && value.abs() <= 64.0 {
        value
    } else {
        0.0
    }
}

fn cluster_uv_origin(vertices: &[ClusterVertex]) -> (f32, f32) {
    let mut min_u = f32::INFINITY;
    let mut min_v = f32::INFINITY;
    for v in vertices {
        let u = clean_uv_coord(v.uv.x);
        let w = clean_uv_coord(v.uv.y);
        if u < min_u {
            min_u = u;
        }
        if w < min_v {
            min_v = w;
        }
    }
    if min_u.is_finite() && min_v.is_finite() {
        (min_u.floor(), min_v.floor())
    } else {
        (0.0, 0.0)
    }
}

pub fn dequantize_position(qx: u16, qy: u16, qz: u16, origin: Vec3, step: f32) -> Vec3 {
    origin + Vec3::new(f32::from(qx), f32::from(qy), f32::from(qz)) * step
}

fn choose_cluster_grid(vertices: &[ClusterVertex], base_step: f32) -> (Vec3, f32, Vec<[u16; 3]>) {
    let mut step = max(base_step, 1.0 / 65_536.0);
    let mut q_coords = Vec::with_capacity(vertices.len());
    loop {
        let inv = 1.0 / step;
        let mut min_i = [i64::MAX; 3];
        let mut max_i = [i64::MIN; 3];
        let mut raw = Vec::with_capacity(vertices.len());
        for v in vertices {
            let gx = (f64::from(v.position.x) * f64::from(inv)).round() as i64;
            let gy = (f64::from(v.position.y) * f64::from(inv)).round() as i64;
            let gz = (f64::from(v.position.z) * f64::from(inv)).round() as i64;
            min_i[0] = min_i[0].min(gx);
            min_i[1] = min_i[1].min(gy);
            min_i[2] = min_i[2].min(gz);
            max_i[0] = max_i[0].max(gx);
            max_i[1] = max_i[1].max(gy);
            max_i[2] = max_i[2].max(gz);
            raw.push([gx, gy, gz]);
        }
        let span_x = max_i[0].saturating_sub(min_i[0]);
        let span_y = max_i[1].saturating_sub(min_i[1]);
        let span_z = max_i[2].saturating_sub(min_i[2]);
        if span_x <= 65_535 && span_y <= 65_535 && span_z <= 65_535 {
            let origin = Vec3::new(
                (min_i[0] as f64 * f64::from(step)) as f32,
                (min_i[1] as f64 * f64::from(step)) as f32,
                (min_i[2] as f64 * f64::from(step)) as f32,
            );
            q_coords.clear();
            for [gx, gy, gz] in raw {
                q_coords.push([
                    (gx - min_i[0]).clamp(0, 65_535) as u16,
                    (gy - min_i[1]).clamp(0, 65_535) as u16,
                    (gz - min_i[2]).clamp(0, 65_535) as u16,
                ]);
            }
            return (origin, step, q_coords);
        }
        step *= 2.0;
    }
}

pub fn encode_cluster_words(
    vertices: &[ClusterVertex],
    triangles: &[[u8; 3]],
    base_step: f32,
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
    let (origin, step, q_coords) = choose_cluster_grid(vertices, base_step);
    let (base_u, base_v) = cluster_uv_origin(vertices);
    let start_len = out.len();
    out.reserve(
        CLUSTER_HEADER_WORDS
            + vertices.len() * CLUSTER_VERTEX_WORDS
            + triangles.len() * CLUSTER_TRI_WORDS,
    );
    out.extend_from_slice(&[
        origin.x.to_bits(),
        origin.y.to_bits(),
        origin.z.to_bits(),
        step.to_bits(),
    ]);
    for (v, [qx, qy, qz]) in vertices.iter().zip(q_coords) {
        let r4 = (clamp(v.color[0], 0.0, 1.0).sqrt() * 15.0).round() as u32;
        let g4 = (clamp(v.color[1], 0.0, 1.0).sqrt() * 15.0).round() as u32;
        let b3 = (clamp(v.color[2], 0.0, 1.0).sqrt() * 7.0).round() as u32;
        let ao5 = (clamp(v.ao, 0.0, 1.0) * 31.0).round() as u32;
        let oct = encode_octahedral(v.normal);
        let su = clamp(clean_uv_coord(v.uv.x) - base_u, 0.0, UV_FIXED_MAX);
        let sv = clamp(clean_uv_coord(v.uv.y) - base_v, 0.0, UV_FIXED_MAX);
        let uv_u = (su * UV_FIXED_SCALE).round() as u16;
        let uv_v = (sv * UV_FIXED_SCALE).round() as u16;
        let w0 = u32::from(qx) | (u32::from(qy) << 16);
        let w1 = u32::from(qz)
            | ((r4 & 0xF) << 16)
            | ((g4 & 0xF) << 20)
            | ((b3 & 0x7) << 24)
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
    let needed = CLUSTER_HEADER_WORDS
        + vert_count * CLUSTER_VERTEX_WORDS
        + tri_count * CLUSTER_TRI_WORDS;
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
    let origin = Vec3::new(
        f32::from_bits(page_words[offset]),
        f32::from_bits(page_words[offset + 1]),
        f32::from_bits(page_words[offset + 2]),
    );
    let step = f32::from_bits(page_words[offset + 3]);
    if !origin.is_finite() || !step.is_finite() || step <= 0.0 {
        return Err(invalid("$.cluster.grid").expected(Expected::Finite));
    }
    let mut vertices = Vec::with_capacity(vert_count);
    let mut cursor = offset + CLUSTER_HEADER_WORDS;
    for _ in 0..vert_count {
        let w0 = page_words[cursor];
        let w1 = page_words[cursor + 1];
        let w2 = page_words[cursor + 2];
        let w3 = page_words[cursor + 3];
        cursor += CLUSTER_VERTEX_WORDS;
        let qx = (w0 & 0xFFFF) as u16;
        let qy = (w0 >> 16) as u16;
        let qz = (w1 & 0xFFFF) as u16;
        let r = f32::from(((w1 >> 16) & 0xF) as u8) / 15.0;
        let g = f32::from(((w1 >> 20) & 0xF) as u8) / 15.0;
        let b = f32::from(((w1 >> 24) & 0x7) as u8) / 7.0;
        let ao = f32::from(((w1 >> 27) & 0x1F) as u8) / 31.0;
        let oct_x = (w2 & 0xFFFF) as u16 as i16;
        let oct_y = ((w2 >> 16) & 0xFFFF) as u16 as i16;
        let uv_u = f32::from((w3 & 0xFFFF) as u16) / UV_FIXED_SCALE;
        let uv_v = f32::from(((w3 >> 16) & 0xFFFF) as u16) / UV_FIXED_SCALE;
        vertices.push(ClusterVertex {
            position: dequantize_position(qx, qy, qz, origin, step),
            normal: decode_octahedral([oct_x, oct_y]),
            uv: Vec2::new(uv_u, uv_v),
            color: [r * r, g * g, b * b],
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
        origin,
        step,
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

pub fn compress_page_words(words: &[u32]) -> Result<Vec<u8>, Error> {
    if words.len() != PAGE_WORDS {
        return Err(invalid("$.page.words")
            .expected(Expected::Matching)
            .got(words.len()));
    }
    let raw = words_to_le_bytes(words);
    let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
    enc.write_all(&raw)
        .map_err(|_| invalid("$.page.compress").expected(Expected::Supported))?;
    enc.finish()
        .map_err(|_| invalid("$.page.compress").expected(Expected::Supported))
}

pub fn decompress_page_words(bytes: &[u8]) -> Result<Vec<u32>, Error> {
    let mut dec = DeflateDecoder::new(bytes);
    let mut raw = Vec::with_capacity(PAGE_SIZE_BYTES);
    dec.read_to_end(&mut raw)
        .map_err(|_| invalid("$.page.decompress").expected(Expected::Supported))?;
    if raw.len() != PAGE_SIZE_BYTES {
        return Err(invalid("$.page.size")
            .expected(Expected::Matching)
            .got(raw.len()));
    }
    page_bytes_as_words(&raw)
}

pub fn encode_compressed_pages(
    page_words: &[u32],
    page_count: u32,
) -> Result<(Vec<u8>, Vec<PageSpan>), Error> {
    let count = page_count as usize;
    if page_words.len() != count * PAGE_WORDS {
        return Err(invalid("$.pages.len")
            .expected(Expected::Matching)
            .got(page_words.len()));
    }
    let mut blob = Vec::with_capacity(count * (PAGE_SIZE_BYTES / 2));
    let mut spans = Vec::with_capacity(count);
    for pid in 0..count {
        let slice = &page_words[pid * PAGE_WORDS..(pid + 1) * PAGE_WORDS];
        let comp = compress_page_words(slice)?;
        let byte_offset = blob.len() as u64;
        let byte_len = comp.len() as u32;
        blob.extend_from_slice(&comp);
        spans.push(PageSpan {
            byte_offset,
            byte_len,
        });
    }
    Ok((blob, spans))
}

pub fn decode_compressed_pages(bytes: &[u8], spans: &[PageSpan]) -> Result<Vec<u32>, Error> {
    let mut out = Vec::with_capacity(spans.len() * PAGE_WORDS);
    for (idx, span) in spans.iter().enumerate() {
        let start = span.byte_offset as usize;
        let end = start
            .checked_add(span.byte_len as usize)
            .ok_or_else(|| invalid(format!("$.page_spans[{idx}]")).expected(Expected::Present))?;
        let slice = bytes
            .get(start..end)
            .ok_or_else(|| invalid(format!("$.page_spans[{idx}]")).expected(Expected::Present))?;
        let words = decompress_page_words(slice)?;
        out.extend_from_slice(&words);
    }
    Ok(out)
}
