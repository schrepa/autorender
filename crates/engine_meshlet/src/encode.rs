use std::io::{Read as _, Write as _};

use engine_codec::{Error, ErrorCode, Expected};
use engine_math::{clamp, max, min};
use flate2::Compression;
use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use glam::{Vec2, Vec3};

use crate::format::{
    CLUSTER_HEADER_WORDS, ClusterHeader, ClusterVertex, DecodedCluster, MAX_CLUSTER_TRIS,
    MAX_CLUSTER_VERTS, PAGE_SIZE_BYTES, PAGE_WORDS, PageSpan,
};

fn invalid(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidMesh, path)
}

pub fn compute_base_quant_step(bounds_min: Vec3, bounds_max: Vec3) -> f32 {
    let extent = (bounds_max - bounds_min).max(Vec3::ZERO);
    let max_ext = extent.max_element();
    let max_abs = max(
        bounds_min.abs().max_element(),
        bounds_max.abs().max_element(),
    );
    let target = max(
        max(max_ext / 262_144.0, max_abs / 4_194_304.0),
        1.0 / 65_536.0,
    );
    let exp = target.log2().ceil() as i32;
    2.0_f32.powi(exp.clamp(-16, 16))
}

fn normal_to_oct(normal: Vec3) -> (f32, f32) {
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
    if unit.z < 0.0 {
        ((1.0 - v.abs()) * sign(u), (1.0 - u.abs()) * sign(v))
    } else {
        (u, v)
    }
}

fn oct_to_normal(px: f32, py: f32) -> Vec3 {
    let px = clamp(px, -1.0, 1.0);
    let py = clamp(py, -1.0, 1.0);
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

pub fn encode_octahedral(normal: Vec3) -> [i16; 2] {
    let (ox, oy) = normal_to_oct(normal);
    [
        (clamp(ox, -1.0, 1.0) * 32767.0).round() as i16,
        (clamp(oy, -1.0, 1.0) * 32767.0).round() as i16,
    ]
}

pub fn decode_octahedral(encoded: [i16; 2]) -> Vec3 {
    oct_to_normal(
        f32::from(encoded[0]) / 32767.0,
        f32::from(encoded[1]) / 32767.0,
    )
}

fn clean_uv_coord(value: f32) -> f32 {
    if value.is_finite() && value.abs() <= 256.0 {
        value
    } else {
        0.0
    }
}

fn cluster_uv_grid(vertices: &[ClusterVertex]) -> (f32, f32, f32) {
    let mut min_u = f32::INFINITY;
    let mut min_v = f32::INFINITY;
    let mut max_u = f32::NEG_INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for v in vertices {
        let u = clean_uv_coord(v.uv.x);
        let w = clean_uv_coord(v.uv.y);
        min_u = min(min_u, u);
        min_v = min(min_v, w);
        max_u = max(max_u, u);
        max_v = max(max_v, w);
    }
    if !min_u.is_finite() || !min_v.is_finite() || !max_u.is_finite() || !max_v.is_finite() {
        return (0.0, 0.0, 1.0 / 4096.0);
    }
    let uv_span = max(max_u - min_u, max_v - min_v);
    let uv_step = max(uv_span / 4095.0, 1.0 / 4096.0);
    (min_u, min_v, uv_step)
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

fn pack_rgb565(color: [f32; 3]) -> u16 {
    let r5 = (clamp(color[0], 0.0, 1.0).sqrt() * 31.0).round() as u16;
    let g6 = (clamp(color[1], 0.0, 1.0).sqrt() * 63.0).round() as u16;
    let b5 = (clamp(color[2], 0.0, 1.0).sqrt() * 31.0).round() as u16;
    (r5 & 0x1F) | ((g6 & 0x3F) << 5) | ((b5 & 0x1F) << 11)
}

fn unpack_rgb565(rgb565: u16) -> [f32; 3] {
    let r = f32::from((rgb565 & 0x1F) as u8) / 31.0;
    let g = f32::from(((rgb565 >> 5) & 0x3F) as u8) / 63.0;
    let b = f32::from(((rgb565 >> 11) & 0x1F) as u8) / 31.0;
    [r * r, g * g, b * b]
}

fn push_bytes_as_le_words(bytes: &[u8], out: &mut Vec<u32>) {
    for chunk in bytes.chunks(4) {
        let mut buf = [0_u8; 4];
        buf[..chunk.len()].copy_from_slice(chunk);
        out.push(u32::from_le_bytes(buf));
    }
}

fn read_le_byte(words: &[u32], byte_idx: usize) -> u8 {
    ((words[byte_idx >> 2] >> ((byte_idx & 3) * 8)) & 0xFF) as u8
}

fn read_u24_le(words: &[u32], item_idx: usize) -> u32 {
    let b = item_idx * 3;
    let b0 = u32::from(read_le_byte(words, b));
    let b1 = u32::from(read_le_byte(words, b + 1));
    let b2 = u32::from(read_le_byte(words, b + 2));
    b0 | (b1 << 8) | (b2 << 16)
}

fn read_u16_le(words: &[u32], u16_idx: usize) -> u16 {
    let w = words[u16_idx >> 1];
    if (u16_idx & 1) == 0 {
        (w & 0xFFFF) as u16
    } else {
        (w >> 16) as u16
    }
}

fn encode_normal_ao_words(vertices: &[ClusterVertex], out: &mut Vec<u32>) {
    let mut norm_bytes = Vec::with_capacity(vertices.len() * 3);
    for v in vertices {
        let (ox, oy) = normal_to_oct(v.normal);
        let qx = ((clamp(ox, -1.0, 1.0) * 511.0).round() as i32 + 511).clamp(0, 1022) as u32;
        let qy = ((clamp(oy, -1.0, 1.0) * 511.0).round() as i32 + 511).clamp(0, 1022) as u32;
        let ao4 = (clamp(v.ao, 0.0, 1.0) * 15.0).round() as u32 & 0xF;
        let packed24 = (qx & 0x3FF) | ((qy & 0x3FF) << 10) | (ao4 << 20);
        norm_bytes.push((packed24 & 0xFF) as u8);
        norm_bytes.push(((packed24 >> 8) & 0xFF) as u8);
        norm_bytes.push(((packed24 >> 16) & 0xFF) as u8);
    }
    push_bytes_as_le_words(&norm_bytes, out);
}

fn encode_uv_words(
    vertices: &[ClusterVertex],
    uv_min_u: f32,
    uv_min_v: f32,
    uv_step: f32,
    out: &mut Vec<u32>,
) {
    let mut uv_bytes = Vec::with_capacity(vertices.len() * 3);
    for v in vertices {
        let qu = clamp(
            ((clean_uv_coord(v.uv.x) - uv_min_u) / uv_step).round(),
            0.0,
            4095.0,
        ) as u32;
        let qv = clamp(
            ((clean_uv_coord(v.uv.y) - uv_min_v) / uv_step).round(),
            0.0,
            4095.0,
        ) as u32;
        let packed24 = (qu & 0xFFF) | ((qv & 0xFFF) << 12);
        uv_bytes.push((packed24 & 0xFF) as u8);
        uv_bytes.push(((packed24 >> 8) & 0xFF) as u8);
        uv_bytes.push(((packed24 >> 16) & 0xFF) as u8);
    }
    push_bytes_as_le_words(&uv_bytes, out);
}

fn encode_color_words(vertices: &[ClusterVertex], out: &mut Vec<u32>) {
    for pair in vertices.chunks(2) {
        let lo = pack_rgb565(pair[0].color);
        let hi = if pair.len() > 1 {
            pack_rgb565(pair[1].color)
        } else {
            0
        };
        out.push(u32::from(lo) | (u32::from(hi) << 16));
    }
}

pub fn encode_cluster_words(
    vertices: &[ClusterVertex],
    triangles: &[[u8; 3]],
    base_step: f32,
    out: &mut Vec<u32>,
) -> Result<(usize, bool), Error> {
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
    let vert_limit = vertices.len() as u8;
    for (ti, &[i0, i1, i2]) in triangles.iter().enumerate() {
        if i0 >= vert_limit || i1 >= vert_limit || i2 >= vert_limit {
            return Err(invalid(format!("$.cluster.triangles[{ti}]"))
                .at_most(u64::from(vert_limit.saturating_sub(1)))
                .got([i0, i1, i2]));
        }
    }
    let vc = vertices.len();
    let tc = triangles.len();
    let (origin, step, q_coords) = choose_cluster_grid(vertices, base_step);
    let (uv_min_u, uv_min_v, uv_step) = cluster_uv_grid(vertices);
    let has_color = vertices.iter().any(|v| {
        (v.color[0] - 1.0).abs() > 0.01
            || (v.color[1] - 1.0).abs() > 0.01
            || (v.color[2] - 1.0).abs() > 0.01
    });
    let pos_words = (vc * 3).div_ceil(2);
    let tri_words = (tc * 3).div_ceil(4);
    let norm_words = (vc * 3).div_ceil(4);
    let uv_words = (vc * 3).div_ceil(4);
    let col_words = if has_color { vc.div_ceil(2) } else { 0 };
    let start_len = out.len();
    out.reserve(
        CLUSTER_HEADER_WORDS + pos_words + tri_words + norm_words + uv_words + col_words,
    );
    out.extend_from_slice(&[
        origin.x.to_bits(),
        origin.y.to_bits(),
        origin.z.to_bits(),
        step.to_bits(),
        uv_min_u.to_bits(),
        uv_min_v.to_bits(),
        uv_step.to_bits(),
    ]);
    let mut pos_u16 = Vec::with_capacity(vc * 3);
    for [qx, qy, qz] in q_coords {
        pos_u16.push(qx);
        pos_u16.push(qy);
        pos_u16.push(qz);
    }
    for pair in pos_u16.chunks(2) {
        let lo = pair[0];
        let hi = if pair.len() > 1 { pair[1] } else { 0 };
        out.push(u32::from(lo) | (u32::from(hi) << 16));
    }
    let mut tri_bytes = Vec::with_capacity(tc * 3);
    for &[i0, i1, i2] in triangles {
        tri_bytes.push(i0);
        tri_bytes.push(i1);
        tri_bytes.push(i2);
    }
    push_bytes_as_le_words(&tri_bytes, out);
    encode_normal_ao_words(vertices, out);
    encode_uv_words(vertices, uv_min_u, uv_min_v, uv_step, out);
    if has_color {
        encode_color_words(vertices, out);
    }
    Ok((out.len() - start_len, has_color))
}

struct ClusterGridHeader {
    origin: Vec3,
    step: f32,
    uv_min_u: f32,
    uv_min_v: f32,
    uv_step: f32,
}

fn decode_cluster_vertices(
    grid: &ClusterGridHeader,
    vert_count: usize,
    pos_words: &[u32],
    norm_words: &[u32],
    uv_words: &[u32],
    col_words: &[u32],
    has_color: bool,
) -> Vec<ClusterVertex> {
    let mut vertices = Vec::with_capacity(vert_count);
    for vi in 0..vert_count {
        let pbase = vi * 3;
        let qx = read_u16_le(pos_words, pbase);
        let qy = read_u16_le(pos_words, pbase + 1);
        let qz = read_u16_le(pos_words, pbase + 2);
        let np = read_u24_le(norm_words, vi);
        let nx = (np & 0x3FF) as i32 - 511;
        let ny = ((np >> 10) & 0x3FF) as i32 - 511;
        let ao = f32::from(((np >> 20) & 0xF) as u8) / 15.0;
        let up = read_u24_le(uv_words, vi);
        let qu = (up & 0xFFF) as f32;
        let qv = ((up >> 12) & 0xFFF) as f32;
        let color = if has_color {
            let cw = col_words[vi >> 1];
            let rgb565 = if (vi & 1) == 0 {
                (cw & 0xFFFF) as u16
            } else {
                (cw >> 16) as u16
            };
            unpack_rgb565(rgb565)
        } else {
            [1.0, 1.0, 1.0]
        };
        vertices.push(ClusterVertex {
            position: dequantize_position(qx, qy, qz, grid.origin, grid.step),
            normal: oct_to_normal(nx as f32 / 511.0, ny as f32 / 511.0),
            uv: Vec2::new(
                grid.uv_min_u + qu * grid.uv_step,
                grid.uv_min_v + qv * grid.uv_step,
            ),
            color,
            ao,
        });
    }
    vertices
}

fn decode_cluster_triangles(
    tri_words: &[u32],
    tri_count: usize,
    vert_limit: u8,
) -> Result<Vec<[u8; 3]>, Error> {
    let mut triangles = Vec::with_capacity(tri_count);
    for ti in 0..tri_count {
        let base = ti * 3;
        let i0 = read_le_byte(tri_words, base);
        let i1 = read_le_byte(tri_words, base + 1);
        let i2 = read_le_byte(tri_words, base + 2);
        if i0 >= vert_limit || i1 >= vert_limit || i2 >= vert_limit {
            return Err(invalid(format!("$.cluster.triangles[{ti}]"))
                .at_most(u64::from(vert_limit.saturating_sub(1)))
                .got([i0, i1, i2]));
        }
        triangles.push([i0, i1, i2]);
    }
    Ok(triangles)
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
    let has_color = header.has_vertex_color();
    let pos_words_len = (vert_count * 3).div_ceil(2);
    let tri_words_len = (tri_count * 3).div_ceil(4);
    let norm_words_len = (vert_count * 3).div_ceil(4);
    let uv_words_len = (vert_count * 3).div_ceil(4);
    let col_words_len = if has_color { vert_count.div_ceil(2) } else { 0 };
    let needed = CLUSTER_HEADER_WORDS
        + pos_words_len
        + tri_words_len
        + norm_words_len
        + uv_words_len
        + col_words_len;
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
    let grid = ClusterGridHeader {
        origin: Vec3::new(
            f32::from_bits(page_words[offset]),
            f32::from_bits(page_words[offset + 1]),
            f32::from_bits(page_words[offset + 2]),
        ),
        step: f32::from_bits(page_words[offset + 3]),
        uv_min_u: f32::from_bits(page_words[offset + 4]),
        uv_min_v: f32::from_bits(page_words[offset + 5]),
        uv_step: f32::from_bits(page_words[offset + 6]),
    };
    if !grid.origin.is_finite()
        || !grid.step.is_finite()
        || grid.step <= 0.0
        || !grid.uv_min_u.is_finite()
        || !grid.uv_min_v.is_finite()
        || !grid.uv_step.is_finite()
        || grid.uv_step <= 0.0
    {
        return Err(invalid("$.cluster.grid").expected(Expected::Finite));
    }
    let pos_start = offset + CLUSTER_HEADER_WORDS;
    let tri_start = pos_start + pos_words_len;
    let norm_start = tri_start + tri_words_len;
    let uv_start = norm_start + norm_words_len;
    let col_start = uv_start + uv_words_len;
    let vertices = decode_cluster_vertices(
        &grid,
        vert_count,
        &page_words[pos_start..tri_start],
        &page_words[norm_start..uv_start],
        &page_words[uv_start..col_start],
        &page_words[col_start..end],
        has_color,
    );
    let triangles = decode_cluster_triangles(
        &page_words[tri_start..norm_start],
        tri_count,
        vert_count as u8,
    )?;
    Ok(DecodedCluster {
        origin: grid.origin,
        step: grid.step,
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
    let mut enc = DeflateEncoder::new(Vec::new(), Compression::best());
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
