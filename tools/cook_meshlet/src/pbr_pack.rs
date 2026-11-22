use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use cook::Refusal;
use cook_texture::{Encoding, Image, Texels, Wrap, byte, chain, decode, encode_level, linear_to_srgb, srgb_to_linear};
use engine_math::clamp;
use flate2::Compression;
use flate2::write::DeflateEncoder;
use serde::Serialize;

pub const PBR_MAGIC: [u8; 4] = *b"SPB2";
pub const PBR_VERSION: u32 = 2;
pub const ALBEDO_DIM: u32 = 1024;
pub const NORMAL_DIM: u32 = 1024;
pub const ORM_DIM: u32 = 512;
pub const BASE_DIM: u32 = 128;

struct UnifiedMatSpec {
    name: &'static str,
    pkg: &'static str,
    src_idx: usize,
    flags: u32,
    emissive: [f32; 3],
    bcf_ov: Option<[f32; 3]>,
    rough_ov: Option<f32>,
    metal_ov: Option<f32>,
}

const UNIFIED_MATS: [UnifiedMatSpec; 32] = [
    UnifiedMatSpec { name: "arch_stone_wall_01", pkg: "main", src_idx: 0, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "stone_trims_01", pkg: "main", src_idx: 1, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "metal_door", pkg: "main", src_idx: 2, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "brickwall_02", pkg: "main", src_idx: 3, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "stones_2ndfloor", pkg: "main", src_idx: 4, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "stone_trims_02", pkg: "main", src_idx: 5, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "ceiling_plaster_02", pkg: "main", src_idx: 6, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "window_frame_01", pkg: "main", src_idx: 7, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "stones_01_tile", pkg: "main", src_idx: 8, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "column_1stfloor", pkg: "main", src_idx: 9, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "column_head_1stfloor", pkg: "main", src_idx: 10, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "lamp_glass_01", pkg: "main", src_idx: 11, flags: 1 | 16, emissive: [2.2, 1.35, 0.58], bcf_ov: Some([0.85, 0.65, 0.42]), rough_ov: Some(0.35), metal_ov: Some(0.0) },
    UnifiedMatSpec { name: "light_bulb", pkg: "main", src_idx: 12, flags: 16, emissive: [9.0, 6.2, 3.0], bcf_ov: Some([1.0, 0.92, 0.78]), rough_ov: Some(0.10), metal_ov: Some(0.0) },
    UnifiedMatSpec { name: "door_stoneframe_02", pkg: "main", src_idx: 13, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "glass", pkg: "main", src_idx: 14, flags: 1 | 8, emissive: [0.0; 3], bcf_ov: Some([0.03, 0.04, 0.05]), rough_ov: Some(0.06), metal_ov: Some(0.15) },
    UnifiedMatSpec { name: "door_stoneframe_01", pkg: "main", src_idx: 15, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "brickwall_01", pkg: "main", src_idx: 16, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "wood_door_01", pkg: "main", src_idx: 17, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "wood_01", pkg: "main", src_idx: 18, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "floor_01", pkg: "main", src_idx: 19, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "ornament_lion", pkg: "main", src_idx: 20, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "column_head_2ndfloor_02", pkg: "main", src_idx: 22, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "column_brickwall_01", pkg: "main", src_idx: 23, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "column_head_2ndfloor_03", pkg: "main", src_idx: 24, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "ceiling_plaster_01", pkg: "main", src_idx: 25, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "roof_tiles_01", pkg: "main", src_idx: 26, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "ornament_01", pkg: "main", src_idx: 27, flags: 0, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "curtain_01", pkg: "curtains", src_idx: 1, flags: 1 | 2, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "curtain_03", pkg: "curtains", src_idx: 2, flags: 1 | 2, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "curtain_02", pkg: "curtains", src_idx: 3, flags: 1 | 2, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "IvyLeaf", pkg: "ivy", src_idx: 0, flags: 1 | 4, emissive: [0.0; 3], bcf_ov: None, rough_ov: None, metal_ov: None },
    UnifiedMatSpec { name: "Lianas", pkg: "ivy", src_idx: 1, flags: 1, emissive: [0.0; 3], bcf_ov: Some([0.0985, 0.0264, 0.008]), rough_ov: Some(0.65), metal_ov: Some(0.0) },
];

#[derive(Clone, Debug, Serialize)]
pub struct PackPbrOutput {
    pub status: &'static str,
    pub base_path: String,
    pub hd_path: String,
    pub mat_count: u32,
    pub layer_count: u32,
    pub base_bytes: usize,
    pub hd_bytes: usize,
    pub elapsed_ms: f64,
}

struct MatRecord {
    layer: Option<u32>,
    flags: u32,
    base_col: [f32; 3],
    rough: f32,
    metal: f32,
    emissive: [f32; 3],
}

struct LayerJob {
    uid: usize,
    bc_path: PathBuf,
    nm_path: Option<PathBuf>,
    mr_path: Option<PathBuf>,
}

struct EncodedLayer {
    uid: usize,
    mean_lin: [f32; 3],
    bc_base: Vec<u8>,
    nm_base: Vec<u8>,
    orm_base: Vec<u8>,
    mid_blob: Vec<u8>,
    hi_blob: Vec<u8>,
}

fn deflate_bytes(data: &[u8]) -> Result<Vec<u8>, Refusal> {
    let mut enc = DeflateEncoder::new(Vec::new(), Compression::new(6));
    enc.write_all(data)
        .map_err(|e| Refusal::new("CompressFailed", "pbr_pack", format!("{e}")))?;
    enc.finish()
        .map_err(|e| Refusal::new("CompressFailed", "pbr_pack", format!("{e}")))
}

fn load_png_image(path: &Path) -> Result<Image, Refusal> {
    let bytes = std::fs::read(path)
        .map_err(|e| Refusal::new("ReadFailed", &path.display().to_string(), format!("{e}")))?;
    decode(&bytes)
        .map_err(|e| Refusal::new("DecodePng", &path.display().to_string(), e))
}

fn resize_image(img: &Image, dst_dim: u32, texels: Texels) -> Image {
    if img.width == dst_dim && img.height == dst_dim {
        return img.clone();
    }
    if img.width == img.height
        && img.width.is_power_of_two()
        && dst_dim.is_power_of_two()
        && img.width > dst_dim
    {
        let mips = chain(texels, img, [Wrap::Repeat, Wrap::Repeat]);
        if let Some(found) = mips.into_iter().find(|m| m.width == dst_dim && m.height == dst_dim) {
            return found;
        }
    }
    let linear: [f32; 256] = std::array::from_fn(|v| srgb_to_linear(v as u8));
    let (sw, sh) = (img.width as usize, img.height as usize);
    let d = dst_dim as usize;
    let mut pixels = Vec::with_capacity(d * d * 4);
    for y in 0..d {
        let sy = clamp((y as f32 + 0.5) * (sh as f32) / (d as f32) - 0.5, 0.0, (sh - 1) as f32);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let fy = sy - y0 as f32;
        for x in 0..d {
            let sx = clamp((x as f32 + 0.5) * (sw as f32) / (d as f32) - 0.5, 0.0, (sw - 1) as f32);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let fx = sx - x0 as f32;
            let w = [
                (1.0 - fx) * (1.0 - fy),
                fx * (1.0 - fy),
                (1.0 - fx) * fy,
                fx * fy,
            ];
            let pts = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];
            let mut acc = [0.0_f32; 4];
            for (&(px, py), &wt) in pts.iter().zip(&w) {
                let off = (py * sw + px) * 4;
                for c in 0..4 {
                    let v = match (texels, c) {
                        (Texels::Srgb, 0..=2) => linear[usize::from(img.pixels[off + c])],
                        (Texels::Normal, 0..=2) => f32::from(img.pixels[off + c]) / 255.0 * 2.0 - 1.0,
                        _ => f32::from(img.pixels[off + c]) / 255.0,
                    };
                    acc[c] += v * wt;
                }
            }
            let out = match texels {
                Texels::Srgb => [
                    linear_to_srgb(acc[0]),
                    linear_to_srgb(acc[1]),
                    linear_to_srgb(acc[2]),
                    byte(acc[3]),
                ],
                Texels::Normal => {
                    let len = (acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]).sqrt();
                    let u = if len > 1.0e-6 {
                        [acc[0] / len, acc[1] / len, acc[2] / len]
                    } else {
                        [0.0, 0.0, 1.0]
                    };
                    [
                        byte(u[0] * 0.5 + 0.5),
                        byte(u[1] * 0.5 + 0.5),
                        byte(u[2] * 0.5 + 0.5),
                        255,
                    ]
                }
                Texels::Linear => acc.map(byte),
            };
            pixels.extend_from_slice(&out);
        }
    }
    Image {
        width: dst_dim,
        height: dst_dim,
        pixels,
    }
}

fn solid_image(dim: u32, rgba: [u8; 4]) -> Image {
    let count = (dim as usize) * (dim as usize);
    let mut pixels = Vec::with_capacity(count * 4);
    for _ in 0..count {
        pixels.extend_from_slice(&rgba);
    }
    Image {
        width: dim,
        height: dim,
        pixels,
    }
}

fn mean_linear_rgb(img: &Image) -> [f32; 3] {
    let linear: [f32; 256] = std::array::from_fn(|v| srgb_to_linear(v as u8));
    let mut sum = [0.0_f64; 3];
    let n = ((img.width as usize) * (img.height as usize)).max(1) as f64;
    for px in img.pixels.as_chunks::<4>().0 {
        sum[0] += f64::from(linear[usize::from(px[0])]);
        sum[1] += f64::from(linear[usize::from(px[1])]);
        sum[2] += f64::from(linear[usize::from(px[2])]);
    }
    [(sum[0] / n) as f32, (sum[1] / n) as f32, (sum[2] / n) as f32]
}

fn encode_layer_job(job: &LayerJob) -> Result<EncodedLayer, Refusal> {
    let bc_raw = load_png_image(&job.bc_path)?;
    let bc_top = resize_image(&bc_raw, ALBEDO_DIM, Texels::Srgb);
    let mean_lin = mean_linear_rgb(&bc_top);
    let bc_chain = chain(Texels::Srgb, &bc_top, [Wrap::Repeat, Wrap::Repeat]);
    let bc_levels: Vec<Vec<u8>> = bc_chain
        .iter()
        .map(|lvl| encode_level(Encoding::Bc7Srgb, lvl, false))
        .collect();

    let nm_top = match &job.nm_path {
        Some(p) if p.is_file() => resize_image(&load_png_image(p)?, NORMAL_DIM, Texels::Normal),
        _ => solid_image(NORMAL_DIM, [128, 128, 255, 255]),
    };
    let nm_chain = chain(Texels::Normal, &nm_top, [Wrap::Repeat, Wrap::Repeat]);
    let nm_levels: Vec<Vec<u8>> = nm_chain
        .iter()
        .map(|lvl| encode_level(Encoding::Bc5, lvl, false))
        .collect();

    let mut orm_top = match &job.mr_path {
        Some(p) if p.is_file() => resize_image(&load_png_image(p)?, ORM_DIM, Texels::Linear),
        _ => solid_image(ORM_DIM, [255, 200, 0, 255]),
    };
    for px in orm_top.pixels.as_chunks_mut::<4>().0 {
        px[0] = 255;
        px[3] = 255;
    }
    let orm_chain = chain(Texels::Linear, &orm_top, [Wrap::Repeat, Wrap::Repeat]);
    let orm_levels: Vec<Vec<u8>> = orm_chain
        .iter()
        .map(|lvl| encode_level(Encoding::Bc7, lvl, false))
        .collect();

    let mut bc_tail = Vec::new();
    for lvl in bc_levels.iter().skip(3) {
        bc_tail.extend_from_slice(lvl);
    }
    let mut nm_tail = Vec::new();
    for lvl in nm_levels.iter().skip(3) {
        nm_tail.extend_from_slice(lvl);
    }
    let mut orm_tail = Vec::new();
    for lvl in orm_levels.iter().skip(2) {
        orm_tail.extend_from_slice(lvl);
    }

    let mut mid_raw = Vec::new();
    mid_raw.extend_from_slice(&bc_levels[1]);
    mid_raw.extend_from_slice(&bc_levels[2]);
    mid_raw.extend_from_slice(&nm_levels[1]);
    mid_raw.extend_from_slice(&nm_levels[2]);
    mid_raw.extend_from_slice(&orm_levels[0]);
    mid_raw.extend_from_slice(&orm_levels[1]);

    let mut hi_raw = Vec::new();
    hi_raw.extend_from_slice(&bc_levels[0]);
    hi_raw.extend_from_slice(&nm_levels[0]);

    Ok(EncodedLayer {
        uid: job.uid,
        mean_lin,
        bc_base: deflate_bytes(&bc_tail)?,
        nm_base: deflate_bytes(&nm_tail)?,
        orm_base: deflate_bytes(&orm_tail)?,
        mid_blob: deflate_bytes(&mid_raw)?,
        hi_blob: deflate_bytes(&hi_raw)?,
    })
}

fn gltf_tex_path(
    gltf: &serde_json::Value,
    pkg_dir: &Path,
    tex_info: Option<&serde_json::Value>,
) -> Option<PathBuf> {
    let tex_idx = tex_info?.get("index")?.as_u64()? as usize;
    let src_idx = gltf.get("textures")?.get(tex_idx)?.get("source")?.as_u64()? as usize;
    let uri = gltf.get("images")?.get(src_idx)?.get("uri")?.as_str()?;
    Some(pkg_dir.join(uri))
}

fn load_pkg_json(path: &Path) -> Result<(PathBuf, serde_json::Value), Refusal> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Refusal::new("ReadFailed", &path.display().to_string(), format!("{e}")))?;
    let val: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Refusal::new("ParseGltf", &path.display().to_string(), format!("{e}")))?;
    let dir = path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
    Ok((dir, val))
}

fn collect_materials_and_jobs(
    intel_sponza_dir: &Path,
) -> Result<(Vec<MatRecord>, Vec<LayerJob>), Refusal> {
    let main_pkg = load_pkg_json(&intel_sponza_dir.join("main_sponza/NewSponza_Main_glTF_003.gltf"))?;
    let curt_pkg = load_pkg_json(&intel_sponza_dir.join("pkg_a_curtains/NewSponza_Curtains_glTF.gltf"))?;
    let ivy_pkg = load_pkg_json(&intel_sponza_dir.join("pkg_b_ivy/NewSponza_IvyGrowth_glTF.gltf"))?;

    let mut mats = Vec::with_capacity(UNIFIED_MATS.len());
    let mut jobs = Vec::new();
    for (uid, spec) in UNIFIED_MATS.iter().enumerate() {
        let (pdir, gltf) = match spec.pkg {
            "main" => &main_pkg,
            "curtains" => &curt_pkg,
            _ => &ivy_pkg,
        };
        let mat_val = gltf
            .get("materials")
            .and_then(|v| v.get(spec.src_idx))
            .ok_or_else(|| {
                Refusal::new("MissingMaterial", spec.name, format!("index {}", spec.src_idx))
            })?;
        let pbr = mat_val.get("pbrMetallicRoughness");
        let bc_path = gltf_tex_path(gltf, pdir, pbr.and_then(|v| v.get("baseColorTexture")));
        let nm_path = gltf_tex_path(gltf, pdir, mat_val.get("normalTexture"));
        let mr_path = gltf_tex_path(
            gltf,
            pdir,
            pbr.and_then(|v| v.get("metallicRoughnessTexture")),
        );
        let rough = spec.rough_ov.unwrap_or_else(|| {
            pbr.and_then(|v| v.get("roughnessFactor"))
                .and_then(serde_json::Value::as_f64)
                .map_or(1.0, |f| f as f32)
        });
        let metal = spec.metal_ov.unwrap_or_else(|| {
            pbr.and_then(|v| v.get("metallicFactor"))
                .and_then(serde_json::Value::as_f64)
                .map_or(1.0, |f| f as f32)
        });
        let (layer, base_col) = if let Some(bc) = bc_path {
            let lidx = jobs.len() as u32;
            jobs.push(LayerJob {
                uid,
                bc_path: bc,
                nm_path,
                mr_path,
            });
            (Some(lidx), [0.6, 0.55, 0.5])
        } else {
            let col = spec.bcf_ov.unwrap_or_else(|| {
                let arr = pbr
                    .and_then(|v| v.get("baseColorFactor"))
                    .and_then(serde_json::Value::as_array);
                if let Some(a) = arr
                    && a.len() >= 3
                {
                    [
                        a[0].as_f64().unwrap_or(1.0) as f32,
                        a[1].as_f64().unwrap_or(1.0) as f32,
                        a[2].as_f64().unwrap_or(1.0) as f32,
                    ]
                } else {
                    [1.0, 1.0, 1.0]
                }
            });
            (None, col)
        };
        mats.push(MatRecord {
            layer,
            flags: spec.flags,
            base_col,
            rough,
            metal,
            emissive: spec.emissive,
        });
    }
    Ok((mats, jobs))
}

fn write_spbr_v2_files(
    mats: &[MatRecord],
    layers: &[EncodedLayer],
    out_dir: &Path,
) -> Result<(PathBuf, PathBuf, usize, usize), Refusal> {
    let header_bytes = 32_usize;
    let mat_bytes = mats.len() * 48;
    let base_table_bytes = layers.len() * 24;
    let hd_table_bytes = layers.len() * 32;
    let base_payload_start = header_bytes + mat_bytes + base_table_bytes + hd_table_bytes;

    let mut hd_out = Vec::new();
    let mut hd_mid_spans = Vec::with_capacity(layers.len());
    for layer in layers {
        let off = hd_out.len() as u64;
        let len = layer.mid_blob.len() as u32;
        hd_out.extend_from_slice(&layer.mid_blob);
        hd_mid_spans.push((off, len));
    }
    let mut hd_hi_spans = Vec::with_capacity(layers.len());
    for layer in layers {
        let off = hd_out.len() as u64;
        let len = layer.hi_blob.len() as u32;
        hd_out.extend_from_slice(&layer.hi_blob);
        hd_hi_spans.push((off, len));
    }

    let mut base_out = Vec::with_capacity(base_payload_start + 2 * 1024 * 1024);
    base_out.extend_from_slice(&PBR_MAGIC);
    for w in [
        PBR_VERSION,
        mats.len() as u32,
        layers.len() as u32,
        ALBEDO_DIM,
        NORMAL_DIM,
        ORM_DIM,
        BASE_DIM,
    ] {
        base_out.extend_from_slice(&w.to_le_bytes());
    }
    for m in mats {
        for f in [
            m.base_col[0],
            m.base_col[1],
            m.base_col[2],
            m.rough,
            m.emissive[0],
            m.emissive[1],
            m.emissive[2],
            m.metal,
        ] {
            base_out.extend_from_slice(&f.to_le_bytes());
        }
        let layer_u = m.layer.unwrap_or(u32::MAX);
        for u in [layer_u, m.flags, 3_u32, 0_u32] {
            base_out.extend_from_slice(&u.to_le_bytes());
        }
    }

    let mut base_payload = Vec::new();
    let mut cursor = base_payload_start as u32;
    for layer in layers {
        let bc_off = cursor;
        let bc_len = layer.bc_base.len() as u32;
        cursor += bc_len;
        base_payload.extend_from_slice(&layer.bc_base);
        let nm_off = cursor;
        let nm_len = layer.nm_base.len() as u32;
        cursor += nm_len;
        base_payload.extend_from_slice(&layer.nm_base);
        let orm_off = cursor;
        let orm_len = layer.orm_base.len() as u32;
        cursor += orm_len;
        base_payload.extend_from_slice(&layer.orm_base);
        for w in [bc_off, bc_len, nm_off, nm_len, orm_off, orm_len] {
            base_out.extend_from_slice(&w.to_le_bytes());
        }
    }
    for (&(mid_off, mid_len), &(hi_off, hi_len)) in hd_mid_spans.iter().zip(&hd_hi_spans) {
        base_out.extend_from_slice(&mid_off.to_le_bytes());
        base_out.extend_from_slice(&mid_len.to_le_bytes());
        base_out.extend_from_slice(&0_u32.to_le_bytes());
        base_out.extend_from_slice(&hi_off.to_le_bytes());
        base_out.extend_from_slice(&hi_len.to_le_bytes());
        base_out.extend_from_slice(&0_u32.to_le_bytes());
    }
    base_out.extend_from_slice(&base_payload);

    std::fs::create_dir_all(out_dir)
        .map_err(|e| Refusal::new("OutputDir", &out_dir.display().to_string(), format!("{e}")))?;
    let base_path = out_dir.join("sponza_pbr_base.bin");
    let hd_path = out_dir.join("sponza_pbr_hd.bin");
    std::fs::write(&base_path, &base_out)
        .map_err(|e| Refusal::new("WriteFailed", &base_path.display().to_string(), format!("{e}")))?;
    std::fs::write(&hd_path, &hd_out)
        .map_err(|e| Refusal::new("WriteFailed", &hd_path.display().to_string(), format!("{e}")))?;
    Ok((base_path, hd_path, base_out.len(), hd_out.len()))
}

pub fn pack_pbr(intel_sponza_dir: &Path, out_dir: &Path) -> Result<PackPbrOutput, Refusal> {
    let start = Instant::now();
    let (mut mats, jobs) = collect_materials_and_jobs(intel_sponza_dir)?;
    let mut layers = Vec::with_capacity(jobs.len());
    for job in &jobs {
        let encoded = encode_layer_job(job)?;
        mats[encoded.uid].base_col = encoded.mean_lin;
        layers.push(encoded);
    }
    let (base_path, hd_path, base_bytes, hd_bytes) = write_spbr_v2_files(&mats, &layers, out_dir)?;
    Ok(PackPbrOutput {
        status: "pass",
        base_path: base_path.display().to_string(),
        hd_path: hd_path.display().to_string(),
        mat_count: mats.len() as u32,
        layer_count: layers.len() as u32,
        base_bytes,
        hd_bytes,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}
