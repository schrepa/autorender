use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use cook::Refusal;
use cook_texture::{
    Encoding, Image, Texels, Wrap, byte, chain, decode, encode_level, linear_to_srgb,
    srgb_to_linear,
};
use engine_math::{clamp, max};
use flate2::Compression;
use flate2::write::DeflateEncoder;
use serde::{Deserialize, Serialize};

pub const PBR_MAGIC: [u8; 4] = *b"SPB2";
pub const PBR_VERSION: u32 = 2;

fn default_out_stem() -> String {
    "pbr".to_owned()
}

const fn default_albedo_dim() -> u32 {
    1024
}

const fn default_normal_dim() -> u32 {
    1024
}

const fn default_orm_dim() -> u32 {
    512
}

const fn default_base_dim() -> u32 {
    128
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialPackEntry {
    pub name: String,
    pub package: String,
    pub source_index: usize,
    #[serde(default)]
    pub flags: u32,
    #[serde(default)]
    pub emissive: [f32; 3],
    #[serde(default)]
    pub base_color: Option<[f32; 3]>,
    #[serde(default)]
    pub roughness: Option<f32>,
    #[serde(default)]
    pub metallic: Option<f32>,
    #[serde(default)]
    pub normal_scale: Option<f32>,
    #[serde(default)]
    pub normal_smooth_passes: u32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PbrPackSpec {
    #[serde(default = "default_out_stem")]
    pub out_stem: String,
    #[serde(default = "default_albedo_dim")]
    pub albedo_dim: u32,
    #[serde(default = "default_normal_dim")]
    pub normal_dim: u32,
    #[serde(default = "default_orm_dim")]
    pub orm_dim: u32,
    #[serde(default = "default_base_dim")]
    pub base_dim: u32,
    pub packages: BTreeMap<String, String>,
    pub materials: Vec<MaterialPackEntry>,
}

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

#[derive(Clone, Copy)]
struct PackDims {
    albedo: u32,
    normal: u32,
    orm: u32,
    base: u32,
}

impl PackDims {
    fn validate(self, file: &str) -> Result<(), Refusal> {
        for (label, dim) in [
            ("albedo_dim", self.albedo),
            ("normal_dim", self.normal),
            ("orm_dim", self.orm),
            ("base_dim", self.base),
        ] {
            if dim < 4 || !dim.is_power_of_two() {
                return Err(Refusal::new(
                    "InvalidSpec",
                    file,
                    format!("{label} ({dim}) must be a power of two >= 4"),
                ));
            }
        }
        if self.albedo < self.base || self.normal < self.base || self.orm < self.base {
            return Err(Refusal::new(
                "InvalidSpec",
                file,
                "albedo_dim, normal_dim, and orm_dim must be >= base_dim",
            ));
        }
        Ok(())
    }

    fn base_mip(self) -> u32 {
        self.albedo.trailing_zeros() - self.base.trailing_zeros()
    }
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
    occ_path: Option<PathBuf>,
    normal_scale: f32,
    normal_smooth_passes: u32,
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
    decode(&bytes).map_err(|e| Refusal::new("DecodePng", &path.display().to_string(), e))
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
        if let Some(found) = mips
            .into_iter()
            .find(|m| m.width == dst_dim && m.height == dst_dim)
        {
            return found;
        }
    }
    let linear: [f32; 256] = std::array::from_fn(|v| srgb_to_linear(v as u8));
    let (sw, sh) = (img.width as usize, img.height as usize);
    let d = dst_dim as usize;
    let mut pixels = Vec::with_capacity(d * d * 4);
    for y in 0..d {
        let sy = clamp(
            (y as f32 + 0.5) * (sh as f32) / (d as f32) - 0.5,
            0.0,
            (sh - 1) as f32,
        );
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let fy = sy - y0 as f32;
        for x in 0..d {
            let sx = clamp(
                (x as f32 + 0.5) * (sw as f32) / (d as f32) - 0.5,
                0.0,
                (sw - 1) as f32,
            );
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
                        (Texels::Normal, 0..=2) => {
                            f32::from(img.pixels[off + c]) / 255.0 * 2.0 - 1.0
                        }
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

struct NormalLevel {
    dim: u32,
    vecs: Vec<[f32; 3]>,
}

fn decode_normal_vecs(img: &Image, normal_scale: f32) -> NormalLevel {
    let mut vecs = Vec::with_capacity((img.width as usize) * (img.height as usize));
    for px in img.pixels.as_chunks::<4>().0 {
        let nx = (f32::from(px[0]) / 255.0 * 2.0 - 1.0) * normal_scale;
        let ny = (f32::from(px[1]) / 255.0 * 2.0 - 1.0) * normal_scale;
        let nz = max(f32::from(px[2]) / 255.0 * 2.0 - 1.0, 0.05);
        let len = max((nx * nx + ny * ny + nz * nz).sqrt(), 1.0e-6);
        vecs.push([nx / len, ny / len, nz / len]);
    }
    NormalLevel {
        dim: img.width,
        vecs,
    }
}

fn halve_normal_vecs(src: &NormalLevel) -> NormalLevel {
    let sw = src.dim as usize;
    let dw = (sw / 2).max(1);
    let mut vecs = Vec::with_capacity(dw * dw);
    for y in 0..dw {
        let y0 = (y * 2).min(sw - 1);
        let y1 = (y * 2 + 1).min(sw - 1);
        for x in 0..dw {
            let x0 = (x * 2).min(sw - 1);
            let x1 = (x * 2 + 1).min(sw - 1);
            let a = src.vecs[y0 * sw + x0];
            let b = src.vecs[y0 * sw + x1];
            let c = src.vecs[y1 * sw + x0];
            let d = src.vecs[y1 * sw + x1];
            vecs.push([
                (a[0] + b[0] + c[0] + d[0]) * 0.25,
                (a[1] + b[1] + c[1] + d[1]) * 0.25,
                (a[2] + b[2] + c[2] + d[2]) * 0.25,
            ]);
        }
    }
    NormalLevel {
        dim: dw as u32,
        vecs,
    }
}

struct NormalPyramid {
    images: Vec<Image>,
    toksvig_vars: Vec<(u32, Vec<f32>)>,
    top_vecs: NormalLevel,
}

fn smooth_normal_level(src: &NormalLevel) -> NormalLevel {
    let w = src.dim as usize;
    let mut horiz = vec![[0.0_f32; 3]; w * w];
    for y in 0..w {
        for x in 0..w {
            let xm = (x + w - 1) % w;
            let xp = (x + 1) % w;
            let a = src.vecs[y * w + xm];
            let b = src.vecs[y * w + x];
            let c = src.vecs[y * w + xp];
            horiz[y * w + x] = [
                a[0] * 0.25 + b[0] * 0.5 + c[0] * 0.25,
                a[1] * 0.25 + b[1] * 0.5 + c[1] * 0.25,
                a[2] * 0.25 + b[2] * 0.5 + c[2] * 0.25,
            ];
        }
    }
    let mut vecs = vec![[0.0_f32; 3]; w * w];
    for y in 0..w {
        let ym = (y + w - 1) % w;
        let yp = (y + 1) % w;
        for x in 0..w {
            let a = horiz[ym * w + x];
            let b = horiz[y * w + x];
            let c = horiz[yp * w + x];
            vecs[y * w + x] = [
                a[0] * 0.25 + b[0] * 0.5 + c[0] * 0.25,
                a[1] * 0.25 + b[1] * 0.5 + c[1] * 0.25,
                a[2] * 0.25 + b[2] * 0.5 + c[2] * 0.25,
            ];
        }
    }
    NormalLevel {
        dim: src.dim,
        vecs,
    }
}

fn build_normal_pyramid(
    nm_path: Option<&PathBuf>,
    target_dim: u32,
    normal_scale: f32,
    smooth_passes: u32,
) -> Result<NormalPyramid, Refusal> {
    let src_img = match nm_path {
        Some(p) if p.is_file() => load_png_image(p)?,
        _ => solid_image(target_dim, [128, 128, 255, 255]),
    };
    let mut cur = if src_img.width == src_img.height
        && src_img.width.is_power_of_two()
        && src_img.width >= target_dim
    {
        decode_normal_vecs(&src_img, normal_scale)
    } else {
        let resized = resize_image(&src_img, target_dim, Texels::Normal);
        decode_normal_vecs(&resized, normal_scale)
    };
    while cur.dim > target_dim {
        cur = halve_normal_vecs(&cur);
    }
    for _ in 0..smooth_passes {
        cur = smooth_normal_level(&cur);
    }
    let mut nm_images = Vec::new();
    let mut toksvig_vars = Vec::new();
    let mut top_vecs = NormalLevel {
        dim: cur.dim,
        vecs: cur.vecs.clone(),
    };
    let mut is_first = true;
    loop {
        let d = cur.dim;
        let mut pixels = Vec::with_capacity((d as usize) * (d as usize) * 4);
        let mut vars = Vec::with_capacity((d as usize) * (d as usize));
        for &v in &cur.vecs {
            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let r_clamped = clamp(r, 0.05, 1.0);
            let var = clamp((1.0 - r_clamped) / max(r_clamped, 0.08), 0.0, 2.5);
            vars.push(var);
            let damp = clamp(r_clamped * r_clamped, 0.30, 1.0);
            let ux = v[0] * damp;
            let uy = v[1] * damp;
            let uz = max(v[2], 0.15);
            let ulen = max((ux * ux + uy * uy + uz * uz).sqrt(), 1.0e-6);
            pixels.extend_from_slice(&[
                byte((ux / ulen) * 0.5 + 0.5),
                byte((uy / ulen) * 0.5 + 0.5),
                byte((uz / ulen) * 0.5 + 0.5),
                255,
            ]);
        }
        if is_first {
            top_vecs = NormalLevel {
                dim: d,
                vecs: cur.vecs.clone(),
            };
            is_first = false;
        }
        nm_images.push(Image {
            width: d,
            height: d,
            pixels,
        });
        toksvig_vars.push((d, vars));
        if d <= 1 {
            break;
        }
        cur = halve_normal_vecs(&cur);
    }
    Ok(NormalPyramid {
        images: nm_images,
        toksvig_vars,
        top_vecs,
    })
}

fn compute_cavity_ao(mut level: NormalLevel, dst_dim: u32) -> Vec<f32> {
    while level.dim > dst_dim {
        level = halve_normal_vecs(&level);
    }
    let w = level.dim as usize;
    let mut out = vec![1.0_f32; w * w];
    for y in 0..w {
        let ym = (y + w - 1) % w;
        let yp = (y + 1) % w;
        for x in 0..w {
            let xm = (x + w - 1) % w;
            let xp = (x + 1) % w;
            let dx = level.vecs[y * w + xp][0] - level.vecs[y * w + xm][0];
            let dy = level.vecs[yp * w + x][1] - level.vecs[ym * w + x][1];
            let crevice = clamp(-(dx + dy) * 0.32, 0.0, 0.42);
            out[y * w + x] = 1.0 - crevice;
        }
    }
    out
}

fn encode_layer_job(job: &LayerJob, dims: PackDims) -> Result<EncodedLayer, Refusal> {
    let bc_raw = load_png_image(&job.bc_path)?;
    let bc_top = resize_image(&bc_raw, dims.albedo, Texels::Srgb);
    let mean_lin = mean_linear_rgb(&bc_top);
    let bc_chain = chain(Texels::Srgb, &bc_top, [Wrap::Repeat, Wrap::Repeat]);
    let bc_levels: Vec<Vec<u8>> = bc_chain
        .iter()
        .map(|lvl| encode_level(Encoding::Bc7Srgb, lvl, false))
        .collect();

    let nm_pyr = build_normal_pyramid(
        job.nm_path.as_ref(),
        dims.normal,
        job.normal_scale,
        job.normal_smooth_passes,
    )?;
    let nm_levels: Vec<Vec<u8>> = nm_pyr
        .images
        .iter()
        .map(|lvl| encode_level(Encoding::Bc5, lvl, false))
        .collect();

    let mut orm_top = match &job.mr_path {
        Some(p) if p.is_file() => resize_image(&load_png_image(p)?, dims.orm, Texels::Linear),
        _ => solid_image(dims.orm, [255, 200, 0, 255]),
    };
    let occ_top = match &job.occ_path {
        Some(p) if p.is_file() => Some(resize_image(&load_png_image(p)?, dims.orm, Texels::Linear)),
        _ => None,
    };
    let cavity = compute_cavity_ao(nm_pyr.top_vecs, dims.orm);
    for (idx, px) in orm_top.pixels.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let occ_v = occ_top
            .as_ref()
            .map_or(1.0, |img| f32::from(img.pixels[idx * 4]) / 255.0);
        let cav_v = cavity.get(idx).copied().unwrap_or(1.0);
        px[0] = byte(clamp(occ_v * cav_v, 0.35, 1.0));
        px[3] = 255;
    }
    let mut orm_chain = chain(Texels::Linear, &orm_top, [Wrap::Repeat, Wrap::Repeat]);
    for lvl in &mut orm_chain {
        if let Some((_, vars)) = nm_pyr.toksvig_vars.iter().find(|(d, _)| *d == lvl.width) {
            for (px, &var) in lvl.pixels.as_chunks_mut::<4>().0.iter_mut().zip(vars.iter()) {
                let r = f32::from(px[1]) / 255.0;
                let r_tok = clamp(r * r + var * 0.30, 0.0016, 0.96).sqrt();
                px[1] = byte(r_tok);
            }
        }
    }
    let orm_levels: Vec<Vec<u8>> = orm_chain
        .iter()
        .map(|lvl| encode_level(Encoding::Bc7, lvl, false))
        .collect();

    let bc_skip = (dims.albedo.trailing_zeros() - dims.base.trailing_zeros()) as usize;
    let nm_skip = (dims.normal.trailing_zeros() - dims.base.trailing_zeros()) as usize;
    let orm_skip = (dims.orm.trailing_zeros() - dims.base.trailing_zeros()) as usize;

    let mut bc_tail = Vec::new();
    for lvl in bc_levels.iter().skip(bc_skip) {
        bc_tail.extend_from_slice(lvl);
    }
    let mut nm_tail = Vec::new();
    for lvl in nm_levels.iter().skip(nm_skip) {
        nm_tail.extend_from_slice(lvl);
    }
    let mut orm_tail = Vec::new();
    for lvl in orm_levels.iter().skip(orm_skip) {
        orm_tail.extend_from_slice(lvl);
    }

    let mut mid_raw = Vec::new();
    for lvl in bc_levels.iter().take(bc_skip).skip(1) {
        mid_raw.extend_from_slice(lvl);
    }
    for lvl in nm_levels.iter().take(nm_skip).skip(1) {
        mid_raw.extend_from_slice(lvl);
    }
    for lvl in orm_levels.iter().take(orm_skip) {
        mid_raw.extend_from_slice(lvl);
    }

    let mut hi_raw = Vec::new();
    if bc_skip > 0 {
        hi_raw.extend_from_slice(&bc_levels[0]);
    }
    if nm_skip > 0 {
        hi_raw.extend_from_slice(&nm_levels[0]);
    }

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
    spec: &PbrPackSpec,
    src_dir: &Path,
) -> Result<(Vec<MatRecord>, Vec<LayerJob>), Refusal> {
    let mut loaded_pkgs: BTreeMap<String, (PathBuf, serde_json::Value)> = BTreeMap::new();
    for (key, rel_path) in &spec.packages {
        let full = src_dir.join(rel_path);
        loaded_pkgs.insert(key.clone(), load_pkg_json(&full)?);
    }

    let mut mats = Vec::with_capacity(spec.materials.len());
    let mut jobs = Vec::new();
    for (uid, entry) in spec.materials.iter().enumerate() {
        let Some((pdir, gltf)) = loaded_pkgs.get(&entry.package) else {
            return Err(Refusal::new(
                "UnknownPackage",
                &entry.name,
                format!("package `{}` not listed in spec.packages", entry.package),
            ));
        };
        let mat_val = gltf
            .get("materials")
            .and_then(|v| v.get(entry.source_index))
            .ok_or_else(|| {
                Refusal::new(
                    "MissingMaterial",
                    &entry.name,
                    format!("index {}", entry.source_index),
                )
            })?;
        let pbr = mat_val.get("pbrMetallicRoughness");
        let bc_path = gltf_tex_path(gltf, pdir, pbr.and_then(|v| v.get("baseColorTexture")));
        let nm_info = mat_val.get("normalTexture");
        let nm_path = gltf_tex_path(gltf, pdir, nm_info);
        let occ_path = gltf_tex_path(gltf, pdir, mat_val.get("occlusionTexture"));
        let mr_path = gltf_tex_path(
            gltf,
            pdir,
            pbr.and_then(|v| v.get("metallicRoughnessTexture")),
        );
        let normal_scale = entry.normal_scale.unwrap_or_else(|| {
            nm_info
                .and_then(|v| v.get("scale"))
                .and_then(serde_json::Value::as_f64)
                .map_or(1.0, |f| f as f32)
        });
        let rough = entry.roughness.unwrap_or_else(|| {
            pbr.and_then(|v| v.get("roughnessFactor"))
                .and_then(serde_json::Value::as_f64)
                .map_or(1.0, |f| f as f32)
        });
        let metal = entry.metallic.unwrap_or_else(|| {
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
                occ_path,
                normal_scale,
                normal_smooth_passes: entry.normal_smooth_passes,
            });
            (Some(lidx), [0.6, 0.55, 0.5])
        } else {
            let col = entry.base_color.unwrap_or_else(|| {
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
            flags: entry.flags,
            base_col,
            rough,
            metal,
            emissive: entry.emissive,
        });
    }
    Ok((mats, jobs))
}

fn write_spbr_v2_files(
    mats: &[MatRecord],
    layers: &[EncodedLayer],
    dims: PackDims,
    out_stem: &str,
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
        dims.albedo,
        dims.normal,
        dims.orm,
        dims.base,
    ] {
        base_out.extend_from_slice(&w.to_le_bytes());
    }
    let base_mip = dims.base_mip();
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
        for u in [layer_u, m.flags, base_mip, 0_u32] {
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
    let base_path = out_dir.join(format!("{out_stem}_base.bin"));
    let hd_path = out_dir.join(format!("{out_stem}_hd.bin"));
    std::fs::write(&base_path, &base_out)
        .map_err(|e| Refusal::new("WriteFailed", &base_path.display().to_string(), format!("{e}")))?;
    std::fs::write(&hd_path, &hd_out)
        .map_err(|e| Refusal::new("WriteFailed", &hd_path.display().to_string(), format!("{e}")))?;
    Ok((base_path, hd_path, base_out.len(), hd_out.len()))
}

pub fn pack_pbr(spec_path: &Path, src_dir: &Path, out_dir: &Path) -> Result<PackPbrOutput, Refusal> {
    let start = Instant::now();
    let spec_str = spec_path.display().to_string();
    let text = std::fs::read_to_string(spec_path)
        .map_err(|e| Refusal::new("ReadFailed", &spec_str, format!("{e}")))?;
    let spec: PbrPackSpec = serde_json::from_str(&text)
        .map_err(|e| Refusal::new("InvalidSpec", &spec_str, format!("{e}")))?;
    let dims = PackDims {
        albedo: spec.albedo_dim,
        normal: spec.normal_dim,
        orm: spec.orm_dim,
        base: spec.base_dim,
    };
    dims.validate(&spec_str)?;
    let (mut mats, jobs) = collect_materials_and_jobs(&spec, src_dir)?;
    let mut layers = Vec::with_capacity(jobs.len());
    for job in &jobs {
        let encoded = encode_layer_job(job, dims)?;
        mats[encoded.uid].base_col = encoded.mean_lin;
        layers.push(encoded);
    }
    let (base_path, hd_path, base_bytes, hd_bytes) =
        write_spbr_v2_files(&mats, &layers, dims, &spec.out_stem, out_dir)?;
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
