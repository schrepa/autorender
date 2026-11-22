use std::sync::{Arc, Mutex};

use engine::prelude::*;

pub const GI_MAGIC: [u8; 4] = *b"GIP1";
pub const GI_VERSION: u32 = 1;

pub struct GiVolumeUpload {
    pub dims: [u32; 3],
    pub grid_min: [f32; 3],
    pub grid_max: [f32; 3],
    pub sh_r: Vec<u8>,
    pub sh_g: Vec<u8>,
    pub sh_b: Vec<u8>,
}

#[derive(Default)]
pub struct GiUploadQueue {
    pub pending: Option<GiVolumeUpload>,
    pub grid_min: [f32; 3],
    pub grid_max: [f32; 3],
    pub dims: [u32; 3],
    pub loaded: bool,
}

#[derive(Resource, Clone, Default)]
pub struct GiUploadBridge(pub Arc<Mutex<GiUploadQueue>>);

fn read_u32_le(bytes: &[u8], off: usize) -> Option<u32> {
    let s: [u8; 4] = bytes.get(off..off + 4)?.try_into().ok()?;
    Some(u32::from_le_bytes(s))
}

fn read_f32_le(bytes: &[u8], off: usize) -> Option<f32> {
    Some(f32::from_bits(read_u32_le(bytes, off)?))
}

pub fn parse_gi_bytes(bytes: &[u8]) -> Option<GiVolumeUpload> {
    if bytes.len() < 48 || bytes[0..4] != GI_MAGIC {
        return None;
    }
    let version = read_u32_le(bytes, 4)?;
    if version != GI_VERSION {
        return None;
    }
    let dx = read_u32_le(bytes, 8)?;
    let dy = read_u32_le(bytes, 12)?;
    let dz = read_u32_le(bytes, 16)?;
    if dx == 0 || dy == 0 || dz == 0 || dx > 64 || dy > 64 || dz > 64 {
        return None;
    }
    let grid_min = [
        read_f32_le(bytes, 24)?,
        read_f32_le(bytes, 28)?,
        read_f32_le(bytes, 32)?,
    ];
    let grid_max = [
        read_f32_le(bytes, 36)?,
        read_f32_le(bytes, 40)?,
        read_f32_le(bytes, 44)?,
    ];
    let vol_bytes = (dx as usize)
        .checked_mul(dy as usize)?
        .checked_mul(dz as usize)?
        .checked_mul(8)?;
    let start = 48_usize;
    let total = start.checked_add(vol_bytes.checked_mul(3)?)?;
    if bytes.len() < total {
        return None;
    }
    let sh_r = bytes[start..start + vol_bytes].to_vec();
    let sh_g = bytes[start + vol_bytes..start + vol_bytes * 2].to_vec();
    let sh_b = bytes[start + vol_bytes * 2..total].to_vec();
    Some(GiVolumeUpload {
        dims: [dx, dy, dz],
        grid_min,
        grid_max,
        sh_r,
        sh_g,
        sh_b,
    })
}

pub fn submit_gi_bytes(bridge: &GiUploadBridge, bytes: &[u8]) -> bool {
    let Some(vol) = parse_gi_bytes(bytes) else {
        return false;
    };
    if let Ok(mut q) = bridge.0.lock() {
        q.grid_min = vol.grid_min;
        q.grid_max = vol.grid_max;
        q.dims = vol.dims;
        q.loaded = true;
        q.pending = Some(vol);
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gi_parser_handles_valid_and_malformed_inputs() {
        assert!(parse_gi_bytes(b"NOPE").is_none(), "short invalid input must fail");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GI_MAGIC);
        for w in [GI_VERSION, 2_u32, 2, 2, 0] {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        for f in [-1.0_f32, 0.0, -2.0, 3.0, 4.0, 5.0] {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        let vol_bytes = 2 * 2 * 2 * 8;
        assert!(
            parse_gi_bytes(&bytes).is_none(),
            "truncated payload must be rejected"
        );
        bytes.resize(48 + vol_bytes * 3, 0x3C);
        let Some(vol) = parse_gi_bytes(&bytes) else {
            panic!("valid GIP1 payload must parse");
        };
        assert_eq!(vol.dims, [2, 2, 2], "dims must match header");
        assert_eq!(vol.sh_r.len(), vol_bytes, "red volume length must match");
        assert_eq!(vol.sh_g.len(), vol_bytes, "green volume length must match");
        assert_eq!(vol.sh_b.len(), vol_bytes, "blue volume length must match");
    }
}
