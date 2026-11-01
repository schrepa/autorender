use bevy_ecs::prelude::Resource;
use bytemuck::{Pod, Zeroable};
use engine_core::{Grounding, ImpostorRecord, Surface, gpu_consts, gpu_struct};
use engine_render::debug::{
    DEBUG_CASTS, DEBUG_KIND_PLACED, DEBUG_KIND_PLACED_CARD, DEBUG_LEVEL_SHIFT,
};
use glam::Mat4;

gpu_struct! {
    pub(crate) struct CullFrame {
        pub eye: [f32; 4],
        pub counts: [u32; 4],
        pub strides: [u32; 4],
        pub fade: [f32; 4],
    }
}

gpu_struct! {
    pub(crate) struct CullView {
        pub planes: [[f32; 4]; 6],
        pub limits: [f32; 4],
    }
}

gpu_struct! {
    pub(crate) struct CullGroup {
        pub ids: u32,
        pub count: u32,
        pub levels: u32,
        pub camera: u32,
        pub errors: [f32; 4],
        pub impostor: f32,
        pub record: u32,
        pub casters: u32,
        pub parts: u32,
    }
}

gpu_struct! {
    pub(crate) struct Snap {
        pub counts: [u32; 4],
        pub eye: [f32; 4],
    }
}

gpu_struct! {
    pub(crate) struct Splat {
        pub area: [f32; 4],
        pub counts: [u32; 4],
        pub place: [u32; 4],
        pub dirty: [[u32; 4]; 2],
        pub finer: [i32; 4],
        pub means: [[u32; 4]; 2],
    }
}

gpu_struct! {
    pub(crate) struct Cover {
        pub footprint: f32,
        pub bottom: f32,
        pub top: f32,
        pub density: f32,
    }
}

gpu_struct! {
    pub(crate) struct ListedDraw {
        pub vertices: u32,
        pub instances: u32,
        pub first_vertex: u32,
        pub first_instance: u32,
    }
}

gpu_struct! {
    pub(crate) struct PlacedFar {
        pub placed: [f32; 4],
        pub size: [f32; 4],
    }
}

gpu_struct! {
    pub(crate) struct PlacedGroup {
        pub sphere: [f32; 4],
        pub extent: [f32; 3],
        pub layer: u32,
        pub emissive: [f32; 3],
        pub flags: u32,
        pub color: [u32; 2],
        pub pad: [u32; 2],
    }
}

gpu_struct! {
    #[derive(Resource)]
    pub struct CanopyPiece {
        pub occupancy: [f32; 4],
        pub occupancy_maps: [f32; 4],
        pub occupancy_centers: [[f32; 4]; 4],
        pub foliage: [f32; 4],
    }
}

gpu_consts! {
    pub(crate) const SHOWN_CARD: u32 = 5;
    pub(crate) const SHOWN_NONE: u32 = 6;
    pub(crate) const FADED: u32 = 65535;
    pub(crate) const STATE_ADOPT: u32 = 0xffffffff;
    pub(crate) const DEAD: u32 = 0xffffffff;
    pub(crate) const HIDDEN: u32 = 0x80000000;
    pub(crate) const TERRAIN_INDEX: u32 = 0x1ff;
    pub(crate) const LAYER_SHIFT: u32 = 9;
    pub(crate) const NO_LAYER: u32 = 0x7f;
    pub(crate) const NEAR_LEVEL_SHIFT: u32 = 16;
    pub(crate) const RANK_SHIFT: u32 = 21;
    pub(crate) const CELLS: u32 = 16;
    pub(crate) const QUAD: u32 = 6;
    pub(crate) const SCATTER_GROUP: u32 = 64;
    pub(crate) const OCCUPANCY_GROUP: u32 = 64;
}

pub(crate) const GENERATED: &[&str] = &[
    WGSL,
    <CullFrame as engine_core::GpuStruct>::WGSL,
    <CullView as engine_core::GpuStruct>::WGSL,
    <CullGroup as engine_core::GpuStruct>::WGSL,
    <Snap as engine_core::GpuStruct>::WGSL,
    <Splat as engine_core::GpuStruct>::WGSL,
    <Cover as engine_core::GpuStruct>::WGSL,
    <ListedDraw as engine_core::GpuStruct>::WGSL,
    <PlacedFar as engine_core::GpuStruct>::WGSL,
    <PlacedGroup as engine_core::GpuStruct>::WGSL,
];

const _: () = assert!(HIDDEN > RANK_MASK << RANK_SHIFT, "hidden_above_rank");

pub(crate) const LEVELS: u32 = SHOWN_CARD;
pub(crate) const STATE_FRESH: u32 = SHOWN_NONE | SHOWN_NONE << 3 | FADED << 6;
pub(crate) const NEAR_LEVEL_MASK: u32 = 0x1f;
pub(crate) const RANK_MAX: f32 = 1023.0;
const RANK_MASK: u32 = 0x3ff;
pub(crate) const MAX_OCCUPANCY_LEVELS: usize = crate::settings::MAX_OCCUPANCY_LEVELS as usize;

pub(crate) const RECORD_LEVELS: u32 = 16;
pub(crate) const RECORD_BYTES: u64 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DrawKind {
    Placed,
    PlacedCard,
}

impl DrawKind {
    const ALL: [Self; 2] = [Self::Placed, Self::PlacedCard];

    fn kind(self) -> u32 {
        match self {
            Self::Placed => DEBUG_KIND_PLACED,
            Self::PlacedCard => DEBUG_KIND_PLACED_CARD,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DrawRecord {
    pub(crate) kind: DrawKind,
    pub(crate) level: u32,
    pub(crate) casts: bool,
}

impl DrawRecord {
    pub(crate) fn of(kind: DrawKind, level: usize, casts: bool) -> Self {
        Self {
            kind,
            level: level.min(RECORD_LEVELS as usize - 1) as u32,
            casts,
        }
    }

    pub(crate) fn offset(self) -> u64 {
        let kind = DrawKind::ALL
            .iter()
            .position(|kind| *kind == self.kind)
            .unwrap_or_default() as u64;
        ((kind * u64::from(RECORD_LEVELS) + u64::from(self.level.min(RECORD_LEVELS - 1))) * 2
            + u64::from(self.casts))
            * RECORD_BYTES
    }

    fn word(self) -> u32 {
        let mut word = self.kind.kind() | (self.level.min(RECORD_LEVELS - 1) << DEBUG_LEVEL_SHIFT);
        if self.casts {
            word |= DEBUG_CASTS;
        }
        word
    }
}

pub(crate) fn record_words() -> Vec<[u32; 4]> {
    let mut words = Vec::new();
    for kind in DrawKind::ALL {
        for level in 0..RECORD_LEVELS {
            for casts in [false, true] {
                let record = DrawRecord { kind, level, casts };
                words.push([record.word(), 0, 0, 0]);
            }
        }
    }
    words
}

#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct PlacedRecord {
    pub center: [f32; 3],
    pub radius: f32,
    pub group: u32,
    pub scale: f32,
    pub terrain: u32,
    pub sink: f32,
    pub state: u32,
    pub pad: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct SlotColor {
    pub color: [u32; 2],
    pub surface: u32,
    pub flags: u32,
}

pub(crate) type PoseData = [[f32; 4]; 3];

pub(crate) type GroundData = [u32; 4];

pub(crate) type DrawArgs = [u32; 5];

pub(crate) fn draw_args((index_count, first_index, base_vertex): (u32, u32, i32)) -> DrawArgs {
    [
        index_count,
        0,
        first_index,
        u32::from_ne_bytes(base_vertex.to_ne_bytes()),
        0,
    ]
}

pub(crate) fn placed_group(impostor: ImpostorRecord, color: [u32; 2]) -> PlacedGroup {
    PlacedGroup {
        sphere: impostor.sphere,
        extent: impostor.extent,
        layer: impostor.layer,
        emissive: impostor.emissive,
        flags: impostor.flags,
        color,
        pad: [0; 2],
    }
}

pub(crate) fn pack_surface(surface: Surface) -> u32 {
    let unit = |value: f32, fallback: f32, max: f32| {
        let value = if value.is_nan() { fallback } else { value };
        (engine_math::clamp(value / max, 0.0, 1.0) * 255.0).round() as u32
    };
    let matte = Surface::MATTE;
    unit(surface.roughness, matte.roughness, 1.0)
        | unit(surface.metalness, matte.metalness, 1.0) << 8
        | unit(surface.emission, matte.emission, Surface::MAX_EMISSION) << 16
        | unit(surface.ripples, matte.ripples, 1.0) << 24
}

pub(crate) fn pack_color(color: [f32; 4]) -> [u32; 2] {
    let unit = |value: f32, fallback: f32| {
        let value = if value.is_nan() { fallback } else { value };
        (engine_math::clamp(value, 0.0, 1.0) * 65535.0).round() as u32
    };
    [
        unit(color[0], 0.0) | unit(color[1], 0.0) << 16,
        unit(color[2], 0.0) | unit(color[3], 1.0) << 16,
    ]
}

pub(crate) fn ground_data(
    grounding: Option<Grounding>,
    normal: [f32; 2],
    sink: f32,
    scale: f32,
) -> GroundData {
    let Some(grounding) = grounding else {
        return [0; 4];
    };
    let snorm = |value: f32| {
        u32::from((engine_math::clamp(value, -1.0, 1.0) * 32767.0).round() as i16 as u16)
    };
    [
        snorm(normal[0]) | snorm(normal[1]) << 16,
        sink.to_bits(),
        (grounding.height * scale).to_bits(),
        0,
    ]
}

pub(crate) fn planes(view_projection: Mat4) -> [[f32; 4]; 6] {
    let rows = [
        view_projection.row(0),
        view_projection.row(1),
        view_projection.row(2),
        view_projection.row(3),
    ];
    [
        rows[3] + rows[0],
        rows[3] - rows[0],
        rows[3] + rows[1],
        rows[3] - rows[1],
        rows[2],
        rows[3] - rows[2],
    ]
    .map(|plane: glam::Vec4| {
        (plane / engine_math::max(plane.truncate().length(), 1.0e-12)).to_array()
    })
}

pub(crate) fn pose_rows(model: Mat4) -> PoseData {
    let rows = model.transpose();
    [
        rows.x_axis.to_array(),
        rows.y_axis.to_array(),
        rows.z_axis.to_array(),
    ]
}
