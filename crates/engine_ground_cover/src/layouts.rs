use engine_core::{GpuBits, GpuStruct, gpu_bits};

engine_core::gpu_struct! {
    pub(crate) struct Grow {
        pub(crate) origin: [f32; 4],
        pub(crate) eye: [f32; 4],
        pub(crate) grid: [i32; 4],
        pub(crate) counts: [u32; 4],
        pub(crate) room: [u32; 4],
        pub(crate) detail: [f32; 4],
        pub(crate) tier: [f32; 4],
        pub(crate) terms: [f32; 4],
        pub(crate) least: [f32; 4],
        pub(crate) wind: [f32; 4],
    }
}

engine_core::gpu_struct! {
    pub(crate) struct CoverFrame {
        pub(crate) wind: [f32; 4],
        pub(crate) blades: [u32; 4],
    }
}

engine_core::gpu_struct! {
    pub(crate) struct CoverBlade {
        pub(crate) root: [f32; 3],
        pub(crate) size: u32,
        pub(crate) facing: u32,
        pub(crate) lean: u32,
        pub(crate) root_color: u32,
        pub(crate) tip_color: u32,
    }
}

engine_core::gpu_struct! {
    pub(crate) struct CoverCard {
        pub(crate) root: [f32; 3],
        pub(crate) size: u32,
        pub(crate) facing: u32,
        pub(crate) lean: u32,
        pub(crate) cell: u32,
        pub(crate) tint: u32,
    }
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CoverFacing {
    #[bits(16)]
    pub(crate) angle: u32,
    #[bits(8)]
    pub(crate) tilt_x: u32,
    #[bits(8)]
    pub(crate) tilt_y: u32,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CoverLean {
    #[bits(8)]
    pub(crate) x: u32,
    #[bits(8)]
    pub(crate) y: u32,
    #[bits(8)]
    pub(crate) wind: u32,
    #[bits(8)]
    pub(crate) wind_before: u32,
}

engine_core::gpu_struct! {
    pub(crate) struct CoverMaterial {
        pub(crate) shape: [f32; 4],
        pub(crate) tint: [f32; 4],
        pub(crate) clump: [f32; 4],
        pub(crate) shade: [f32; 4],
        pub(crate) patches: [f32; 4],
        pub(crate) extra: [f32; 4],
        pub(crate) accents: [u32; 4],
        pub(crate) tuft: [f32; 4],
        pub(crate) fans: [f32; 4],
        pub(crate) form: [f32; 4],
        pub(crate) drift: [f32; 4],
        pub(crate) stand: [f32; 4],
        pub(crate) cells: [u32; 4],
        pub(crate) lean: [f32; 4],
        pub(crate) limits: [f32; 4],
    }
}

engine_core::gpu_struct! {
    pub(crate) struct CoverHeader {
        pub(crate) lush: [f32; 4],
        pub(crate) shading: [f32; 4],
        pub(crate) layers: [f32; 4],
    }
}

engine_core::gpu_struct! {
    pub(crate) struct ShapeRows {
        pub(crate) bends: [f32; 4],
        pub(crate) arc: [f32; 4],
        pub(crate) tips: [f32; 4],
        pub(crate) steps: [f32; 4],
    }
}

engine_core::gpu_struct! {
    pub(crate) struct LightRows {
        pub(crate) glow: [f32; 4],
    }
}

engine_core::gpu_struct! {
    pub(crate) struct SeenRows {
        pub(crate) along: [f32; 4],
        pub(crate) tip: [f32; 4],
    }
}

engine_core::gpu_struct! {
    pub(crate) struct CardRows {
        pub(crate) form: [f32; 4],
    }
}

engine_core::gpu_struct! {
    pub(crate) struct CoverTerrain {
        pub(crate) base: u32,
        pub(crate) densest: f32,
        pub(crate) widest: f32,
        pub(crate) tallest: f32,
    }
}

engine_core::gpu_struct! {
    pub(crate) struct CoverCell {
        pub(crate) axis: [f32; 4],
        pub(crate) span: [f32; 4],
    }
}

engine_core::gpu_consts! {
    pub(crate) const COVER_MATERIAL_ROWS: u32 = 15;
    pub(crate) const COVER_SHAPE_ROW: u32 = 3;
    pub(crate) const COVER_LIGHT_ROW: u32 = 7;
    pub(crate) const COVER_SEEN_ROW: u32 = 8;
    pub(crate) const COVER_CARD_ROW: u32 = 10;
    pub(crate) const COVER_HEADER_ROWS: u32 = 11;
    pub(crate) const COVER_CELL_ROWS: u32 = 2;
    pub(crate) const SEEN_ABOVE_LOW: f32 = 0.465;
    pub(crate) const SEEN_ABOVE_MIDDLE: f32 = 0.107;
    pub(crate) const SEEN_ABOVE_HIGH: f32 = 0.0274;
    pub(crate) const SKY_COSINE: f32 = 2.0 / 3.0;
}

const ROW: usize = size_of::<[u32; 4]>();

const _: () = assert!(
    size_of::<CoverMaterial>() == COVER_MATERIAL_ROWS as usize * ROW,
    "material_fills_its_rows"
);

const _: () = assert!(
    size_of::<CoverHeader>() == COVER_SHAPE_ROW as usize * ROW,
    "shape_follows_the_header"
);

const _: () = assert!(
    size_of::<ShapeRows>() == (COVER_LIGHT_ROW - COVER_SHAPE_ROW) as usize * ROW,
    "light_follows_the_shape"
);

const _: () = assert!(
    size_of::<LightRows>() == (COVER_SEEN_ROW - COVER_LIGHT_ROW) as usize * ROW,
    "seen_follows_the_light"
);

const _: () = assert!(
    size_of::<SeenRows>() == (COVER_CARD_ROW - COVER_SEEN_ROW) as usize * ROW,
    "cards_follow_the_seen"
);

const _: () = assert!(
    size_of::<CardRows>() == (COVER_HEADER_ROWS - COVER_CARD_ROW) as usize * ROW,
    "cards_end_the_header"
);

const _: () = assert!(size_of::<CoverTerrain>() == ROW, "terrain_fills_a_row");

const _: () = assert!(
    size_of::<CoverCell>() == COVER_CELL_ROWS as usize * ROW,
    "cell_fills_its_rows"
);

pub(crate) const GENERATED: &[&str] = &[
    Grow::WGSL,
    CoverFrame::WGSL,
    CoverBlade::WGSL,
    CoverCard::WGSL,
    <CoverFacing as GpuBits>::WGSL,
    <CoverLean as GpuBits>::WGSL,
    CoverMaterial::WGSL,
    CoverHeader::WGSL,
    ShapeRows::WGSL,
    LightRows::WGSL,
    SeenRows::WGSL,
    CardRows::WGSL,
    CoverTerrain::WGSL,
    CoverCell::WGSL,
    WGSL,
];
