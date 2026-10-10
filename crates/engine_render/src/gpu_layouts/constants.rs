//! Values Rust and WGSL share, declared once here with the WGSL constants
//! mirroring them; constants of other crates shaders mirror are registered
//! by path. Rust code reads them from here.

use super::scene::{MAX_CASCADES, MAX_LEVELS};
use super::terrain::PaletteEntry;

gpu_consts! {
    // crate::gpu_data
    /// Candidate flags of `presentation.wgsl` the compaction passes gather.
    pub(crate) const FLAG_VISIBLE: u32 = 1 => ["VISIBLE" in PRESENTATION_SHADER];
    pub(crate) const FLAG_PHASE1: u32 = 8 => ["PHASE1" in PRESENTATION_SHADER];
    pub(crate) const FLAG_PHASE2: u32 = 16 => ["PHASE2" in PRESENTATION_SHADER];
    /// Reflection probe captures: candidates of the cube face
    /// being captured.
    pub(crate) const FLAG_PROBE: u32 = 128 => ["PROBE" in PRESENTATION_SHADER];
    /// Frames in a row a slot must display an unchanged pose before reflection
    /// probes capture it.
    pub(crate) const STILL_FRAMES: u32 = 16 => ["STILL_FRAMES"];
    /// Bit of a visible slot ID marking a slot whose displayed pose can differ
    /// from the previous frame's; slot indices stay below it.
    pub(crate) const MOVED_SLOT: u32 = 1 << 31 => ["MOVED_SLOT"];
    /// Levels of detail a draw group has at most: the mesh and two
    /// simplifications, each with its own plane of visible IDs and draws.
    pub(crate) const LOD_LEVELS: usize = 3 => ["LOD_LEVELS" in PRESENTATION_SHADER, "IMPOSTOR" in PRESENTATION_SHADER];
    /// A slot's level of detail state's flag of being set: the slot drew
    /// with its level tracked, which the state's low two bits hold.
    pub(crate) const LOD_STATE: u32 = 1 << 31 => ["LOD_STATE" in PRESENTATION_SHADER];
    /// Indexed indirect draw arguments written by the GPU.
    pub(crate) const DRAW_ARGS_BYTES: u64 = 20 => ["size_of DrawArgs" in PRESENTATION_SHADER, "size_of DrawArgs" in SHADOW_SHADER];
    /// Composed active-parent record of `presentation.wgsl`.
    pub(crate) const COMPOSED_BYTES: u64 = 112 => ["size_of Composed" in PRESENTATION_SHADER];

    // crate::gpu_presentation
    // Candidates per workgroup of the per-candidate passes, and of the
    // compaction passes (`GROUP` in presentation.wgsl), whose workgroup arrays
    // are kept small for Firefox's compile times.
    pub(crate) const PRESENTATION_GROUP: u32 = 256 => ["@shift_previous" in PRESENTATION_SHADER, "@cull" in PRESENTATION_SHADER, "@count_still" in PRESENTATION_SHADER, "@fade_levels" in PRESENTATION_SHADER, "@occlusion_test" in PRESENTATION_SHADER, "@probe_cull" in PRESENTATION_SHADER];
    pub(crate) const SCAN_GROUP: u32 = 64 => ["GROUP" in PRESENTATION_SHADER, "@sum_groups" in PRESENTATION_SHADER, "@scan_groups" in PRESENTATION_SHADER, "@scatter" in PRESENTATION_SHADER];
    pub(crate) const PARENT_GROUP: u32 = 64 => ["@compose_parents" in PRESENTATION_SHADER];
    pub(crate) const JOINT_GROUP: u32 = 64 => ["@skin_palette" in PRESENTATION_SHADER];
    pub(crate) const SKIN_BOUNDS_GROUP: u32 = 64 => ["@skin_bounds_pass" in PRESENTATION_SHADER];
    pub(crate) const ARGS_GROUP: u32 = 64 => ["@batch_args" in PRESENTATION_SHADER];

    // crate::gpu_presentation::fades
    /// Vertices of an impostor card: two triangles.
    pub(crate) const CARD_VERTICES: u32 = 6 => ["IMPOSTOR_VERTICES" in PRESENTATION_SHADER];

    // crate::scatter
    /// Levels each draw group lists: its mesh and four coarser levels.
    pub(crate) const SCATTER_LEVELS: u32 = 5 => ["SHOWN_CARD" in SCATTER_CULL_SHADER];
    /// How the camera shows an object (`scatter_cull.wgsl`): its
    /// levels, its card or nothing, and how far its last switch has faded.
    /// Objects new to the view start showing nothing, so they fade in; those
    /// written again in place take what the camera would show at once.
    pub(crate) const SHOWN_NONE: u32 = 6 => ["SHOWN_NONE" in SCATTER_CULL_SHADER];
    pub(crate) const FADED: u32 = 65535 => ["FADED" in SCATTER_CULL_SHADER];
    pub(crate) const STATE_ADOPT: u32 = u32::MAX => ["STATE_ADOPT" in SCATTER_CULL_SHADER];
    /// The draw group of slots no object holds, which culling skips.
    pub(crate) const DEAD: u32 = u32::MAX => ["DEAD"];
    /// A record's terrain word: the index this frame of the terrain the object
    /// stands on, nine bits (`TERRAIN_INDEX` itself: none); its placement
    /// layer, seven bits from `LAYER_SHIFT` (`NO_LAYER`: authored, or none);
    /// for objects standing far away, one more than the level
    /// whose resident tiles hold them instead, five bits from
    /// `NEAR_LEVEL_SHIFT`; its rank under the density where it stands,
    /// ten bits from `RANK_SHIFT`; and the top bit (the
    /// shaders' `HIDDEN`) while tiles holding it instead are resident where it
    /// stands, which setting it on the terrain marks and culling skips.
    pub(crate) const TERRAIN_INDEX: u32 = 0x1ff => ["TERRAIN_INDEX"];
    pub(crate) const LAYER_SHIFT: u32 = 9 => ["LAYER_SHIFT"];
    pub(crate) const NO_LAYER: u32 = 0x7f => ["NO_LAYER"];
    pub(crate) const NEAR_LEVEL_SHIFT: u32 = 16 => ["NEAR_LEVEL_SHIFT"];
    pub(crate) const RANK_SHIFT: u32 = 21 => ["RANK_SHIFT" in SCATTER_CULL_SHADER];
    /// Entries of the thinning table each terrain takes: one
    /// per placement layer a surface may have (a validity bound).
    pub(crate) const THIN_LAYERS: usize = 64 => ["THIN_LAYERS"];

    // crate::shadows
    // Near cascades are maps 0 to MAX_CASCADES - 1; the far maps follow, then
    // the kept levels, one map of selected casters for the one
    // redrawn.
    pub(crate) const MOVING_MAP: usize = MAX_CASCADES => ["MOVING_MAP" in SHADOW_SHADER];
    pub(crate) const KEPT_MAP: usize = MAX_CASCADES + 1 => ["KEPT_MAP" in SHADOW_SHADER];
    pub(crate) const KEPT_LEVELS: usize = MAX_CASCADES + 2 => ["KEPT_LEVELS" in SHADOW_SHADER];
    pub(crate) const SHADOW_MAPS: usize = MAX_CASCADES + 3 => ["MAPS" in SHADOW_SHADER];
    pub(crate) const SHADOW_GROUP: u32 = 256 => ["@select_casters" in SHADOW_SHADER];
    pub(crate) const SHADOW_SKINNED_GROUP: u32 = 64 => ["@select_skinned" in SHADOW_SHADER];
    // Mode bits of the selection pass.
    pub(crate) const STILLNESS: u32 = 1 => ["STILLNESS" in SHADOW_SHADER];
    pub(crate) const REFRESH: u32 = 2 => ["REFRESH" in SHADOW_SHADER];
    pub(crate) const LEVEL_REFRESH: u32 = 4 => ["LEVEL_REFRESH" in SHADOW_SHADER];

    // crate::shadow_fit
    /// Tiles across each side of the kept far map: kept casters
    /// that move mark the tiles under them dirty, and frames redraw them.
    pub(crate) const SHADOW_TILES: u32 = 16 => ["TILES" in SHADOW_SHADER];
    /// Words of the dirty-tile mask, each holding two rows of tiles, as the
    /// selection pass marks them.
    pub(crate) const SHADOW_TILE_WORDS: usize = (SHADOW_TILES * SHADOW_TILES / 32) as usize => ["TILE_WORDS" in SHADOW_SHADER];

    // crate::shadows::selection
    // Dirty tiles of the kept map, then of each kept level, then
    // the four caster totals, read back together.
    pub(crate) const DIRTY_WORDS: usize = SHADOW_TILE_WORDS * (1 + MAX_LEVELS) => ["DIRTY_WORDS" in SHADOW_SHADER];

    // crate::lights
    pub(crate) const NO_SLOT: u32 = u32::MAX => ["NO_SLOT" in LIGHT_SHADER];
    /// Lights each binning workgroup places (`LIGHTS` of lights.wgsl).
    pub(crate) const GROUP_LIGHTS: u32 = 8 => ["LIGHTS" in LIGHT_SHADER];
    /// Bytes of a binned light (WGSL `Light`).
    pub(crate) const LIGHT_BYTES: u64 = 48 => ["size_of Light"];

    // crate::impostors
    /// Frames per side of the octahedral grid, as `meshes/impostors.wgsl`
    /// declares it.
    pub(crate) const IMPOSTOR_FRAMES: u32 = 8 => ["IMPOSTOR_FRAMES"];
    /// The flag of records whose frames cover only the upper hemisphere, above
    /// their texels across a frame.
    pub(crate) const IMPOSTOR_HEMISPHERE: u32 = 1 << 16 => ["IMPOSTOR_HEMISPHERE"];

    // crate::decals
    /// `DecalRecord::pose` of decals drawn from their own matrix.
    pub(crate) const NO_DECAL_POSE: u32 = u32::MAX => ["NO_DECAL_POSE"];

    // crate::terrain
    /// A directory entry of a tile not resident.
    pub(crate) const NOT_RESIDENT: u32 = u32::MAX => ["NOT_RESIDENT"];
    /// Vertices along a patch side, and the grid's before the skirts'.
    pub(crate) const TERRAIN_GRID: u32 = engine_core::terrain_data::PATCH_QUADS + 1 => ["TERRAIN_GRID"];
    pub(crate) const TERRAIN_GRID_VERTICES: u32 = TERRAIN_GRID * TERRAIN_GRID => ["TERRAIN_GRID_VERTICES"];

    // crate::terrain_horizons
    /// A source tile no resident level holds.
    pub(crate) const MISSING: u32 = u32::MAX => ["MISSING"];
    /// Lines of a grid a workgroup sweeps.
    pub(crate) const HORIZON_GROUP: u32 = 64 => ["@build_horizon" in TERRAIN_HORIZON_SHADER];

    // crate::terrain_sky
    /// Workgroup side of a build.
    pub(crate) const SKY_MAP_GROUP: u32 = 8 => ["@build_sky" in TERRAIN_SKY_SHADER];

    // crate::terrain_vt
    /// Texels along a page's side, the border around it, and a slot's side.
    pub(crate) const VT_PAGE_TEXELS: u32 = 128 => ["VT_PAGE"];
    pub(crate) const VT_BORDER_TEXELS: u32 = 8 => ["VT_BORDER"];
    pub(crate) const VT_SLOT_TEXELS: u32 = VT_PAGE_TEXELS + 2 * VT_BORDER_TEXELS => ["VT_SLOT"];
    /// Pages along a side of each mip's window around the eye: as many as
    /// cover the distances where pixels read the mip, whatever it is.
    pub(crate) const VT_WINDOW: i32 = 64 => ["VT_WINDOW"];
    /// Mips, and the meters mip 0's texels span: about the material maps'
    /// own texels where they repeat every few meters.
    pub(crate) const VT_MIPS: usize = 18 => ["VT_MIPS"];

    // crate::deformation
    /// Texels along a side of a block of the map's mask.
    pub(crate) const DEFORMATION_BLOCK: u32 = 16 => ["BLOCK" in DEFORMATION_SHADER];

    // crate::virtual_geometry
    pub(crate) const NONE: u32 = u32::MAX => ["NONE" in VIRTUAL_SHADER];
    pub(crate) const VIRTUAL_GROUP: u32 = 64 => ["GROUP" in VIRTUAL_SHADER, "@seed" in VIRTUAL_SHADER, "@traverse_even" in VIRTUAL_SHADER, "@traverse_odd" in VIRTUAL_SHADER, "@select_clusters" in VIRTUAL_SHADER, "@retest_clusters" in VIRTUAL_SHADER, "@bin_args" in VIRTUAL_SHADER];
    /// Views one frame selects for: the camera's and those of the shadow maps
    /// drawn every frame (up to four cascades and the moving map).
    pub(crate) const MAX_VIEWS: usize = 8 => ["MAX_VIEWS" in VIRTUAL_SHADER];
    /// Planes of a virtual shadow view: its selection region's five, and the
    /// view frustum slice's that the sun's sweep never crosses.
    pub(crate) const VIRTUAL_SHADOW_PLANES: usize = 12 => ["VIEW_PLANES" in VIRTUAL_SHADER];
    /// Camera tests (`virtual_geometry.wgsl`).
    pub(crate) const TEST_PREVIOUS: u32 = 1 => ["TEST_PREVIOUS" in VIRTUAL_SHADER];
    pub(crate) const TEST_SUBPIXEL: u32 = 2 => ["TEST_SUBPIXEL" in VIRTUAL_SHADER];

    // crate::visibility
    /// Pixels across a tile's side.
    pub(crate) const VISIBILITY_TILE: u32 = 8 => ["@classify_tiles" in VISIBILITY_SHADER];

    // crate::depth_pyramid
    pub(crate) const PYRAMID_GROUP: u32 = 8 => ["@pyramid_first" in DEPTH_PYRAMID_SHADER, "@pyramid_down" in DEPTH_PYRAMID_SHADER, "@pyramid_first_nearest" in DEPTH_PYRAMID_SHADER, "@pyramid_down_nearest" in DEPTH_PYRAMID_SHADER];

    // crate::water
    /// The deepest water the depth texture holds (meters, `water.wgsl`'s
    /// `WATER_DEPTH_RANGE`): it keeps the square root of each depth's share of
    /// it, so shallows keep centimeters and a deep lake's middle its depth.
    pub(crate) const WATER_DEPTH_RANGE: f32 = 256.0 => ["WATER_DEPTH_RANGE"];

    // crate::ground_cover
    /// Bytes of a blade (`CoverBlade`).
    pub(crate) const BLADE_BYTES: u64 = 32 => ["size_of CoverBlade"];

    // crate::bindings
    /// A pyramid level's `Extent` of `depth_pyramid.wgsl`.
    pub(crate) const PYRAMID_EXTENT_BYTES: usize = 16 => ["size_of Extent" in DEPTH_PYRAMID_SHADER];

    // crate::gpu_layouts::terrain
    /// Surface materials every terrain together can show: as many as fit the
    /// 16 KiB uniform binding every device allows.
    pub(crate) const PALETTE_ENTRIES: usize = 16_384 / size_of::<PaletteEntry>() => ["TERRAIN_PALETTE_ENTRIES" in SHADER, "PALETTE_ENTRIES" in COVER_GROW_SHADER];

    /// Cells along a side of the cloud shadows' grid.
    pub(crate) const SHADOW_CELLS: u32 = 128 => ["CLOUD_SHADOW_CELLS"];

    // crate::gpu_data::Variant: the bits of a material's variant passes
    // drawn rarely read from its uniform.
    pub(crate) const VARIANT_MAPS: u16 = 1 => ["VARIANT_MAPS" in SHADER];
    pub(crate) const VARIANT_NORMAL_MAP: u16 = 2 => ["VARIANT_NORMAL_MAP" in SHADER];
    pub(crate) const VARIANT_DOUBLE_SIDED: u16 = 8 => ["VARIANT_DOUBLE_SIDED" in SHADER];
    pub(crate) const VARIANT_METALLIC_ROUGHNESS_MAP: u16 = 128 => ["VARIANT_METALLIC_ROUGHNESS_MAP" in SHADER];
    pub(crate) const VARIANT_OCCLUSION_MAP: u16 = 256 => ["VARIANT_OCCLUSION_MAP" in SHADER];
    pub(crate) const VARIANT_EMISSIVE_MAP: u16 = 512 => ["VARIANT_EMISSIVE_MAP" in SHADER];
    pub(crate) const VARIANT_PACKED_OCCLUSION: u16 = 1024 => ["VARIANT_PACKED_OCCLUSION" in SHADER];
    pub(crate) const VARIANT_WORLD_MAPPING: u16 = 4096 => ["VARIANT_WORLD_MAPPING" in SHADER];
    pub(crate) const VARIANT_TOP_LAYER: u16 = 8192 => ["VARIANT_TOP_LAYER" in SHADER];

    // crate::probes
    /// Words of an entry of the reflection probes' grid lists.
    pub(crate) const PROBE_ENTRY_WORDS: usize = 3 => ["PROBE_ENTRY_WORDS" in SHADER];

    // Workgroup sides that dispatches divide by.
    pub(crate) const SCATTER_GROUP: u32 = 64 => ["@snap_scatter" in SCATTER_SNAP_SHADER, "@cull_scatter" in SCATTER_CULL_SHADER];
    pub(crate) const DEFORMATION_GROUP: u32 = 8 => ["@press" in DEFORMATION_SHADER, "@slopes" in DEFORMATION_SHADER, "@scroll" in DEFORMATION_SHADER, "@wipe" in DEFORMATION_SHADER, "@rebase" in DEFORMATION_SHADER];
    pub(crate) const CLOUD_GROUP: u32 = 8 => ["@march_view" in CLOUDS_SHADER, "@march_clouds" in CLOUDS_SHADER, "@shadow_clouds" in CLOUDS_SHADER];
    pub(crate) const NOISE_GROUP: u32 = 4 => ["@make_noise" in CLOUDS_SHADER];
    pub(crate) const SKY_TABLE_GROUP: u32 = 8 => ["@build_transmittance" in SKY_SHADER, "@build_multiple" in SKY_SHADER, "@build_sky_view" in SKY_SHADER];
    pub(crate) const CONTROL_GROUP: u32 = 8 => ["@tile_control" in TERRAIN_CONTROL_SHADER];
    pub(crate) const NORMALS_GROUP: u32 = 16 => ["@tile_normals" in TERRAIN_NORMALS_SHADER];
    pub(crate) const RESOLVE_ARGS_GROUP: u32 = 64 => ["@resolve_args" in VISIBILITY_SHADER];
    pub(crate) const VT_FEEDBACK_GROUP: u32 = 8 => ["@cs_terrain_vt_feedback" in SHADER];
    /// Pixels a side of the blocks the virtual texture's feedback tests one
    /// pixel of, a different one each frame.
    pub(crate) const VT_FEEDBACK_BLOCK: u32 = 8 => ["VT_FEEDBACK_BLOCK" in SHADER];

    // Values only shaders hold, in several modules under several names.
    /// A card word's flag of fading, and of fading out.
    pub(crate) const CARD_FADING: u32 = 1 << 31 => ["FADING" in SCATTER_CULL_SHADER, "IMPOSTOR_FADING", "CARD_FADING" in PRESENTATION_SHADER];
    pub(crate) const CARD_FADING_OUT: u32 = 1 << 30 => ["FADING_OUT" in SCATTER_CULL_SHADER, "IMPOSTOR_FADING_OUT", "CARD_FADING_OUT" in PRESENTATION_SHADER];
    /// A card word's flag of the card drawn over its mesh.
    pub(crate) const CARD_OVER: u32 = 1 << 15 => ["IMPOSTOR_OVER", "CARD_OVER" in PRESENTATION_SHADER];
    /// Where the cloud layer fades into the horizon (the view ray's rise).
    pub(crate) const CLOUD_HORIZON: f32 = 0.05 => ["HORIZON" in CLOUDS_SHADER, "CLOUD_HORIZON"];
    /// How far the cloud march reaches, in meters.
    pub(crate) const CLOUD_REACH: f32 = 60000.0 => ["REACH" in CLOUDS_SHADER, "CLOUD_PARALLAX_REACH"];
    /// Vertices drawn per cluster of virtual geometry: three a triangle.
    pub(crate) const CLUSTER_DRAWN_VERTICES: u32 = 3 * engine_core::CLUSTER_TRIANGLES => ["CLUSTER_VERTICES" in VIRTUAL_SHADER];
}

gpu_mirrors! {
    engine_core::PAGE_WORDS => ["PAGE_WORDS" in VIRTUAL_SHADER, "VIRTUAL_PAGE_WORDS" in SHADER];
    engine_core::NO_PARENT => ["NO_PARENT" in PRESENTATION_SHADER];
    engine_core::NODE_VALID => ["NODE_VALID" in PRESENTATION_SHADER];
    engine_core::NODE_BLEND => ["NODE_BLEND" in PRESENTATION_SHADER];
    engine_core::NODE_ACTIVE => ["NODE_ACTIVE" in PRESENTATION_SHADER];
    engine_core::NODE_PARENT_ACTIVE => ["NODE_PARENT_ACTIVE" in PRESENTATION_SHADER];
    engine_core::presentation_bounds::MAX_COEFFICIENT => ["MAX_COEFFICIENT" in PRESENTATION_SHADER];
    engine_core::presentation_bounds::ROUNDING_MARGIN => ["ROUNDING_MARGIN" in PRESENTATION_SHADER];
    engine_core::presentation_bounds::MAX_PRODUCT_BOUND => ["MAX_PRODUCT_BOUND" in PRESENTATION_SHADER];
    engine_core::presentation_bounds::COEFFICIENT_LIMIT => ["COEFFICIENT_LIMIT" in PRESENTATION_SHADER];
    engine_core::presentation_bounds::MIN_LINEAR_SCALE => ["MIN_LINEAR_SCALE" in PRESENTATION_SHADER];
    engine_core::presentation_bounds::COMMON_DETERMINANT_MARGIN => ["COMMON_DETERMINANT_MARGIN" in PRESENTATION_SHADER];
    engine_core::presentation_bounds::LINEAR_ROTATION_DOT_THRESHOLD => ["LINEAR_ROTATION_DOT" in PRESENTATION_SHADER];
    engine_core::RIM_START => ["RIM_START" in DEFORMATION_SHADER];
    engine_core::RIM_REACH => ["RIM_REACH" in DEFORMATION_SHADER];
    engine_core::terrain_data::TILE_QUADS => ["TILE_QUADS" in TERRAIN_HORIZON_SHADER, "TILE_QUADS" in TERRAIN_SKY_SHADER];
    engine_core::terrain_data::TILE_STRIDE => ["TILE_STRIDE" in TERRAIN_CONTROL_SHADER, "TILE_STRIDE" in TERRAIN_NORMALS_SHADER];
    engine_core::terrain_data::RIVER_FAR => ["RIVER_FAR" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::PATH_FAR => ["PATH_FAR" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::MAX_STACK => ["MAX_STACK" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::MAX_MATERIALS => ["MAX_MATERIALS" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::PROGRAM => ["H_PROGRAM" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::BIOMES => ["H_BIOMES" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::READS => ["H_READS" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::HYDROLOGY => ["H_HYDROLOGY" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::RIVERS => ["H_RIVERS" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::PATHS => ["H_PATHS" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::FADE => ["H_FADE" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::CONSTANT => ["OP_CONSTANT" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::HEIGHT => ["OP_HEIGHT" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::GRADIENT => ["OP_GRADIENT" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::CURVATURE => ["OP_CURVATURE" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::FLOW => ["OP_FLOW" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::WETNESS => ["OP_WETNESS" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::LAKE => ["OP_LAKE" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::RIVER => ["OP_RIVER" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::PATH => ["OP_PATH" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::NOISE => ["OP_NOISE" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::MULTIPLY => ["OP_MULTIPLY" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::ADD => ["OP_ADD" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::MIN => ["OP_MIN" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::MAX => ["OP_MAX" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::INVERT => ["OP_INVERT" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::REMAP => ["OP_REMAP" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::END => ["OP_END" in TERRAIN_CONTROL_SHADER];
}
