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
    // redrawn, then the sky map redrawn this frame (`shadows::sky`).
    pub(crate) const MOVING_MAP: usize = MAX_CASCADES => ["MOVING_MAP" in SHADOW_SHADER];
    pub(crate) const KEPT_MAP: usize = MAX_CASCADES + 1 => ["KEPT_MAP" in SHADOW_SHADER];
    pub(crate) const KEPT_LEVELS: usize = MAX_CASCADES + 2 => ["KEPT_LEVELS" in SHADOW_SHADER];
    pub(crate) const SKY_MAP: usize = MAX_CASCADES + 3 => ["SKY_MAP" in SHADOW_SHADER];
    pub(crate) const SHADOW_MAPS: usize = MAX_CASCADES + 4 => ["MAPS" in SHADOW_SHADER];
    pub(crate) const SHADOW_GROUP: u32 = 256 => ["@select_casters" in SHADOW_SHADER];
    pub(crate) const SHADOW_SKINNED_GROUP: u32 = 64 => ["@select_skinned" in SHADOW_SHADER];
    // Mode bits of the selection pass.
    pub(crate) const STILLNESS: u32 = 1 => ["STILLNESS" in SHADOW_SHADER];
    pub(crate) const REFRESH: u32 = 2 => ["REFRESH" in SHADOW_SHADER];
    pub(crate) const LEVEL_REFRESH: u32 = 4 => ["LEVEL_REFRESH" in SHADOW_SHADER];
    pub(crate) const SKY_REFRESH: u32 = 8 => ["SKY_REFRESH" in SHADOW_SHADER];
    /// Directions of the sky that sky occlusion tests at most, and its
    /// maps: one per direction near the camera, then one per direction
    /// reaching farther.
    pub(crate) const SKY_DIRECTIONS: usize = 13 => ["SKY_DIRECTIONS" in POST_SHADER];
    pub(crate) const SKY_LAYERS: usize = 2 * SKY_DIRECTIONS => ["SKY_LAYERS" in POST_SHADER];

    // crate::shadow_fit
    /// Tiles across each side of the kept far map: kept casters
    /// that move mark the tiles under them dirty, and frames redraw them.
    pub(crate) const SHADOW_TILES: u32 = 16 => ["TILES" in SHADOW_SHADER];
    /// Words of the dirty-tile mask, each holding two rows of tiles, as the
    /// selection pass marks them.
    pub(crate) const SHADOW_TILE_WORDS: usize = (SHADOW_TILES * SHADOW_TILES / 32) as usize => ["TILE_WORDS" in SHADOW_SHADER];
    /// Texels of a kept level a sample of the ground it holds spans at
    /// least: finer samples draw detail its texels cannot hold. The ground's
    /// own fragments look each level up from that level's ground.
    pub(crate) const GROUND_TEXELS: f32 = 2.0 => ["KEPT_GROUND_TEXELS" in SHADER];
    /// Layers of the far maps' texture: the kept map's, then the moving
    /// map's.
    pub(crate) const FAR_KEPT: u32 = 0 => ["FAR_KEPT"];
    pub(crate) const FAR_MOVING: u32 = 1 => ["FAR_MOVING"];

    // crate::shadows::selection
    // Dirty tiles of the kept map, then of each kept level, then
    // the four caster totals, read back together.
    pub(crate) const DIRTY_WORDS: usize = SHADOW_TILE_WORDS * (1 + MAX_LEVELS) => ["DIRTY_WORDS" in SHADOW_SHADER];

    // crate::lights
    pub(crate) const NO_SLOT: u32 = u32::MAX => ["NO_SLOT" in LIGHT_SHADER];
    /// Lights each binning workgroup places (`LIGHTS` of lights.wgsl).
    pub(crate) const GROUP_LIGHTS: u32 = 8 => ["LIGHTS" in LIGHT_SHADER];
    /// Bytes of a binned light (WGSL `Light`).
    pub(crate) const LIGHT_BYTES: u64 = 64 => ["size_of Light"];
    /// How lights shade (the clusters' `lights.y`): from their spheres and
    /// tubes, and kept to their regions.
    pub(crate) const SHAPED_LIGHTS: u32 = 1 => ["SHAPED_LIGHTS" in LIGHT_SHADER, "SHAPED_LIGHTS" in SHADER];
    pub(crate) const KEPT_REGIONS: u32 = 2 => ["KEPT_REGIONS" in SHADER];
    /// Where a light shines (`engine_core::LightRegion`): inside
    /// interiors only, or outside them only; everywhere otherwise.
    pub(crate) const LIGHT_INTERIOR: u32 = 1 => ["LIGHT_INTERIOR" in SHADER];
    pub(crate) const LIGHT_EXTERIOR: u32 = 2 => ["LIGHT_EXTERIOR" in SHADER];

    // crate::impostors
    /// The bit from which impostor records hold their frames along each
    /// side of their grid, in five bits.
    pub(crate) const IMPOSTOR_FRAMES_SHIFT: u32 = 19 => ["IMPOSTOR_FRAMES_SHIFT"];
    /// The flag of records whose frames cover only the upper hemisphere, above
    /// their texels across a frame.
    pub(crate) const IMPOSTOR_HEMISPHERE: u32 = 1 << 16 => ["IMPOSTOR_HEMISPHERE"];
    /// The flag of impostor records whose material, or a part's, is masked.
    pub(crate) const IMPOSTOR_MASKED: u32 = 1 << 17 => ["IMPOSTOR_MASKED"];
    /// The flag of impostor records whose material is divided into rooms
    /// (`Material::windows`).
    pub(crate) const IMPOSTOR_ROOMS: u32 = 1 << 18 => ["IMPOSTOR_ROOMS"];
    /// The flag of impostor records whose meshes shade as foliage: masked,
    /// or of foliage materials (`Material::is_foliage`); above the frames.
    pub(crate) const IMPOSTOR_FOLIAGE: u32 = 1 << 24 => ["IMPOSTOR_FOLIAGE"];

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
    /// Vectors of the record each terrain's horizon map takes (where it lies
    /// and how its shadows soften, then where its bounce light lies: its
    /// levels' layout, and each level's window, four levels a vector), and
    /// those the records' uniform holds: as many as fit 6 KiB, 64 terrains,
    /// more than the maps' array layers hold at the limits every device
    /// grants.
    pub(crate) const HORIZON_RECORD_VECTORS: u32 = 4 + engine_core::MAX_BOUNCE_LEVELS / 4 => ["HORIZON_RECORD_VECTORS"];
    pub(crate) const HORIZON_RECORDS: usize = 64 * HORIZON_RECORD_VECTORS as usize => ["HORIZON_RECORDS"];

    // crate::terrain_bounce
    /// Words of a probe in the probes' buffer.
    pub(crate) const BOUNCE_RECORD_WORDS: usize = engine_core::terrain_data::bounce::PROBE_BYTES / 4 => ["BOUNCE_RECORD_WORDS" in BOUNCE_SHADER];

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
    /// Layers of the sheets' array a sheet takes: its base color, its
    /// cover's canopy and its normals' turn (sRGB, linear data stored as is).
    pub(crate) const VT_SHEET_LAYERS: u32 = 3 => ["VT_SHEET_LAYERS" in TERRAIN_VT_FAR_SHADER];
    /// Bytes of a row of a slot's layer as copies between the sheets and
    /// buffers take it: whole multiples of what copies align rows to.
    pub(crate) const VT_ROW_BYTES: u32 = (VT_SLOT_TEXELS * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) => ["VT_ROW_BYTES" in TERRAIN_VT_FAR_SHADER];

    // crate::terrain_maps
    /// The most anisotropy surface materials' maps filter to, their
    /// sampler's clamp: past it their filtering under-samples a pixel's
    /// footprint along its long side, so shaders sampling them widen their
    /// derivatives' short side to at least the long side over it.
    pub(crate) const MATERIAL_ANISOTROPY: u16 = 16 => ["TERRAIN_ANISOTROPY" in SHADER];

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

    // crate::terrain::water_blocks
    /// Samples along a side of a block of tiles' water: a patch's, its edges
    /// included, as its grid's vertices.
    pub(crate) const WATER_BLOCK: u32 = TERRAIN_GRID => ["WATER_BLOCK"];
    /// Blocks along a side of a layer of the water arrays.
    pub(crate) const WATER_BLOCKS_SIDE: u32 = 16 => ["WATER_BLOCKS_SIDE"];
    /// No block: a patch whose parent tile's water holds none over it.
    pub(crate) const NO_WATER_BLOCK: u32 = u32::MAX => ["NO_WATER_BLOCK"];
    /// The surface a block's sample no water reaches holds: below any
    /// ground, so water never shows there.
    pub(crate) const NO_WATER_SURFACE: f32 = -1.0e30 => ["NO_WATER_SURFACE"];
    /// Meters a second either way a block's packed flow spans.
    pub(crate) const WATER_FLOW_RANGE: f32 = engine_core::terrain_data::WATER_FLOW_RANGE => ["WATER_FLOW_RANGE"];
    /// Meters of half width a block's packed width of a river's water spans.
    pub(crate) const WATER_WIDTH_RANGE: f32 = engine_core::terrain_data::WATER_WIDTH_RANGE => ["WATER_WIDTH_RANGE"];
    /// Per second of shear a block's packed shear spans.
    pub(crate) const WATER_SHEAR_RANGE: f32 = engine_core::terrain_data::WATER_SHEAR_RANGE => ["WATER_SHEAR_RANGE"];
    /// Meters a step of a block's packed offsets to obstacles spans, and of
    /// their packed radius.
    pub(crate) const WATER_OBSTACLE_STEP: f32 = engine_core::terrain_data::WATER_OBSTACLE_STEP => ["WATER_OBSTACLE_STEP"];
    pub(crate) const WATER_RADIUS_STEP: f32 = engine_core::terrain_data::WATER_RADIUS_STEP => ["WATER_RADIUS_STEP"];

    // crate::water_waves
    /// Texels along a side of a tile of water's waves: wave numbers of its
    /// spectrum, and points of its slopes.
    pub(crate) const WAVE_TEXELS: u32 = engine_core::table_data::waves::TEXELS => ["WAVE_TEXELS" in WATER_WAVES_SHADER];
    /// Tiles of water's waves, each repeating at its own length.
    pub(crate) const WAVE_TILES: usize = engine_core::table_data::waves::TILES => ["WAVE_TILES" in WATER_WAVES_SHADER, "WATER_WAVE_TILES"];
    /// Invocations transforming a line of a tile: a butterfly each.
    pub(crate) const WAVE_LINE_GROUP: u32 = 128 => ["@evolve_rows" in WATER_WAVES_SHADER, "@transform_columns" in WATER_WAVES_SHADER];
    /// Workgroup side of the moments' levels.
    pub(crate) const WAVE_LEVEL_GROUP: u32 = 8 => ["@average_level" in WATER_WAVES_SHADER];

    // crate::water_sim
    /// Workgroup side of the simulation's passes over cells and faces: a
    /// block of cells, as its slots store them and water draws them.
    pub(crate) const WATER_SIM_GROUP: u32 = engine_core::terrain_data::BLOCK_SIDE => ["WATER_SIM_GROUP" in WATER_SIM_SHADER, "SIM_WATER_BLOCK" in SHADER, "@sim_exchange" in WATER_SIM_SHADER, "@sim_waves" in WATER_SIM_SHADER, "@sim_seconds" in WATER_SIM_SHADER, "@sim_drain" in WATER_SIM_SHADER, "@sim_update" in WATER_SIM_SHADER, "@sim_carry" in WATER_SIM_SHADER, "@sim_restart" in WATER_SIM_SHADER, "@sim_churn" in WATER_SIM_SHADER, "@sim_churned" in WATER_SIM_SHADER, "@sim_tally" in WATER_SIM_SHADER, "@sim_activity" in WATER_SIM_SHADER, "@sim_look" in WATER_SIM_SHADER];
    /// The most rectangles a dispatch of the simulation runs, each a slot
    /// of its buffers.
    pub(crate) const SIM_SLOTS: usize = 64 => ["SIM_SLOTS" in WATER_SIM_SHADER];
    /// Entries of four floats a settling cell's sums take (its water, what
    /// it loses and how it varies, how it moves, and the seconds it ran at
    /// the fastest water runs and the seconds summed), and a block's sums at
    /// a look take (the water its flowing cells hold, their speeds and
    /// counts; their energy, heave, swirl and those held back; and the water
    /// standing against the blocks the slot does not store).
    pub(crate) const SIM_CELL_SUMS: u64 = 4 => ["SIM_CELL_SUMS" in WATER_SIM_SHADER];
    pub(crate) const SIM_BLOCK_LOOKS: u64 = 3 => ["SIM_BLOCK_LOOKS" in WATER_SIM_SHADER];
    /// A rectangle's flags: it steps; each cell by its own seconds; water
    /// leaves over its edges freely and none comes in; it sums each cell's
    /// water over its steps; the dispatch looks at its sums; and only its
    /// blocks water can reach step.
    pub(crate) const SIM_STEPPING: u32 = 1 => ["SIM_STEPPING" in WATER_SIM_SHADER];
    pub(crate) const SIM_LOCAL: u32 = 2 => ["SIM_LOCAL" in WATER_SIM_SHADER];
    pub(crate) const SIM_OPEN: u32 = 4 => ["SIM_OPEN" in WATER_SIM_SHADER];
    pub(crate) const SIM_SUMMING: u32 = 8 => ["SIM_SUMMING" in WATER_SIM_SHADER];
    pub(crate) const SIM_LOOKING: u32 = 16 => ["SIM_LOOKING" in WATER_SIM_SHADER];
    pub(crate) const SIM_SPARSE: u32 = 32 => ["SIM_SPARSE" in WATER_SIM_SHADER];
    /// Cells rivers bring water to a workgroup of the simulation's source
    /// pass takes.
    pub(crate) const WATER_SIM_SOURCE_GROUP: u32 = 64 => ["WATER_SIM_SOURCE_GROUP" in WATER_SIM_SHADER, "@sim_source" in WATER_SIM_SHADER];

    // crate::water_fall_map
    /// How many times longer along falling water than across it water
    /// reads falling water's map.
    pub(crate) const FALL_MAP_STRETCH: f32 = engine_core::table_data::falls::STRETCH => ["FALL_MAP_STRETCH"];

    // crate::ground_cover
    /// Bytes of a blade (`CoverBlade`).
    pub(crate) const BLADE_BYTES: u64 = 32 => ["size_of CoverBlade"];
    /// The share of ground cover's radius within which it grows densest:
    /// past it blades thin as this share of the radius over the distance,
    /// widening as much to cover as much ground, up to its inverse at the
    /// radius (`terrain_variety.wgsl`'s `cover_share`).
    pub(crate) const COVER_CORE: f32 = 0.2 => ["COVER_CORE"];
    /// Joints along a blade too short on screen to show its curve, which
    /// the blades' second indirect draw draws straight.
    pub(crate) const STRAIGHT_JOINTS: u32 = 3 => ["STRAIGHT_JOINTS"];
    /// The second draw's first vertex, by which the vertex stage tells its
    /// straight blades from whole ones.
    pub(crate) const STRAIGHT_FIRST_VERTEX: u32 = 1024 => ["STRAIGHT_FIRST_VERTEX"];

    // crate::bindings
    /// A pyramid level's `Extent` of `depth_pyramid.wgsl`.
    pub(crate) const PYRAMID_EXTENT_BYTES: usize = 16 => ["size_of Extent" in DEPTH_PYRAMID_SHADER];

    // crate::gpu_layouts::terrain
    /// Surface materials every terrain together can show: as many as fit the
    /// 16 KiB uniform binding every device allows.
    pub(crate) const PALETTE_ENTRIES: usize = 16_384 / size_of::<PaletteEntry>() => ["TERRAIN_PALETTE_ENTRIES" in SHADER, "PALETTE_ENTRIES" in COVER_GROW_SHADER, "PALETTE_ENTRIES" in BOUNCE_SHADER, "PALETTE_ENTRIES" in SCATTER_SNAP_SHADER];

    /// Cells along a side of the cloud shadows' grid.
    pub(crate) const SHADOW_CELLS: u32 = 128 => ["CLOUD_SHADOW_CELLS"];
    /// Columns along a side of the clouds' light map, and the heights
    /// through the layer each holds.
    pub(crate) const LIGHT_TEXELS: u32 = 384 => ["LIGHT_TEXELS" in CLOUDS_SHADER];
    pub(crate) const LIGHT_HEIGHTS: u32 = 16 => ["LIGHT_HEIGHTS" in CLOUDS_SHADER];

    // crate::sky
    /// The light falling at noon the physically based sky's exposure adapts
    /// toward, a multiple of the sun's intensity, and the share of its
    /// adaptation setting that is the power of the light's ratio exposure
    /// rises by, and the most it rises by, as a multiple (auto exposure's
    /// time of day follows them).
    pub(crate) const NOON_LIGHT: f32 = 1.1 => ["NOON_LIGHT" in SKY_SHADER];
    pub(crate) const ADAPTATION_POWER: f32 = 0.75 => ["ADAPTATION_POWER" in SKY_SHADER];
    pub(crate) const ADAPTATION_MOST: f32 = 16.0 => ["ADAPTATION_MOST" in SKY_SHADER];

    // crate::exposure
    /// Bins of auto exposure's histogram of light in stops, and the samples
    /// its meter reads across and down the view, in workgroups of a side.
    pub(crate) const EXPOSURE_BINS: u32 = 128 => ["EXPOSURE_BINS" in EXPOSURE_SHADER];
    pub(crate) const EXPOSURE_GRID: u32 = 128 => ["EXPOSURE_GRID" in EXPOSURE_SHADER];
    pub(crate) const EXPOSURE_GROUP: u32 = 16 => ["@meter_exposure" in EXPOSURE_SHADER];

    // crate::gpu_data::Variant: the bits of a material's variant passes
    // drawn rarely read from its uniform.
    pub(crate) const VARIANT_MAPS: u32 = 1 => ["VARIANT_MAPS" in SHADER];
    pub(crate) const VARIANT_NORMAL_MAP: u32 = 2 => ["VARIANT_NORMAL_MAP" in SHADER];
    pub(crate) const VARIANT_DOUBLE_SIDED: u32 = 8 => ["VARIANT_DOUBLE_SIDED" in SHADER];
    pub(crate) const VARIANT_METALLIC_ROUGHNESS_MAP: u32 = 128 => ["VARIANT_METALLIC_ROUGHNESS_MAP" in SHADER];
    pub(crate) const VARIANT_OCCLUSION_MAP: u32 = 256 => ["VARIANT_OCCLUSION_MAP" in SHADER];
    pub(crate) const VARIANT_EMISSIVE_MAP: u32 = 512 => ["VARIANT_EMISSIVE_MAP" in SHADER];
    pub(crate) const VARIANT_PACKED_OCCLUSION: u32 = 1024 => ["VARIANT_PACKED_OCCLUSION" in SHADER];
    pub(crate) const VARIANT_WORLD_MAPPING: u32 = 4096 => ["VARIANT_WORLD_MAPPING" in SHADER];
    pub(crate) const VARIANT_TOP_LAYER: u32 = 8192 => ["VARIANT_TOP_LAYER" in SHADER];
    pub(crate) const VARIANT_FOLIAGE: u32 = 65536 => ["VARIANT_FOLIAGE" in SHADER];

    // crate::probes
    /// Words of an entry of the reflection probes' grid lists.
    pub(crate) const PROBE_ENTRY_WORDS: usize = 3 => ["PROBE_ENTRY_WORDS" in SHADER];

    // crate::debug_view
    /// Views of the debug view's pass (`debug_view.wgsl`), by
    /// `engine_core::DebugViewMode`.
    pub(crate) const DEBUG_VIEW_KINDS: u32 = 1 => ["DEBUG_VIEW_KINDS" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_LEVELS: u32 = 2 => ["DEBUG_VIEW_LEVELS" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_CARDS: u32 = 3 => ["DEBUG_VIEW_CARDS" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_SHADOWS: u32 = 4 => ["DEBUG_VIEW_SHADOWS" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_CASTERS: u32 = 5 => ["DEBUG_VIEW_CASTERS" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_OVERDRAW: u32 = 6 => ["DEBUG_VIEW_OVERDRAW" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_NORMALS: u32 = 7 => ["DEBUG_VIEW_NORMALS" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_TILES: u32 = 8 => ["DEBUG_VIEW_TILES" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_MATERIALS: u32 = 9 => ["DEBUG_VIEW_MATERIALS" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_SKY: u32 = 10 => ["DEBUG_VIEW_SKY" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_DETAIL: u32 = 11 => ["DEBUG_VIEW_DETAIL" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_WATER_DEPTH: u32 = 12 => ["DEBUG_VIEW_WATER_DEPTH" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_WATER_SPEED: u32 = 13 => ["DEBUG_VIEW_WATER_SPEED" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_WATER_FROUDE: u32 = 14 => ["DEBUG_VIEW_WATER_FROUDE" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_WATER_TURBULENCE: u32 = 15 => ["DEBUG_VIEW_WATER_TURBULENCE" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_WATER_WHITE: u32 = 16 => ["DEBUG_VIEW_WATER_WHITE" in DEBUG_VIEW_SHADER];
    pub(crate) const DEBUG_VIEW_WATER_REACH: u32 = 17 => ["DEBUG_VIEW_WATER_REACH" in DEBUG_VIEW_SHADER];
    /// Gravity, and the meters of water under which it lies dry, by which
    /// debug views reckon water's Froude number (`meshes/debug.wgsl`): the
    /// shallow-water scheme's.
    pub(crate) const DEBUG_GRAVITY: f32 = engine_core::terrain_data::hydraulics::GRAVITY => ["DEBUG_GRAVITY" in SHADER];
    pub(crate) const DEBUG_DRY: f32 = engine_core::terrain_data::shallow_water::DRY => ["DEBUG_DRY" in SHADER];
    /// What drew a pixel of the debug targets (`meshes/debug.wgsl`): its
    /// kind in the low four bits of its draw's record, its level of detail
    /// in the four above, then flags; the sun's lookup above
    /// `DEBUG_SOURCE_SHIFT`.
    pub(crate) const DEBUG_KIND_MESH: u32 = 1 => ["DEBUG_KIND_MESH"];
    pub(crate) const DEBUG_KIND_PLACED: u32 = 2 => ["DEBUG_KIND_PLACED"];
    pub(crate) const DEBUG_KIND_CARD: u32 = 3 => ["DEBUG_KIND_CARD"];
    pub(crate) const DEBUG_KIND_PLACED_CARD: u32 = 4 => ["DEBUG_KIND_PLACED_CARD"];
    pub(crate) const DEBUG_KIND_VIRTUAL: u32 = 5 => ["DEBUG_KIND_VIRTUAL"];
    pub(crate) const DEBUG_KIND_TERRAIN: u32 = 6 => ["DEBUG_KIND_TERRAIN"];
    pub(crate) const DEBUG_KIND_COVER: u32 = 7 => ["DEBUG_KIND_COVER"];
    pub(crate) const DEBUG_KIND_WATER: u32 = 8 => ["DEBUG_KIND_WATER"];
    pub(crate) const DEBUG_KIND_SEA: u32 = 9 => ["DEBUG_KIND_SEA"];
    pub(crate) const DEBUG_KIND_FALLS: u32 = 10 => ["DEBUG_KIND_FALLS"];
    pub(crate) const DEBUG_KIND_SKY: u32 = 11 => ["DEBUG_KIND_SKY"];
    pub(crate) const DEBUG_LEVEL_SHIFT: u32 = 4 => ["DEBUG_LEVEL_SHIFT"];
    /// A card fading in or out; a draw casting sun shadows; placed objects'
    /// casters the kept map holds.
    pub(crate) const DEBUG_FADING: u32 = 0x100 => ["DEBUG_FADING"];
    pub(crate) const DEBUG_CASTS: u32 = 0x200 => ["DEBUG_CASTS"];
    pub(crate) const DEBUG_KEPT: u32 = 0x400 => ["DEBUG_KEPT"];
    /// The sun's lookup at a pixel: none, the near cascades from the first,
    /// the kept levels from the finest, the kept far map, the moving map, a
    /// terrain's horizon, past the shadow distance; and the flag of a point
    /// it leaves in shadow.
    pub(crate) const DEBUG_SOURCE_SHIFT: u32 = 16 => ["DEBUG_SOURCE_SHIFT"];
    pub(crate) const DEBUG_SUN_NONE: u32 = 0 => ["DEBUG_SUN_NONE"];
    pub(crate) const DEBUG_SUN_CASCADE: u32 = 1 => ["DEBUG_SUN_CASCADE"];
    pub(crate) const DEBUG_SUN_LEVEL: u32 = DEBUG_SUN_CASCADE + MAX_CASCADES as u32 => ["DEBUG_SUN_LEVEL"];
    pub(crate) const DEBUG_SUN_KEPT: u32 = DEBUG_SUN_LEVEL + MAX_LEVELS as u32 => ["DEBUG_SUN_KEPT"];
    pub(crate) const DEBUG_SUN_MOVING: u32 = DEBUG_SUN_KEPT + 1 => ["DEBUG_SUN_MOVING"];
    pub(crate) const DEBUG_SUN_HORIZON: u32 = DEBUG_SUN_MOVING + 1 => ["DEBUG_SUN_HORIZON"];
    pub(crate) const DEBUG_SUN_BEYOND: u32 = DEBUG_SUN_HORIZON + 1 => ["DEBUG_SUN_BEYOND"];
    pub(crate) const DEBUG_SHADOWED: u32 = 0x10 => ["DEBUG_SHADOWED"];
    /// Terrain pixels (`meshes/terrain_debug.wgsl`): on a tile's edge, on a
    /// patch's, in their record, with the patch within its tile above
    /// `DEBUG_PATCH_SHIFT`; a material slot holding none.
    pub(crate) const DEBUG_TILE_EDGE: u32 = 0x800 => ["DEBUG_TILE_EDGE"];
    pub(crate) const DEBUG_PATCH_EDGE: u32 = 0x1000 => ["DEBUG_PATCH_EDGE"];
    pub(crate) const DEBUG_PATCH_SHIFT: u32 = 24 => ["DEBUG_PATCH_SHIFT"];
    pub(crate) const DEBUG_NO_MATERIAL: u32 = 0xff => ["DEBUG_NO_MATERIAL"];

    // crate::occupancy
    /// Cells along a side of an occupancy map, which its window follows the
    /// camera by and frames redraw, a bit each, sixteen a row.
    pub(crate) const OCCUPANCY_CELLS: u32 = 16 => ["CELLS" in OCCUPANCY_SHADER];
    /// Vertices of a footprint's or a cell's quad: two triangles.
    pub(crate) const OCCUPANCY_QUAD: u32 = 6 => ["QUAD" in OCCUPANCY_SHADER];

    // Workgroup sides that dispatches divide by.
    pub(crate) const SCATTER_GROUP: u32 = 64 => ["@snap_scatter" in SCATTER_SNAP_SHADER, "@cull_scatter" in SCATTER_CULL_SHADER];
    pub(crate) const OCCUPANCY_GROUP: u32 = 64 => ["@select_covers" in OCCUPANCY_SHADER];
    pub(crate) const DEFORMATION_GROUP: u32 = 8 => ["@press" in DEFORMATION_SHADER, "@slopes" in DEFORMATION_SHADER, "@scroll" in DEFORMATION_SHADER, "@wipe" in DEFORMATION_SHADER, "@rebase" in DEFORMATION_SHADER];
    pub(crate) const CLOUD_GROUP: u32 = 8 => ["@march_view" in CLOUDS_SHADER, "@march_clouds" in CLOUDS_SHADER, "@shadow_clouds" in CLOUDS_SHADER, "@light_clouds" in CLOUDS_SHADER];
    pub(crate) const SKY_TABLE_GROUP: u32 = 8 => ["@build_sky_view" in SKY_SHADER];
    pub(crate) const CONTROL_GROUP: u32 = 8 => ["@tile_control" in TERRAIN_CONTROL_SHADER];
    pub(crate) const BOUNCE_GROUP: u32 = 8 => ["@bounce_ground" in BOUNCE_SHADER, "@bounce_down" in BOUNCE_SHADER, "@bounce_gather" in BOUNCE_SHADER];
    pub(crate) const NORMALS_GROUP: u32 = 16 => ["@tile_normals" in TERRAIN_NORMALS_SHADER];
    pub(crate) const RESOLVE_ARGS_GROUP: u32 = 64 => ["@resolve_args" in VISIBILITY_SHADER];
    pub(crate) const VT_FEEDBACK_GROUP: u32 = 8 => ["@cs_terrain_vt_feedback" in SHADER];
    pub(crate) const VT_FAR_GROUP: u32 = 8 => ["@far_mean" in TERRAIN_VT_FAR_SHADER];
    /// Pixels a side of the blocks the virtual texture's feedback tests one
    /// pixel of, a different one each frame.
    pub(crate) const VT_FEEDBACK_BLOCK: u32 = 8 => ["VT_FEEDBACK_BLOCK" in SHADER];

    // The wind's layers (`wind.wgsl`), which meshes, depth passes and shadow
    // casters sway placed objects by alike: each a share of the object's
    // one sway (`PlacementAsset::sway`), the trunk's.
    /// The wind speed in which placed objects sway fully, meters a second,
    /// and the meters up a trunk, and out from its axis, at which they sway
    /// their sway.
    pub(crate) const WIND_FULL_SPEED: f32 = 12.0 => ["WIND_FULL_SPEED"];
    pub(crate) const WIND_SPAN: f32 = 10.0 => ["WIND_SPAN"];
    /// The trunk's own flutter about its gusts: its share of the push and
    /// its rate (radians a second).
    pub(crate) const WIND_TRUNK_FLUTTER: f32 = 0.12 => ["WIND_TRUNK_FLUTTER"];
    pub(crate) const WIND_TRUNK_RATE: f32 = 4.7 => ["WIND_TRUNK_RATE"];
    /// Branches' bob, as far up as along the wind: its share of the sway,
    /// growing with the square of a vertex's reach from the object's
    /// vertical axis; its wave's rate (radians a second), and how far the
    /// wave's phase turns a meter up the object (radians), besides three
    /// turns around it.
    pub(crate) const WIND_BRANCH: f32 = 4.0 => ["WIND_BRANCH"];
    pub(crate) const WIND_BRANCH_RATE: f32 = 3.1 => ["WIND_BRANCH_RATE"];
    pub(crate) const WIND_BRANCH_RISE: f32 = 2.1 => ["WIND_BRANCH_RISE"];
    /// Needles' and leaves' flutter, up and across the wind: its share of
    /// the sway, growing with a vertex's reach, and its rate (radians a
    /// second).
    pub(crate) const WIND_FLUTTER: f32 = 0.2 => ["WIND_FLUTTER"];
    pub(crate) const WIND_FLUTTER_RATE: f32 = 13.0 => ["WIND_FLUTTER_RATE"];
    /// The pixels on screen, seen from the detail eye, a shaking layer's
    /// widest swing spans where it shows whole and where it has stilled
    /// (the gusts' swing about the trunk's mean bend, branches' bob and
    /// needles' flutter): as other detail, it hands over once it moves the
    /// object less than a pixel, while the mean bend holds at every
    /// distance.
    pub(crate) const WIND_SHAKE_WHOLE: f32 = 1.0 => ["WIND_SHAKE_WHOLE"];
    pub(crate) const WIND_SHAKE_STILL: f32 = 0.5 => ["WIND_SHAKE_STILL"];

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
    engine_core::table_data::sky::MULTIPLE_SIDE => ["SKY_MULTIPLE_SIZE"];
    engine_core::table_data::falls::SPREAD => ["FALL_MAP_SPREAD"];
    engine_core::terrain_data::hydraulics::GRAVITY => ["SIM_GRAVITY" in WATER_SIM_SHADER, "WATER_GRAVITY"];
    engine_core::terrain_data::shallow_water::THETA => ["SIM_THETA" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::THIN => ["SIM_THIN" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::DRY => ["SIM_DRY" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::FASTEST => ["SIM_FASTEST" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::SOURCE_SPEED => ["SIM_SOURCE_SPEED" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::UNHELD => ["SIM_UNHELD" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::DENSITY => ["SIM_DENSITY" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::FERGUSON_A1 => ["SIM_FERGUSON_A1" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::FERGUSON_A2 => ["SIM_FERGUSON_A2" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::SHALLOWEST => ["SIM_SHALLOWEST" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::DEEPEST => ["SIM_DEEPEST" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::STRICKLER => ["SIM_STRICKLER" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::CELL_MEANS => ["SIM_CELL_MEANS" in WATER_SIM_SHADER];
    engine_core::terrain_data::NO_BLOCK => ["SIM_NO_BLOCK" in WATER_SIM_SHADER];
    engine_core::terrain_data::BLOCK_GROUND => ["SIM_BLOCK_GROUND" in WATER_SIM_SHADER, "SIM_BLOCK_GROUND" in SHADER];
    engine_core::terrain_data::shallow_water::WET_MOST => ["SIM_WET_MOST" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::FOAM_LIFE => ["SIM_FOAM_LIFE" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::FOAM_MOST => ["SIM_FOAM_MOST" in WATER_SIM_SHADER, "WATER_FOAM_MOST"];
    engine_core::terrain_data::shallow_water::FOAM_WET => ["SIM_FOAM_WET" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::FOAM_LOSS => ["SIM_FOAM_LOSS" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::FOAM_DROP => ["SIM_FOAM_DROP" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::FOAM_JUMP => ["SIM_FOAM_JUMP" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::FOAM_SWIRL => ["SIM_FOAM_SWIRL" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::FOAM_PILING => ["SIM_FOAM_PILING" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::FOAM_POWER => ["SIM_FOAM_POWER" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::COURANT => ["SIM_COURANT" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::LOCAL_SHARE => ["SIM_LOCAL_SHARE" in WATER_SIM_SHADER];
    engine_core::terrain_data::shallow_water::SLOWEST => ["SIM_SLOWEST" in WATER_SIM_SHADER];
    engine_core::terrain_data::TURBULENT_SPEED => ["WATER_TURBULENT_SPEED"];
    engine_core::terrain_data::TURBULENT_SWIRL => ["WATER_TURBULENT_SWIRL"];
    engine_core::terrain_data::BREAK_ONSET => ["WATER_BREAK_ONSET"];
    engine_core::terrain_data::BREAK_FULL => ["WATER_BREAK_FULL"];
    engine_core::terrain_data::BREAK_THINNEST => ["WATER_BREAK_THINNEST"];
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
    engine_core::GUST_SLOW_SHARE => ["GUST_SLOW_SHARE"];
    engine_core::GUST_SLOW_RATE => ["GUST_SLOW_RATE"];
    engine_core::GUST_SLOW_TRAVEL => ["GUST_SLOW_TRAVEL"];
    engine_core::GUST_QUICK_SHARE => ["GUST_QUICK_SHARE"];
    engine_core::GUST_QUICK_RATE => ["GUST_QUICK_RATE"];
    engine_core::GUST_QUICK_TRAVEL => ["GUST_QUICK_TRAVEL"];
    engine_core::GUST_QUICK_ACROSS => ["GUST_QUICK_ACROSS"];
    engine_core::RIM_REACH => ["RIM_REACH" in DEFORMATION_SHADER];
    engine_core::terrain_data::lighting::DIRECTIONS => ["HORIZON_DIRECTIONS"];
    engine_core::terrain_data::lighting::HORIZON_LAYERS => ["HORIZON_LAYERS"];
    engine_core::terrain_data::bounce::BINS => ["BOUNCE_BINS" in BOUNCE_SHADER];
    engine_core::terrain_data::bounce::LAYERS => ["BOUNCE_PROBE_LAYERS" in BOUNCE_SHADER];
    engine_core::terrain_data::bounce::LOW_PROBE => ["BOUNCE_LOW"];
    engine_core::terrain_data::bounce::HIGH_PROBE => ["BOUNCE_HIGH"];
    engine_core::terrain_data::bounce::MAX_PROBES => ["BOUNCE_WINDOW_ROW"];
    engine_core::terrain_data::bounce::NEAREST => ["BOUNCE_NEAREST" in BOUNCE_SHADER];
    engine_core::terrain_data::bounce::DISTANCE_STEPS => ["BOUNCE_STEPS" in BOUNCE_SHADER];
    engine_core::terrain_data::TILE_STRIDE => ["TILE_STRIDE" in TERRAIN_CONTROL_SHADER, "TILE_STRIDE" in TERRAIN_NORMALS_SHADER];
    engine_core::terrain_data::RIVER_FAR => ["RIVER_FAR" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::RIVER_EDGE_STEP => ["RIVER_EDGE_STEP" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::RIVER_EDGE_ZERO => ["RIVER_EDGE_ZERO" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::RIVER_EDGE_FAR => ["RIVER_EDGE_FAR" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::PATH_FAR => ["PATH_FAR" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::MAX_STACK => ["MAX_STACK" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::MAX_MATERIALS => ["MAX_MATERIALS" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::PROGRAM => ["H_PROGRAM" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::BIOMES => ["H_BIOMES" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::READS => ["H_READS" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::HYDROLOGY => ["H_HYDROLOGY" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::PATHS => ["H_PATHS" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::FADE => ["H_FADE" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::header::WET_BIOMES => ["H_WET_BIOMES" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::BIOME_WORDS => ["BIOME_WORDS" in TERRAIN_CONTROL_SHADER];
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
    engine_core::terrain_data::gpu::op::GRAIN => ["OP_GRAIN" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::POWER => ["OP_POWER" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::FLOOD => ["OP_FLOOD" in TERRAIN_CONTROL_SHADER];
    engine_core::terrain_data::gpu::op::END => ["OP_END" in TERRAIN_CONTROL_SHADER];
}
