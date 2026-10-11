//! Upload structs of `crate::gpu_data`.

use bytemuck::{Pod, Zeroable};

/// Most near shadow cascades.
pub(crate) const MAX_CASCADES: usize = engine_core::MAX_SHADOW_CASCADES as usize;
/// Most kept shadow levels.
pub(crate) const MAX_LEVELS: usize = engine_core::MAX_KEPT_LEVELS as usize;
/// Planes of a map's caster volume ([`ShadowCullUniform::planes`]): the
/// view frustum's sides, near and far planes, and the receivers' first and
/// last view depths, those facing the sun.
pub(crate) const CASTER_PLANES: usize = 8;

/// A slot pose as WGSL `Rows` reads it: core's three affine rows, with the
/// implicit final row [0, 0, 0, 1]. Frames upload `RenderFrame::pose_rows`
/// as they are.
pub(crate) type PoseData = engine_core::PoseRows;

/// WGSL `grounds` of `meshes/declarations.wgsl`: what lies
/// under a placed object, x and z of the ground's normal as two signed
/// 16-bit fractions, the meters its base sinks below the ground, the meters
/// up its stain reaches (0: none) and the stain's linear color as four
/// bytes.
pub(crate) type GroundData = [u32; 4];

gpu_structs! {
    /// The camera as shaders read it. `view_projection` is the rasterized
    /// projection.
    /// Temporally antialiased frames jitter it by `jitter.xy` clip
    /// units per w, and their motion vectors use the previous frame's unjittered
    /// `previous_view_projection` and `jitter.zw`, two over the viewport size in
    /// pixels. Other frames repeat the view-projection and keep the jitter zero.
    /// The unjittered `inverse_view_projection` turns pixels into sky rays and
    /// `position` (xyz) is the eye, for view-dependent lighting;
    /// its w is presentation time in seconds, for animated surfaces.
    /// `options.x` is 1 while a reflection probe captures a cube face:
    /// its surfaces read shadows from the far maps alone, which
    /// hold every caster whatever the main camera sees; `options.y` is the angle
    /// a pixel spans at the view's center ([`pixel_angle`](crate::gpu_data::pixel_angle));
    /// `options.z` turns fades' dither each temporally antialiased frame.
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct CameraUniform as ["Camera"] {
        pub view_projection: [[f32; 4]; 4],
        pub previous_view_projection: [[f32; 4]; 4],
        pub inverse_view_projection: [[f32; 4]; 4],
        pub position: [f32; 4],
        pub jitter: [f32; 4],
        pub options: [f32; 4],
        /// The wind's velocity (x and z), gust strength and the previous
        /// frame's presentation time.
        pub wind: [f32; 4],
    }

    /// The scene's atmosphere and grade, as `scene.wgsl` reads them.
    /// Scalars ride in the fourth components.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct LightingUniform as ["Lighting"] {
        /// Normalized direction toward the sun; w: rim strength.
        pub sun: [f32; 4],
        /// w: specular strength.
        pub sun_color: [f32; 4],
        /// w: ambient scale.
        pub zenith: [f32; 4],
        /// w: exposure.
        pub horizon: [f32; 4],
        /// w: saturation.
        pub ground: [f32; 4],
        /// Start, density, height falloff and maximum.
        pub fog: [f32; 4],
        /// Emission scale, the exposed brightness above which light glows,
        /// the roughness beyond which surfaces reflect only the
        /// sky, and 1 when screen-space reflections run.
        pub effects: [f32; 4],
        /// The contrast curve's weight (see
        /// [`contrast_weight`](crate::gpu_data::contrast_weight)) and vibrance;
        /// 1 with the physically based sky; w
        /// unused.
        pub grade: [f32; 4],
        /// Luminance-normalized split-toning gains of shadows and highlights; w
        /// unused.
        pub shadow_tint: [f32; 4],
        pub highlight_tint: [f32; 4],
        /// Refraction through glass: the bend's strength (0
        /// without refraction) and the blur radius in pixels of fully rough
        /// glass; zw unused.
        pub refraction: [f32; 4],
        /// Aerial perspective: start, density at height zero,
        /// height falloff and the sun's glow; the air's color (w unused).
        pub aerial: [f32; 4],
        pub aerial_color: [f32; 4],
        /// Occupancy: the map's center (x and z), the inverse of
        /// its side in meters and the strength covers darken by (0 without a
        /// map); then the reference height its heights are above and its texels
        /// a side.
        pub occupancy: [f32; 4],
        pub occupancy_heights: [f32; 4],
        pub occupancy_far: [f32; 4],
        /// Under the sea: the light its water scatters toward
        /// the eye (rgb) and the meters through which half the light passes
        /// (w); zero above it.
        pub underwater: [f32; 4],
        /// The sea: its still level, the meters through which
        /// half the light passes, and 1 while it stands (0 without one); w
        /// unused.
        pub sea: [f32; 4],
        /// Clouds' shadows: the grid's first cell's corner (x
        /// and z), cells per meter and the shadows' strength (0 without
        /// clouds); then the clouds' base height.
        pub cloud_shadow: [f32; 4],
        pub cloud_layer: [f32; 4],
    }

    /// Separate per-instance color, opacity and packed surface; unchanged colors
    /// need no pose upload. The color is packed by
    /// [`pack_color`](crate::gpu_data::pack_color).
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct ColorData as ["SlotColor"] {
        pub color: [u32; 2],
        pub surface: u32,
        pub flags: u32,
    }

    /// A vertex: the position, the [`octahedral`](crate::gpu_data::octahedral) normal and texture
    /// coordinates.
    /// The CPU asset representation has no GPU byte-layout contract.
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct Vertex {
        pub position: [f32; 3],
        pub normal: [i16; 2],
        pub uv: [f32; 2],
    }

    /// The second vertex stream of meshes with vertex colors or second texture
    /// coordinates: linear RGBA as four 16-bit
    /// fractions (white without colors) and the second coordinates (the first
    /// without them).
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ExtraVertex {
        pub color: [u16; 4],
        pub uv1: [f32; 2],
    }

    /// The skin stream of meshes with joints: four
    /// joint indices and their weights as 16-bit fractions, which the shaders
    /// renormalize.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SkinVertex {
        pub joints: [u16; 4],
        pub weights: [u16; 4],
    }

    /// A material's maps as `meshes/declarations.wgsl` reads them:
    /// each map's texture coordinate transform, the emissive color and the map
    /// parameters of one material, and its top layer.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct MaterialUniform as ["MaterialMaps"] {
        /// Rows of each map's 2x3 transform (base color, metallic-roughness,
        /// normal, occlusion and emissive), w unused.
        pub transforms: [[f32; 4]; 10],
        /// Linear emissive color; w the meters one repeat of world-mapped maps
        /// covers, 0 for texture coordinates.
        pub emissive: [f32; 4],
        /// Normal scale, occlusion strength, alpha cutoff, and the maps reading
        /// the second texture coordinates as bits in map order.
        pub params: [f32; 4],
        /// Transmission: its share, index of refraction and
        /// thickness; w the material's variant bits as `f32` bits, which uber
        /// pipelines read.
        pub volume: [f32; 4],
        /// The top layer: its linear color and coverage, then its
        /// roughness.
        pub top: [f32; 4],
        pub layer: [f32; 4],
    }

    /// WGSL `Shadow` uniform of `scene.wgsl`:
    /// world to each near cascade, to the far shadow maps and to light space,
    /// each kept level's part of it, and their settings.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ShadowUniform as ["Shadow"] {
        /// World to each near cascade: x and y in [-1, 1] across it, z in
        /// [0, 1].
        pub near: [[[f32; 4]; 4]; MAX_CASCADES],
        /// World to both far maps.
        pub far: [[f32; 4]; 4],
        /// Shadows on, the kept far map holds casters, the near cascades in use,
        /// and the distance where shadows begin to fade.
        pub state: [f32; 4],
        /// View distance each near cascade reaches; unreachable beyond those in
        /// use.
        pub reach: [f32; 4],
        /// Normal offsets (world units) of lookups in each near cascade.
        pub near_offsets: [f32; 4],
        /// Steepest receiver plane (depth per texel) of each near cascade.
        pub near_limits: [f32; 4],
        /// Distance where shadows are gone, the normal offset (world units) of
        /// far lookups and the far maps' steepest receiver plane; w unused.
        pub offsets: [f32; 4],
        /// Texels a side of the near cascades and the far maps; zw unused.
        pub sizes: [f32; 4],
        /// World to light space: the sun's right, up and
        /// away-from-sun axes, which every kept level scales.
        pub light: [[f32; 4]; 3],
        /// Each kept level's light space: one over its width (x and y in level
        /// widths, whose fractions place a point in its toroidal map), and its
        /// band's nearest depth and one over its depth (z in [0, 1]); w unused.
        pub level_scales: [[f32; 4]; MAX_LEVELS],
        /// Each level's window: its first corner in level widths, the normal
        /// offset (world units) of its lookups and its steepest receiver plane.
        pub level_windows: [[f32; 4]; MAX_LEVELS],
        /// The tiles of each level holding their casters: a bit per tile slot,
        /// eight words a level.
        pub level_valid: [[u32; 4]; 2 * MAX_LEVELS],
        /// Levels in use, their texels a side, a bit for each level with
        /// tiles not yet drawn, and the terrain horizon records' count.
        pub level_state: [f32; 4],
    }

    /// A shadow map's caster selection, as `shadows.wgsl` reads it. Each draw
    /// group's ID region start is in a storage buffer beside it.
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct ShadowCullUniform as ["Cull" in SHADOW_SHADER] {
        /// World to light space: the sun's right, up and away-from-sun axes.
        pub light: [[f32; 4]; 3],
        /// Light-space rectangles (minimum x and y, maximum x and y) of each
        /// near cascade, the far maps and the region of the kept map redrawn
        /// this frame.
        pub near: [[f32; 4]; MAX_CASCADES],
        pub far: [f32; 4],
        pub refresh: [f32; 4],
        /// The part of the far maps the moving map selects casters from: where
        /// visible receivers can be.
        pub moving: [f32; 4],
        /// Light-space depth beyond which casters cannot darken receivers of
        /// each near cascade; then (y and z) of the far maps and the refreshed
        /// region, and (w) of the moving map's selection.
        pub near_depths: [f32; 4],
        pub depths: [f32; 4],
        pub candidates: u32,
        pub groups: u32,
        /// Instance IDs per map.
        pub capacity: u32,
        /// `STILLNESS | REFRESH` of the shader.
        pub mode: u32,
        /// Kept-map tiles: light-space x and y of the first tile's corner, and
        /// the tile width and height.
        pub tiles: [f32; 4],
        /// Near cascades drawn this frame.
        pub cascades: u32,
        /// Skinned slots `select_skinned` visits on GPU-presented frames.
        pub skinned: u32,
        /// Kept levels in use.
        pub levels: u32,
        pub pad: u32,
        /// The light-space rectangle of the kept level redrawn this frame, and
        /// its depth band's nearest and farthest depths (zw unused).
        pub level_refresh: [f32; 4],
        pub level_depths: [f32; 4],
        /// Each kept level's tiles: light-space x and y of its first tile's
        /// corner and the tile width; w unused.
        pub level_tiles: [[f32; 4]; MAX_LEVELS],
        /// Inward world planes (`xyz . p + w >= 0` inside) bounding where
        /// casters can darken each near cascade's receivers, then the moving
        /// map's: their part of the view swept toward the sun; unused planes
        /// pass everything.
        pub planes: [[[f32; 4]; CASTER_PLANES]; MAX_CASCADES + 1],
    }

    /// WGSL `Post` uniform of `effects.wgsl`: 240
    /// bytes.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct PostUniform as ["Post"] {
        /// Unjittered clip to world space.
        pub inverse_view_projection: [[f32; 4]; 4],
        /// Unjittered world to clip space, for reflection rays.
        pub view_projection: [[f32; 4]; 4],
        /// The eye; w: depth-buffer pixels per world unit at distance 1.
        pub eye: [f32; 4],
        /// Scene target size in pixels, and its inverse.
        pub size: [f32; 4],
        /// Radius, strength, unused and 1 when ambient occlusion runs.
        pub occlusion: [f32; 4],
        /// Strength and 1 when bloom runs; half-resolution texel size.
        pub bloom: [f32; 4],
        /// Ray distance, strength, 1 when reflections run, and which pixel of
        /// each 2x2 block the half-resolution reflections trace this frame.
        pub reflections: [f32; 4],
        /// Strength, forward scattering, distance and 1 when sun shafts run.
        pub shafts: [f32; 4],
        /// The sun's exposed color; w unused.
        pub sun: [f32; 4],
        /// Contact shadows: each ray's reach in meters, the
        /// share of sunlight a hit takes away, the distance within which
        /// surfaces trace, and 1 when they run.
        pub contact: [f32; 4],
        /// The unit direction toward the sun; w unused.
        pub sun_direction: [f32; 4],
    }

    /// The grade after radiance composition, as `grade.wgsl` reads it.
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct GradeUniform as ["Grade"] {
        /// Contrast weight, saturation, vibrance, unused.
        pub parameters: [f32; 4],
        /// Shadow tint normalized to unit luminance.
        pub shadows: [f32; 4],
        /// Highlight tint normalized to unit luminance.
        pub highlights: [f32; 4],
    }

    /// Temporal antialiasing's resolve, as `temporal.wgsl` reads it.
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct ResolveUniform as ["Resolve"] {
        /// Size of one target pixel in texture coordinates.
        pub texel: [f32; 2],
        /// Weight of this frame's color against the reprojected history.
        pub current_weight: f32,
        /// Zero when the history must not be read (reset).
        pub history_valid: u32,
    }

    /// A GPU-presented frame, as `presentation.wgsl` reads it.
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct PresentationFrame as ["Frame" in PRESENTATION_SHADER] {
        pub alpha: f32,
        pub candidates: u32,
        pub budget: u32,
        pub culling: u32,
        pub groups: u32,
        pub batches: u32,
        /// Node slots per half of the pose buffer; previous poses follow.
        pub node_capacity: u32,
        /// Zero while previous poses are inconsistent (no blending).
        pub blend: u32,
        pub planes: [[f32; 4]; 6],
        pub absolute_view_projection: [[f32; 4]; 4],
        /// Nonzero on temporally antialiased frames with valid previous poses:
        /// visible IDs of moving slots then carry [`MOVED_SLOT`](super::constants::MOVED_SLOT).
        pub motion: u32,
        /// Joints of skinned slots' palettes.
        pub joints: u32,
        /// Candidate flags compaction gathers:
        /// [`FLAG_VISIBLE`](super::constants::FLAG_VISIBLE) without occlusion
        /// culling, [`FLAG_PHASE1`](super::constants::FLAG_PHASE1) or
        /// [`FLAG_PHASE2`](super::constants::FLAG_PHASE2) with it.
        pub mask: u32,
        /// Nonzero when occlusion culling splits drawing into two phases.
        pub occlusion: u32,
        /// The projection the frame rasterizes.
        pub view_projection: [[f32; 4]; 4],
        /// Viewport width and height in pixels, and the depth pyramid's levels.
        pub viewport: [f32; 4],
        /// Skinned slots whose palette bounds `skin_bounds_pass`
        /// computes and culls.
        pub skinned: u32,
        /// Nonzero when candidates covering no pixel center are left undrawn.
        pub subpixel: u32,
        /// Candidates per visible-ID plane: each level of detail's IDs start
        /// this many entries after the last's.
        pub plane: u32,
        /// The screen error in pixels levels of detail keep within; 0 draws
        /// every instance at full detail.
        pub lod_error: f32,
        /// The eye the frame rasterizes from, which levels of detail follow
        /// the distance to; w is 1 where impostor cards may stand in.
        pub eye: [f32; 4],
        /// Fades: the share of a switch between a mesh and its
        /// card this frame fades (1 for previews, which keep no level) and the
        /// share of its distance past a switch the eye goes before a candidate
        /// switches back.
        pub fade: [f32; 4],
    }

    /// One candidate slot, its hierarchy node, its draw group and its index in
    /// draw-group order.
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct CandidateData as ["Candidate"] {
        pub slot: u32,
        pub node: u32,
        pub batch: u32,
        pub position: u32,
    }

    /// One draw group: its candidate range (also its compacted visible-ID region),
    /// mesh index count, shadow draw group, local bounds and
    /// levels of detail. `bounds_min[3]` is 1 for meshes with joints,
    /// whose joints place them: GPU-presented frames bound their
    /// slots by their palettes ([`SkinnedData`]), and they never cast kept
    /// shadows.
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct BatchData as ["Batch" in PRESENTATION_SHADER, "Batch" in SHADOW_SHADER] {
        pub start: u32,
        pub count: u32,
        pub index_count: u32,
        /// The draw group: its index among the groups shadows draw.
        pub group: u32,
        pub bounds_min: [f32; 4],
        pub bounds_max: [f32; 4],
        // Where the batch's mesh lies in the mesh arena: its first index and
        // base vertex, which its draw arguments carry, with its first instance:
        // `start` where devices draw indirectly from nonzero first instances,
        // otherwise 0.
        pub first_index: u32,
        pub base_vertex: i32,
        pub first_instance: u32,
        /// What each coarser level's first instance adds per level:
        /// the visible-ID plane stride where first instances name
        /// regions, otherwise 0.
        pub first_plane: u32,
        /// Levels of detail 1 and 2: each one's first index and
        /// index count in the mesh arena.
        pub lod_ranges: [u32; 4],
        /// Levels 1 and 2's geometric errors in mesh units, the level count
        /// (1 to [`LOD_LEVELS`](super::constants::LOD_LEVELS)), and the impostor
        /// card's error (0: none).
        pub lod_errors: [f32; 4],
    }

    /// A skinned slot of GPU-presented frames: its slot, its
    /// candidate's draw-group position, its draw group and its mesh's joint
    /// count.
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct SkinnedData as ["Skinned"] {
        pub slot: u32,
        pub position: u32,
        pub batch: u32,
        pub joints: u32,
    }
}
