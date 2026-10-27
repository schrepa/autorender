//! Upload structs of `crate::gpu_data`.

use bytemuck::{Pod, Zeroable};

use super::constants::{SHADOW_REGIONS, SKY_LAYERS};

/// Most levels of the shadow clipmap.
pub(crate) const MAX_LEVELS: usize = engine_core::MAX_SHADOW_LEVELS as usize;
/// Most tiles moving casters draw over in a frame.
pub(crate) const MAX_MOVING_TILES: usize = engine_core::MAX_MOVING_TILES as usize;
/// Most occupancy maps, whose centers the lighting uniform holds two a
/// vector.
pub(crate) const MAX_OCCUPANCY_LEVELS: usize = engine_core::MAX_OCCUPANCY_LEVELS as usize;

/// A slot pose as WGSL `Rows` reads it: core's three affine rows, with the
/// implicit final row [0, 0, 0, 1]. Frames upload `RenderFrame::pose_rows`
/// as they are.
pub(crate) type PoseData = engine_core::PoseRows;

/// WGSL `grounds` of `meshes/declarations.wgsl`: what lies
/// under a placed object, x and z of the ground's normal as two signed
/// 16-bit fractions, the meters its base sinks below the ground, the meters
/// up it blends into the ground (0: none) and the ground's color there as
/// four bytes, which `scatter_snap.wgsl` writes (0 until it does).
pub(crate) type GroundData = [u32; 4];

gpu_structs! {
    /// The camera as shaders read it. `view_projection` is the rasterized
    /// projection.
    /// Temporally antialiased frames jitter it by `jitter.xy` clip
    /// units per w, and their motion vectors use the previous frame's unjittered
    /// `previous_view_projection` and `jitter.zw`, two over the render size in
    /// pixels. Other frames repeat the view-projection and keep the jitter zero.
    /// The unjittered `inverse_view_projection` turns pixels into sky rays and
    /// `position` (xyz) is the eye, for view-dependent lighting;
    /// its w is presentation time in seconds, for animated surfaces.
    /// `options.x` is 1 while a reflection probe captures a cube face;
    /// `options.y` is the angle
    /// a display pixel spans at the view's center ([`pixel_angle`](crate::gpu_data::pixel_angle));
    /// `options.z` turns fades' dither each temporally antialiased frame;
    /// `options.w` is the mip bias material textures sample with
    /// ([`FrameRegion::mip_bias`](crate::targets::FrameRegion::mip_bias)).
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
        /// The frame's region: its render size in pixels, which it draws at
        /// from the top-left corner of its screen-sized targets, and that
        /// size's share of the display on each axis
        /// ([`FrameRegion::uniform`](crate::targets::FrameRegion::uniform)).
        pub region: [f32; 4],
        /// The detail eye every handover measures from (xyz, world space as
        /// `position`; w unused)
        /// ([`set_detail`](crate::gpu_data::CameraUniform::set_detail)).
        pub detail: [f32; 4],
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
        /// 1 with the physically based sky; 1 when 8-bit color writes dither.
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
        /// height falloff and the sun's glow; the air's color, and 1 with
        /// haze, when fog and the air take the sky's light (w).
        pub aerial: [f32; 4],
        pub aerial_color: [f32; 4],
        /// Occupancy: one over the finest map's side in meters, its
        /// texels a side, the meters around the detail eye within which it
        /// holds every point (each coarser map twice as far) and the strength
        /// covers darken by (0 without maps); the maps' count and the
        /// reference height their heights are above (zw unused); then each
        /// map's window's center (x and z), two maps a vector.
        pub occupancy: [f32; 4],
        pub occupancy_maps: [f32; 4],
        pub occupancy_centers: [[f32; 4]; MAX_OCCUPANCY_LEVELS / 2],
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
        /// The night sky: the stars' brightness (0 without them), the
        /// moon's (0 without one), how much of the sky cirrus covers, and
        /// the share of the moon's face the sun lights.
        pub night: [f32; 4],
        /// The unit quaternion turning world directions into the galaxy's
        /// frame: x toward its centre, z toward its north pole.
        pub star_turn: [f32; 4],
        /// The direction toward the moon, and its size as a multiple of its
        /// usual angular radius.
        pub moon: [f32; 4],
        /// Foliage: its transmission, its plants' variation, 1 where
        /// masked leaves dissolve with hashed cutoffs this frame (a
        /// temporally antialiased one), and the meters its bases blend into
        /// the ground over; all 0 off.
        pub foliage: [f32; 4],
        /// Highlights sized by their lights: the cosine and
        /// sine of the sun's angular radius (1 and 0 for a point sun, both 0
        /// while highlights are neither sized nor capped), the exposed
        /// brightness local lights' highlights ease toward (0 caps none),
        /// and the radius of local lights' smallest sphere.
        pub highlights: [f32; 4],
        /// Windows lighting room by room: the sun's elevation,
        /// the elevation rooms switch on around and how far either side
        /// they spread (0 while windows are lit as their textures light
        /// them), all in degrees, and the share of rooms staying dark.
        pub windows: [f32; 4],
        /// The night sky's shapes: cirrus' height above the eye and drift
        /// along x and z each second (world units), and how many stars
        /// show, as a multiple.
        pub night_shape: [f32; 4],
        /// The haze's march: how far along the view it gives the sunlight
        /// the air scatters (0 where the view marches none), the forward
        /// scattering of its phase, how far its density strays from its
        /// mean, and one over the size of its largest patches (meters).
        pub haze_march: [f32; 4],
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
        /// roughness; then the rooms across the base color texture of a
        /// material divided into them (0 for none).
        pub top: [f32; 4],
        pub layer: [f32; 4],
    }

    /// WGSL `Shadow` uniform of `scene.wgsl`: world to light space and the
    /// shadow clipmap's levels, whose tiles lie in the slots of one atlas.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ShadowUniform as ["Shadow"] {
        /// World to light space: the sun's right, up and away-from-sun
        /// axes.
        pub light: [[f32; 4]; 3],
        /// Shadows on (x, 0 off) and the meters into crowns the sun reaches
        /// (y, foliage's crown depth); zw unused.
        pub state: [f32; 4],
        /// Levels in use, the finest level's texel (world units), texels
        /// along a tile's side, and the terrain horizon records' count.
        pub level_state: [f32; 4],
        /// Tiles along a side of each level's window, slots along a row of
        /// the atlas, and one over the atlas's width and height in texels.
        pub atlas: [f32; 4],
        /// Each level's window: light-space x and y of its first tile's
        /// corner, its tiles' width and its texels' (world units).
        pub windows: [[f32; 4]; MAX_LEVELS],
        /// Each level's depth band: its nearest depth from the sun and one
        /// over its depth, the normal offset of its lookups (world units)
        /// and its steepest receiver plane (depth per texel).
        pub bands: [[f32; 4]; MAX_LEVELS],
    }

    /// Caster selection, as `shadows.wgsl` reads it: the regions of tiles
    /// redrawn this frame and every level's window, whose tiles moving
    /// casters draw over. Each draw group's ID region start is in a storage
    /// buffer beside it.
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct ShadowCullUniform as ["Cull" in SHADOW_SHADER] {
        /// World to light space: the sun's right, up and away-from-sun axes.
        pub light: [[f32; 4]; 3],
        /// Each region of tiles this frame redraws: its light-space
        /// rectangle (minimum x and y, maximum x and y).
        pub regions: [[f32; 4]; SHADOW_REGIONS],
        /// Each region's farthest receiver depth from the sun (x), beyond
        /// which casters darken none of its receivers; yzw unused.
        pub region_depths: [[f32; 4]; SHADOW_REGIONS],
        /// Each level's window: light-space x and y of its first tile's
        /// corner and its tiles' width; w unused.
        pub windows: [[f32; 4]; MAX_LEVELS],
        pub candidates: u32,
        pub groups: u32,
        /// Instance IDs of each region's map, and of the sky map.
        pub capacity: u32,
        /// `STILLNESS | SKY_REFRESH` of the shader.
        pub mode: u32,
        /// Regions this frame redraws, levels in use, skinned slots
        /// `select_skinned` visits on GPU-presented frames, and tiles along
        /// a side of a level's window.
        pub counts: [u32; 4],
        /// The words of the counters: those of a mask of every level's
        /// tiles, where the dirty tiles' mask starts after the totals, where
        /// the moving tiles' does, and where the draw groups' starts do.
        pub words: [u32; 4],
        /// The slot of the first moving tile, past the still casters'
        /// slots; yzw unused.
        pub moving: [u32; 4],
        /// The sky map redrawn this frame (`SKY_REFRESH`): world to its
        /// space, its right, up and away-from-the-sky axes; its rectangle
        /// there; and (x) the depth beyond which casters hide no sky from
        /// its receivers.
        pub sky: [[f32; 4]; 3],
        pub sky_rect: [f32; 4],
        pub sky_depths: [f32; 4],
    }

    /// The routes of this frame's moving tiles, as `shadows.wgsl` draws
    /// moving casters through them into the atlas: each tile's world to the
    /// atlas's clip space, mapping its tile onto its slot, and its slot's
    /// texels (first x and y, then past the last); and two over the atlas's
    /// width and height (zw unused).
    #[derive(Clone, Copy, PartialEq, Pod, Zeroable)]
    pub(crate) struct RouteData as ["Routes" in SHADOW_SHADER] {
        pub clips: [[[f32; 4]; 4]; MAX_MOVING_TILES],
        pub slots: [[f32; 4]; MAX_MOVING_TILES],
        pub atlas: [f32; 4],
    }

    /// A tile composed from its region's layer into its slot of the atlas,
    /// an instance of `shadows.wgsl`'s `vs_compose`: the slot's texels
    /// (first x and y, then past the last), and the region's layer with
    /// what to add to an atlas texel of the slot for the region's texel
    /// holding it (w unused).
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ComposeData {
        pub place: [f32; 4],
        pub source: [i32; 4],
    }

    /// WGSL `Post` uniform of `effects.wgsl`: 368
    /// bytes.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct PostUniform as ["Post"] {
        /// Unjittered clip to world space.
        pub inverse_view_projection: [[f32; 4]; 4],
        /// Unjittered world to clip space, for reflection rays.
        pub view_projection: [[f32; 4]; 4],
        /// The eye; w: render pixels per world unit at distance 1.
        pub eye: [f32; 4],
        /// The render size in pixels, and its inverse: the part of the
        /// screen-sized targets the frame draws, from their top-left corner.
        pub size: [f32; 4],
        /// The screen-sized targets' size in pixels, the display's, and its
        /// inverse.
        pub extent: [f32; 4],
        /// Ambient occlusion: its radius in world units, its strength, the
        /// finest occupancy map's texel over the meters around the detail
        /// eye it holds (its share of the render pixels a meter spans a meter
        /// away is where the occlusion fades out, over the octave above) and
        /// 1 when it runs.
        pub occlusion: [f32; 4],
        /// Strength and 1 when bloom runs (zw unused).
        pub bloom: [f32; 4],
        /// Ray distance, strength and 1 when reflections run (w unused).
        pub reflections: [f32; 4],
        /// The haze: the turn of its march's jitter this frame, the pixels
        /// across and down its history holds (0 without one), and 1 when it
        /// runs.
        pub haze: [f32; 4],
        /// Last frame's unjittered world to clip space and its eye (w
        /// unused), which the haze's history reprojects through.
        pub previous_view_projection: [[f32; 4]; 4],
        pub previous_eye: [f32; 4],
        /// The unit direction toward the sun; w: 1 when 8-bit color writes
        /// dither.
        pub sun_direction: [f32; 4],
        /// Sky occlusion: the share of hidden sky's light taken away, the
        /// least share any point keeps, and (w) 1 when it runs.
        pub sky: [f32; 4],
        /// The detail eye ambient occlusion hands over by its distance from
        /// (xyz, world space as `eye`); w unused.
        pub detail: [f32; 4],
    }

    /// Sky occlusion's maps as the screen effects' occlusion pass reads them
    /// (`post.wgsl`): one per direction near the camera, then one per
    /// direction reaching farther.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SkyUniform as ["Sky" in POST_SHADER] {
        /// World to each map's clip space.
        pub clips: [[[f32; 4]; 4]; SKY_LAYERS],
        /// Each map's unit direction toward the sky, and (w) how much it
        /// counts: 0 until drawn, fading in to 1 over its first frames.
        pub directions: [[f32; 4]; SKY_LAYERS],
        /// Directions in use, the near and the far maps' texels in world
        /// units, and their reach (half a side) over their distances.
        pub counts: [f32; 4],
    }

    /// Temporal antialiasing's resolve, as `temporal.wgsl` reads it.
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct ResolveUniform as ["Resolve"] {
        /// Size of one display pixel in texture coordinates: the canvas's and
        /// the history's.
        pub texel: [f32; 2],
        /// Weight of this frame's color against the reprojected history.
        pub current_weight: f32,
        /// Zero when the history must not be read (reset).
        pub history_valid: u32,
        /// This frame's turn of the dither (the camera's `options.z`); 1
        /// following the nearest surface's motion; 1 with the rounded clamp;
        /// particles' least weight of this frame's color.
        pub options: [f32; 4],
        /// The render size this frame drew its targets at, in pixels, and
        /// how far its samples moved off their pixels' centers, in render
        /// pixels.
        pub render: [f32; 4],
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
        /// The detail eye, which levels of detail follow the distance to;
        /// w is 1 where impostor cards may stand in.
        pub eye: [f32; 4],
        /// Fades: the share of a switch between a mesh and its
        /// card this frame fades (1 for previews, which keep no level) and the
        /// share of its distance past a switch the detail eye goes before a
        /// candidate switches back.
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
