//! Upload structs of `crate::terrain`.

pub(crate) use super::constants::PALETTE_ENTRIES;
use bytemuck::{Pod, Zeroable};

/// Bytes of the palette.
pub(crate) const PALETTE_BYTES: usize = PALETTE_ENTRIES * size_of::<PaletteEntry>();

gpu_structs! {
    /// A surface material as the terrain palette holds it: its linear color
    /// (its base color factor with maps) and roughness; its tiling, height
    /// contrast, projection and maps' layer plus one (0 without maps); its
    /// normal scale, occlusion strength, metalness and mean occlusion; its
    /// maps' mean color and roughness, its factors applied; the meters its
    /// relief spans (none where it displaces the ground), tracks sink it at
    /// most, the mean square slope its normal maps' mean averages away and
    /// the meters it displaces the ground; its cover's canopy: its blades'
    /// linear tip color and their leaves' area per square meter of ground;
    /// its placement layers'
    /// objects, which the ground takes on where they retire: their mean
    /// linear color and the share of the ground they cover where it lies
    /// alone, and the base-2 logarithms of the bounding spheres' diameters
    /// of those ranked first and last, the largest and smallest; and its
    /// height map's mean.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct PaletteEntry as ["TerrainPaletteEntry", "PaletteEntry" in COVER_GROW_SHADER, "PaletteEntry" in BOUNCE_SHADER, "PaletteEntry" in SCATTER_SNAP_SHADER] {
        pub color: [f32; 4],
        pub params: [f32; 4],
        pub maps: [f32; 4],
        pub far: [f32; 4],
        pub relief: [f32; 4],
        pub canopy: [f32; 4],
        pub placed: [f32; 4],
        pub placed_size: [f32; 4],
        pub heights: [f32; 4],
    }

    /// A surface material's cover, at its palette entry's index: blades
    /// per square meter, the shortest and tallest blades and their root
    /// width, then its tip color and sway.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct CoverEntry as ["CoverSpec"] {
        pub shape: [f32; 4],
        pub color: [f32; 4],
    }

    /// A patch of the kept shadow levels as `PatchInput` reads it: its first
    /// sample, its spacing and skirt's depth, its tile's layer and first
    /// texel, and its hole run.
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct PatchRecord {
        pub first: [i32; 2],
        pub shape: [f32; 2],
        pub atlas: [u32; 2],
        pub holes: u32,
    }

    /// WGSL `TerrainView`.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ViewUniform as ["TerrainView"] {
        pub relative_view_projection: [[f32; 4]; 4],
        pub eye_offset: [f32; 4],
        pub inverse_relative_view_projection: [[f32; 4]; 4],
    }

    /// WGSL `TerrainUniform`.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct TerrainUniform as ["TerrainUniform"] {
        pub anchor: [f32; 4],
        pub eye: [f32; 4],
        /// Its subdivision's leaves: where they start in the frame's leaves,
        /// where each layer's word naming its tile's hole run starts in the
        /// hole buffer, and its anchor's cell on the coarsest level's grid
        /// (x and z, signed).
        pub surface: [u32; 4],
        pub origin: [f32; 4],
        pub directory: [u32; 4],
        /// The ground's relief: the meters the deepest of the terrain's
        /// materials with maps sinks it, so pixels where no relief could
        /// move half a pixel look up no materials for it, and the most steps
        /// its rays take; the meters the deepest of them displaces it, so
        /// vertices where none could move a pixel read none, and the meters
        /// between its subdivision's finest leaves' meshlet vertices.
        pub relief: [f32; 4],
        pub deform_grid: [i32; 4],
        pub deform: [f32; 4],
        pub cover: [f32; 4],
        pub macro_map: [u32; 4],
        pub steep: [f32; 4],
        /// The pixels a level's samples span at least where it shades the
        /// ground whole (`TerrainSettings::pixel_error`), as its tiles'
        /// ranges count them; yzw unused.
        pub pixel_error: [f32; 4],
        /// Its tiles' water: where water's directory starts in the
        /// directory buffer, after the directory's own words; yzw unused.
        pub water: [u32; 4],
    }

    /// WGSL `TerrainSubdivision` (`meshes/terrain_subdivision.wgsl`): a
    /// terrain's square and error target; its tree's deepest level, where
    /// its bitfield starts, where its leaves start and how many it holds;
    /// whether the frame merges and its place among the arguments; the
    /// ground it draws and its cutout; and the camera's frustum, in the
    /// world.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SubdivisionUniform as ["TerrainSubdivision"] {
        pub square: [f32; 4],
        pub tree: [u32; 4],
        pub frame: [u32; 4],
        pub extent: [f32; 4],
        pub cutout: [f32; 4],
        pub planes: [[f32; 4]; 6],
    }
}
