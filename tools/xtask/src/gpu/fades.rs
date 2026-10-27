//! Fades by size on screen, which `cargo xtask gpu` refuses: a shader
//! decides how much of a pattern shows (a `smoothstep`, `step` or
//! `saturate`, a `select` or `mix` weight, an `if`, a loop's bound or a
//! comparison turned into a number) by a size on screen (what the detail
//! eye's measures give, `scene.wgsl`'s `detail_distance`, `detail_across`
//! and `detail_span`, and a pixel's derivatives) only to filter it, a
//! pattern drawn wherever its features span about two pixels, or the
//! steady size temporal antialiasing holds (`scene.wgsl`'s
//! `STEADY_PIXELS`), and finer ones averaged (a level of its mipmaps or of
//! a pyramid, its mean, its variance into roughness), or to hand over
//! between two representations with the same mean (an impostor and its
//! mesh, blades and their canopy, occlusion's levels); never to fade it
//! toward nothing. Every such decision lies in a function [`FILTERS`]
//! lists, each with what stands in for the detail it decides on, or on a
//! line marked `// filter: <what stands in>`; any other fails the build.
//! Sizes are followed as the handover check follows the camera
//! (`handovers::sweep`): through expressions, locals, arguments, results
//! and calls; a direction (`normalize`'s) holds none, and what a texture
//! gives is its texels as filtering averages them.

use super::Checked;
use super::handovers::{Watch, sweep};

/// Filters and matched handovers: functions deciding by sizes on screen
/// how a pattern is averaged, or handing it over to a representation with
/// the same mean, each with what stands in for the detail; and the pixel's
/// own geometry (a direction, a receiver's plane), which measures no
/// detail. They are not followed: what they give, and what they pass the
/// functions they call, depends on no size.
const FILTERS: &[(&str, &str)] = &[
    (
        "bake_mask",
        "a cutout's coverage, kept over its mipmaps' levels as masked fragments keep it",
    ),
    (
        "cover_share",
        "ground cover's blades thinning under a pixel into tufts a pixel wide holding their \
         leaves, and tufts retiring under a pixel, the ground's canopy showing the leaves they \
         leave, their average",
    ),
    (
        "detail_level",
        "a pyramid's level whose texels match the pixel, the next blended in as brilinear \
         filtering blends mipmaps",
    ),
    (
        "fall_cover",
        "a sheet's cover past its threshold, its edge as soft as its pattern changes across \
         the pixel",
    ),
    (
        "map_lookup",
        "the shadow clipmap's level whose texels match the pixel, compared with the receiver's \
         plane across the pixel",
    ),
    (
        "masked_cut",
        "a cutout's coverage, kept over its mipmaps' levels (Golus's mip scale)",
    ),
    (
        "masked_out",
        "a cutout's coverage, kept over its mipmaps' levels (Golus's mip scale)",
    ),
    (
        "occlusion_share",
        "screen-space occlusion handing over to the occupancy maps as a finer level of them \
         would",
    ),
    (
        "ocean_keep",
        "the sea's waves a sample spacing or the steady size resolves, the shorter ones' \
         slopes as roughness",
    ),
    (
        "seen_share",
        "things retiring as they shrink under a pixel on screen, the ground taking on what they \
         leave",
    ),
    ("stable_normalize", "a direction, holding no size"),
    (
        "subdivision_measure",
        "a leaf's meshlet within the pixel error of its surface, its coarser leaf its finer \
         ones' surface",
    ),
    (
        "terrain_filtered",
        "a map's derivatives, the shorter widened to the anisotropy its filtering resolves",
    ),
    (
        "terrain_placed_share",
        "placed objects retired under a pixel on screen, the ground taking on their mean color",
    ),
    (
        "terrain_shading_level",
        "the ground's level whose samples span the pixel error, the next blended in as \
         trilinear filtering turns between mipmaps",
    ),
    (
        "water_caustics",
        "the caustics a pixel's beam of sunlight gathers",
    ),
    (
        "water_contact",
        "foam against what stands behind the pixel, a band narrower than the steady size \
         spreading its share over it",
    ),
    (
        "water_drawn",
        "the tiles' level whose spacing matches the pixel on the water's plane",
    ),
    ("water_edge", "a waterline as sharp as a pixel resolves it"),
    (
        "water_froth",
        "foam's grain as far as the steady size resolves it, the rest averaged into its cover",
    ),
    (
        "water_resolved",
        "standing waves whole where their features span the steady size, the finer ones' \
         slopes as roughness",
    ),
    (
        "water_tile_sample",
        "the waves' tiles through the levels and anisotropy whose texels span the steady \
         size, the slopes they average as variance",
    ),
    (
        "water_turning",
        "the water's normal turning within the pixel, its spread as roughness (Kaplanyan)",
    ),
    (
        "water_wind_waves",
        "the wind's tiles the steady size resolves, the finer ones' slopes as roughness",
    ),
    (
        "widened",
        "normals turning within the pixel, their spread as roughness (Kaplanyan)",
    ),
    (
        "wind_shown",
        "sway moving an object less than a pixel stilled, its mean bend kept",
    ),
    (
        "cs_terrain_vt_feedback",
        "the virtual texture's pages a pixel's footprint asks for: the level whose texels match it",
    ),
    (
        "terrain_displacement",
        "the ground's relief displaced only where it moves the surface a pixel or more, a flat surface standing in below",
    ),
    (
        "terrain_parallax",
        "the ground's relief and tracks moved along the view as far as they move it a pixel or more, the surface standing in below",
    ),
    (
        "terrain_resolved",
        "the depth edges of what hides what, and the virtual texture's pages blended in as the pixel's dither picks them",
    ),
    (
        "terrain_seen",
        "where the view ray meets the ground: relief moved as far as it moves a pixel, and the shading levels the pixel's footprint picks",
    ),
    (
        "terrain_vt_entry",
        "a virtual texture page of the mip whose texels match the pixel",
    ),
    (
        "terrain_vt_page",
        "a virtual texture page of the mip whose texels match the pixel",
    ),
    (
        "terrain_vt_shown",
        "the virtual texture's pages handing over to the ground computed there, baked from the same ground at their texels",
    ),
    (
        "terrain_vt_texels",
        "the virtual texture's pages whose texels match the pixel, handing over to the ground computed there, baked from it",
    ),
];

/// A line marked filter: its decision by a size on screen filters, or
/// hands over to what the mark says stands in for the detail.
const MARK: &str = "// filter:";

/// Every module's fades by size on screen; returns how many filters and
/// handovers decide by sizes: those listed and the lines marked.
pub(super) fn check(modules: &[Checked]) -> Result<usize, String> {
    let listed = |name: &str| FILTERS.iter().any(|&(filter, _)| filter == name);
    let swept = sweep(
        modules,
        Watch::Sizes,
        &listed,
        FILTERS,
        "tools/xtask/src/gpu/fades.rs (FILTERS)",
        MARK,
        "what stands in",
    );
    let mut problems = swept.problems;
    if !swept.found.is_empty() {
        problems.push(format!(
            "{} decisions by a size on screen neither filter nor hand over: filter it: mipmaps, \
             a level, or its variance into roughness; or hand over to a representation with \
             the same mean. Then list the function in tools/xtask/src/gpu/fades.rs (FILTERS) \
             or mark the line `{MARK} <what stands in>`:",
            swept.found.len()
        ));
        problems.extend(swept.found);
    }
    if problems.is_empty() {
        Ok(FILTERS.len() + swept.marks)
    } else {
        Err(problems.join("\n"))
    }
}
