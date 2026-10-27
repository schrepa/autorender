//! Fades by size on screen, which `cargo xtask gpu` refuses: a shader
//! decides how much of a pattern shows (a `smoothstep`, `step` or
//! `saturate`, a `select` or `mix` weight, an `if`, a loop's bound or a
//! comparison turned into a number) by a size on screen (what the detail
//! eye's measures give, `scene.wgsl`'s `detail_distance`, `detail_across`
//! and `detail_span`, and a pixel's derivatives) only to filter it, a
//! pattern drawn wherever its features span about two pixels and finer
//! ones averaged (a level of its mipmaps or of a pyramid, its mean, its
//! variance into roughness), or to hand over between two representations
//! with the same mean (an impostor and its mesh, blades and their canopy,
//! occlusion's levels); never to fade it toward nothing. Every such
//! decision lies in a function [`FILTERS`] lists, each with what stands in
//! for the detail it decides on, or on a line marked
//! `// filter: <what stands in>`; any other fails the build. Sizes are
//! followed as the handover check follows the camera (`handovers::sweep`):
//! through expressions, locals, arguments, results and calls; a direction
//! (`normalize`'s) holds none, and what a texture gives is its texels as
//! filtering averages them.

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
        "terrain_octave",
        "noise computed where it shows, whole while its features' wavelength spans four of the \
         pixels the footprint resolves, its mean by two, the Nyquist limit mipmaps keep",
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
        "foam against what stands behind the pixel, a band narrower than a pixel spreading \
         its share over it",
    ),
    (
        "water_drawn",
        "the tiles' level whose spacing matches the pixel on the water's plane",
    ),
    ("water_edge", "a waterline as sharp as a pixel resolves it"),
    (
        "water_ends",
        "where the samples holding the water's surface end, as sharp as a pixel resolves it",
    ),
    (
        "water_ramp",
        "a fade over the water's depth averaged over the depths the pixel spans",
    ),
    (
        "water_shown",
        "a computed pattern's part drawn while it spans two pixels along the way it varies, as \
         mipmaps keep a texture's, the finer ones' slopes as roughness",
    ),
    (
        "water_turning",
        "the water's normal turning within the pixel, its spread as roughness (Kaplanyan)",
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
        "terrain_displacement",
        "the ground's mesh band: its materials' heights at the mip level whose features the mesh carries there, the pixel band holding the rest; none where their whole depth moves the surface less than a pixel, the smooth surface standing in",
    ),
    (
        "terrain_parallax",
        "the ground's pixel band and tracks: the view ray meets its materials' whole heights from the drawn ground where their whole depth moves it a pixel or more, the drawn ground standing in below",
    ),
    (
        "terrain_resolved",
        "the depth edges of what hides what, the smooth surface's derivatives standing in across them",
    ),
    (
        "terrain_seen",
        "where the view ray meets the ground: the mesh band stepped back, the pixel band moved where it moves a pixel, and the shading levels the pixel's footprint picks",
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
