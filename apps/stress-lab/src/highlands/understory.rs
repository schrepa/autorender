//! The highlands' understory and the rock where it gathers, placed over the
//! ground its rules lay: talus on the aprons at the feet of its rock faces,
//! erratics on its ridges and benches, junipers on its dry heath and at its
//! forests' edges, and logs and stumps in its forests, each from the
//! scene's own generators ([`rocks`], [`shrubs`], [`deadfall`]). Where a
//! layer's objects retire, the ground takes on their mean color over the
//! share of it they cover, on the surface materials the layer stands on
//! (the palette's placed objects), so each layer stands on the materials
//! its objects lie on, and its stand-in stays the average of what it
//! replaces.

use engine::prelude::*;
use engine::terrain::*;

use super::{MOSS, boulder_choices, deadfall, noise, remap, rocks, shrubs, trees};

/// Talus block, juniper, log and stump shapes the layers choose from.
const BLOCKS: u32 = 4;
const JUNIPERS: u32 = 3;
const LOGS: u32 = 4;
const STUMPS: u32 = 3;

/// How far up junipers' stems and the deadfall's undersides blend into the
/// ground where the terrain places them, in mesh units (meters at scale
/// 1): the litter about a bush's foot and under a log, a few centimeters.
const LITTER: Grounding = Grounding { height: 0.07 };

/// How far junipers sway: meters at ten meters up in full wind, which bobs
/// their stiff limbs some three centimeters a meter out (`wind.wgsl`'s
/// branches: four times this times the square of a tenth of their reach).
const JUNIPER_SWAY: f32 = 0.8;

/// The understory's asset IDs, in the order [`placed`] lists their
/// placement assets: the talus blocks, the junipers, the logs, then the
/// stumps.
pub(super) fn assets() -> impl Iterator<Item = String> {
    let named = |name: &'static str, count: u32| {
        (0..count).map(move |index| format!("{name}{index}"))
    };
    named("block", BLOCKS)
        .chain(named("juniper", JUNIPERS))
        .chain(named("log", LOGS))
        .chain(named("stump", STUMPS))
}

/// The understory's placement assets, in the order [`assets`] names them:
/// talus blocks in the boulders' `rock`, colliding as spheres a little
/// inside their faces about their middle; junipers in the spruces' needles
/// over `atlas`, swaying a little and colliding where their stems gather;
/// and logs and stumps in the spruces' bark there, moss settling on their
/// tops, each colliding as its own box ([`deadfall`]). Junipers and
/// deadfall blend into the litter at their feet ([`LITTER`]).
pub(super) fn placed(
    world: &mut World,
    atlas: TextureHandle,
    rock: MaterialHandle,
) -> engine::Result<Vec<PlacementAsset>> {
    let needles = trees::material(atlas)?.with_grounding(LITTER)?;
    let bark = Material::new([1.0; 4])?
        .with_surface(Surface {
            roughness: 0.9,
            ..Surface::MATTE
        })?
        .with_base_color_texture(TextureRef::new(atlas))?
        .with_top_layer(MOSS)?
        .with_grounding(LITTER)?;
    let (needles, bark) = {
        let mut assets = world.resource_mut::<AssetLibrary>();
        let needles = assets.insert_material(needles)?;
        (needles, assets.insert_material(bark)?)
    };
    let mut placed = Vec::new();
    for seed in 0..BLOCKS {
        placed.push(PlacementAsset {
            mesh: rocks::block_record(world, seed)?,
            material: rock,
            lods: Vec::new(),
            parts: Vec::new(),
            sway: 0.0,
            collider: Some(PlacedCollider::new(
                PlacedShape::Sphere { radius: 0.28 },
                [0.0, 0.1, 0.0],
            )),
            shadow: None,
        });
    }
    for seed in 0..JUNIPERS {
        let juniper = shrubs::juniper(world, seed, needles)?;
        placed.push(PlacementAsset {
            sway: JUNIPER_SWAY,
            collider: Some(PlacedCollider::new(
                PlacedShape::Cuboid {
                    half_extents: [0.2, 0.2, 0.2],
                },
                [0.0, 0.2, 0.0],
            )),
            ..juniper
        });
    }
    for seed in 0..LOGS {
        placed.push(deadfall::log(world, seed, bark)?);
    }
    for seed in 0..STUMPS {
        placed.push(deadfall::stump(world, seed, bark)?);
    }
    Ok(placed)
}

/// The understory's placement layers, `forest` the mask of where spruces
/// stand: talus, erratics, junipers, logs and stumps.
pub(super) fn layers(forest: &Mask) -> [PlacementLayer; 5] {
    [talus(), erratics(), junipers(forest), logs(forest), stumps()]
}

/// Talus: blocks fallen from the rock faces, most of them fresh and
/// angular, a third older and rounded, gathered on the scree aprons at the
/// faces' feet. Talus rests at the angle of repose of angular rock debris,
/// about 35 degrees, and steeper ground sheds it, so blocks lie on slopes
/// from 25 degrees, where an apron's toe eases out, to 38, none from 41;
/// most in its hollows, where falling blocks come to rest, in clusters
/// where they fell together; none on roads or in streams. They stand on
/// the scree, on the foot of the rock over it, where the steep layer's
/// rock first paints the ground, and on bare ground, not on the meadows,
/// whose far look their stand-in would grey. They are placed on the
/// kilometer tiles, whose four-meter samples read the apron's hollows
/// rather than the ground's meter bumps, and which stay resident some 1.4
/// kilometers out at 1080p; there the largest blocks span a pixel or two
/// and the ground's stand-in takes them on, as it does those retiring by
/// their size, so no far tiles.
fn talus() -> PlacementLayer {
    PlacementLayer {
        id: "talus".into(),
        assets: choices("block", BLOCKS, 1.0)
            .into_iter()
            .chain(boulder_choices().into_iter().map(|choice| PlacementChoice {
                weight: 0.4,
                ..choice
            }))
            .collect(),
        density: 500.0,
        footprint: 2.5,
        materials: vec!["scree".into(), "rock".into(), "dirt".into()],
        mask: Mask::Multiply(vec![
            Mask::Slope([22.0, 27.0]),
            Mask::Invert(Box::new(Mask::Slope([37.0, 41.0]))),
            Mask::Curvature([-0.04, 0.08]),
            remap(noise(30.0, 81, 2), 0.35, 0.65),
            Mask::Invert(Box::new(Mask::Path([6.0, 2.0]))),
            Mask::Invert(Box::new(Mask::River([1.0, -1.0]))),
        ]),
        scale: [0.45, 2.2],
        // Resting on the apron's slope, settled into it.
        align: 0.8,
        downhill: 0.0,
        sink: 0.0,
        bed: 0.2,
        level: 2,
        far: None,
    }
}

/// Erratics: boulders the glaciers carried, rounding them, and left where
/// their ice melted, standing singly or a few together on ridges and on
/// the benches of valley sides: convex ground no steeper than a ridge's
/// crest, and gentle ground that water drains off rather than gathers on,
/// so not the valley floors; never on roads, in streams or lakes, nor
/// under snow. They are placed on the two-kilometer tiles, whose
/// eight-meter samples read the ridges and benches tens of meters across
/// rather than the ground's bumps, and which stay resident some 2.8
/// kilometers out at 1080p; there the largest span a few pixels and the
/// ground's stand-in takes them on, so no far tiles.
fn erratics() -> PlacementLayer {
    PlacementLayer {
        id: "erratics".into(),
        assets: boulder_choices(),
        density: 6.0,
        footprint: 8.0,
        materials: vec![
            "grass".into(),
            "dry_grass".into(),
            "dirt".into(),
            "forest_floor".into(),
            "rock".into(),
        ],
        mask: Mask::Multiply(vec![
            Mask::Max(vec![
                Mask::Multiply(vec![
                    Mask::Curvature([-0.01, -0.05]),
                    Mask::Invert(Box::new(Mask::Slope([24.0, 32.0]))),
                ]),
                Mask::Multiply(vec![
                    Mask::Invert(Box::new(Mask::Slope([7.0, 13.0]))),
                    Mask::Invert(Box::new(Mask::Flow([3.5, 4.5]))),
                ]),
            ]),
            remap(noise(60.0, 85, 2), 0.4, 0.7),
            Mask::Invert(Box::new(Mask::Path([16.0, 6.0]))),
            Mask::Invert(Box::new(Mask::River([3.0, 0.0]))),
            Mask::Invert(Box::new(Mask::Lake([0.1, 0.6]))),
        ]),
        scale: [1.4, 4.0],
        align: 0.2,
        downhill: 0.0,
        sink: 0.0,
        bed: 0.45,
        level: 3,
        far: None,
    }
}

/// Junipers: common juniper, light-hungry and drought-hardy, on dry heath
/// and at the forests' edges, never under a stand's closed canopy, which
/// shades it out: where the forest has begun but not yet closed, or on dry
/// ground anywhere, its wetness index under 7.5 to 9.5, as on steep or
/// convex ground; no steeper than heath holds, in loose groups, as the
/// birds spreading its berries leave it; off roads, streams and lakes. They
/// stand on the dry grass of heath and on the forest floor at the stands'
/// edges. Every one still spans pixels enough to show where the finest
/// tiles end (346 meters at 1080p), so all stand on the kilometer tiles
/// too and retire by their size beyond.
fn junipers(forest: &Mask) -> PlacementLayer {
    PlacementLayer {
        id: "junipers".into(),
        assets: choices("juniper", JUNIPERS, 1.0),
        density: 50.0,
        footprint: 4.0,
        materials: vec!["dry_grass".into(), "forest_floor".into()],
        mask: Mask::Multiply(vec![
            Mask::Invert(Box::new(remap(forest.clone(), 0.35, 0.6))),
            Mask::Max(vec![
                remap(forest.clone(), 0.0, 0.12),
                Mask::Invert(Box::new(Mask::Wetness([7.5, 9.5]))),
            ]),
            Mask::Invert(Box::new(Mask::Slope([28.0, 36.0]))),
            remap(noise(25.0, 91, 2), 0.3, 0.65),
            Mask::Invert(Box::new(Mask::Path([4.0, 1.0]))),
            Mask::Invert(Box::new(Mask::River([2.0, 0.0]))),
            Mask::Invert(Box::new(Mask::Lake([0.1, 0.5]))),
        ]),
        scale: [0.55, 1.3],
        align: 0.25,
        downhill: 0.0,
        sink: 0.05,
        bed: 0.3,
        level: 0,
        far: Some(FarPlacement {
            level: 2,
            share: 1.0,
        }),
    }
}

/// Logs: spruces' deadfall where storms and age felled the forest's trees,
/// most where its edges stand in the wind and half as many inside its
/// stands, gathered in the gaps windthrow opens ([`windthrow`]): about ten
/// cubic meters a hectare inside the stands, between a managed forest's few
/// and an old-growth spruce forest's sixty to ninety (Siitonen 2001). A
/// tree falls away from its stump down the slope, so each log turns its
/// length downhill, the more so the steeper the ground, and lies along the
/// ground under it, a little sunk into the litter; none on roads or in
/// streams. They lie on the forest floor alone. Where the finest tiles end
/// a log spans a pixel across, so no far tiles.
fn logs(forest: &Mask) -> PlacementLayer {
    PlacementLayer {
        id: "deadfall".into(),
        assets: choices("log", LOGS, 1.0),
        density: 120.0,
        footprint: 6.0,
        materials: vec!["forest_floor".into()],
        mask: Mask::Multiply(vec![
            Mask::Max(vec![
                Mask::Multiply(vec![
                    remap(forest.clone(), 0.12, 0.25),
                    // Falling as the canopy closes.
                    remap(forest.clone(), 0.7, 0.45),
                ]),
                Mask::Constant(0.5),
            ]),
            windthrow(),
            Mask::Invert(Box::new(Mask::Path([4.0, 1.0]))),
            Mask::Invert(Box::new(Mask::River([1.0, -0.5]))),
        ]),
        scale: [0.75, 1.25],
        align: 1.0,
        // A slope of 1 in 2 turns every log within thirty degrees of
        // downhill.
        downhill: 4.0,
        sink: 0.05,
        bed: 0.0,
        level: 0,
        far: None,
    }
}

/// Stumps: the feet of spruces that snapped in storms or were felled, in
/// the windthrow gaps the logs lie in ([`windthrow`]), some twenty a
/// hectare there, standing upright and resting on the lowest ground about
/// them; off roads. They stand on the forest floor alone. Where the finest
/// tiles end a stump spans about a pixel, so no far tiles.
fn stumps() -> PlacementLayer {
    PlacementLayer {
        id: "stumps".into(),
        assets: choices("stump", STUMPS, 1.0),
        density: 40.0,
        footprint: 4.0,
        materials: vec!["forest_floor".into()],
        mask: Mask::Multiply(vec![
            windthrow(),
            Mask::Invert(Box::new(Mask::Path([4.0, 1.0]))),
        ]),
        scale: [0.8, 1.2],
        align: 0.1,
        downhill: 0.0,
        sink: 0.0,
        bed: 0.3,
        level: 0,
        far: None,
    }
}

/// The gaps windthrow opens in the stands, tens of meters across, where
/// storms felled trees together: the logs' and the stumps' alike.
fn windthrow() -> Mask {
    remap(noise(50.0, 95, 2), 0.3, 0.7)
}

/// Each of the `count` shapes of asset `name` as a layer's choice, each
/// weighing `weight`.
fn choices(name: &str, count: u32, weight: f32) -> Vec<PlacementChoice> {
    (0..count)
        .map(|index| PlacementChoice {
            asset: format!("{name}{index}"),
            weight,
        })
        .collect()
}
