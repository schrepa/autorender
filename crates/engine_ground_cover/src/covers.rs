use std::collections::BTreeMap;
use std::f32::consts::{FRAC_1_PI, PI};

use bevy_ecs::prelude::*;
use engine_core::math::{max, min};
use engine_open_world_kit::ground::{GroundTiles, PALETTE_ENTRIES, TerrainEntry, TerrainTiles};

use crate::layouts::{
    COVER_HEADER_ROWS, COVER_MATERIAL_ROWS, CardRows, CoverCell, CoverHeader, CoverMaterial,
    CoverTerrain, LightRows, SEEN_ABOVE_HIGH, SEEN_ABOVE_LOW, SEEN_ABOVE_MIDDLE, SeenRows,
    ShapeRows,
};
use crate::settings::{
    CoverAtlas, CoverAtlasCell, CoverLook, CoverPlants, CoverShape, GroundCover,
};

const SEEN_STEPS: u32 = 256;
const LEAN_STEPS: u32 = 8;
const SEEN_ABOVE: [f32; 3] = [SEEN_ABOVE_LOW, SEEN_ABOVE_MIDDLE, SEEN_ABOVE_HIGH];

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Least {
    pub(crate) width: f32,
    pub(crate) height: f32,
    pub(crate) pixels: f32,
}

impl Default for Least {
    fn default() -> Self {
        Self {
            width: f32::MAX,
            height: f32::MAX,
            pixels: f32::MAX,
        }
    }
}

impl Least {
    fn of(dims: Dims, grown: Grown) -> Self {
        Self {
            width: dims.width * grown.fades,
            height: dims.mean * grown.fades,
            pixels: dims.smallest,
        }
    }

    fn or(self, other: Self) -> Self {
        Self {
            width: min(self.width, other.width),
            height: min(self.height, other.height),
            pixels: min(self.pixels, other.pixels),
        }
    }

    pub(crate) fn row(self, lushest: f32) -> [f32; 3] {
        [self.width / lushest, self.height / lushest, self.pixels]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct SlotTerms {
    pub(crate) blades: f32,
    pub(crate) plants: f32,
    pub(crate) cards: f32,
    pub(crate) widest: f32,
    pub(crate) tallest: f32,
    pub(crate) span: f32,
    pub(crate) stand: f32,
    pub(crate) least: Least,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub(crate) struct CoverTable {
    pub(crate) terrains: Vec<CoverTerrain>,
    pub(crate) slots: Vec<Vec<SlotTerms>>,
    pub(crate) materials: Vec<CoverMaterial>,
    pub(crate) kinds: Vec<[u32; 4]>,
    pub(crate) header: CoverHeader,
    pub(crate) shape: ShapeRows,
    pub(crate) light: LightRows,
    pub(crate) seen: SeenRows,
    pub(crate) cards: CardRows,
}

#[derive(Clone, Copy)]
struct Dims {
    width: f32,
    mean: f32,
    span: f32,
    smallest: f32,
    stand: f32,
    cards: f32,
}

impl Dims {
    fn leaf(self) -> f32 {
        2.0 * self.mean * self.width
    }
}

#[derive(Clone, Copy)]
struct Grown {
    dense: f32,
    grows: f32,
    fades: f32,
}

impl Grown {
    fn of(look: &CoverLook) -> Self {
        Self {
            dense: look.density * (1.0 + look.patch.density),
            grows: 1.0 + look.patch.height,
            fades: 1.0 - look.patch.height,
        }
    }
}

#[derive(Clone, Copy)]
struct Far {
    lowest: f32,
    leaves: f32,
}

#[derive(Clone, Copy, Default)]
struct Counted {
    blades: f32,
    plants: f32,
    cards: f32,
    widest: f32,
    tallest: f32,
    span: f32,
    stand: f32,
    least: Least,
}

impl Counted {
    fn settle(&mut self, far: &mut Far, (dims, grown): (Dims, Grown), share: f32) {
        let count = grown.dense * share;
        if dims.cards > 0.0 {
            self.plants += count;
            self.cards += count * dims.cards;
        } else {
            self.blades += count;
        }
        self.seen((dims, grown));
        far.lowest = min(far.lowest, dims.mean * grown.fades / dims.smallest);
        far.leaves = max(
            far.leaves,
            grown.dense * dims.width * dims.mean * grown.grows * grown.grows,
        );
        self.span = max(self.span, dims.span * grown.grows);
    }

    fn seen(&mut self, (dims, grown): (Dims, Grown)) {
        self.widest = max(self.widest, dims.width * grown.grows / dims.smallest);
        self.tallest = max(self.tallest, dims.mean * grown.grows / dims.smallest);
        self.stand = max(self.stand, dims.stand);
        self.least = self.least.or(Least::of(dims, grown));
    }

    fn join(self, slot: &mut SlotTerms, counted: bool) {
        if counted {
            slot.blades = max(slot.blades, self.blades);
            slot.plants = max(slot.plants, self.plants);
            slot.cards = max(slot.cards, self.cards);
        }
        slot.widest = max(slot.widest, self.widest);
        slot.tallest = max(slot.tallest, self.tallest);
        slot.span = max(slot.span, self.span);
        slot.stand = max(slot.stand, self.stand);
        slot.least = slot.least.or(self.least);
    }

    fn limits(self, lushest: f32) -> [f32; 4] {
        let [width, height, pixels] = self.least.row(lushest);
        [self.tallest * lushest, width, height, pixels]
    }
}

fn cell_rows(atlas: &CoverAtlas, cell: &CoverAtlasCell) -> CoverCell {
    let [across, down] = atlas.size.map(|texels| texels as f32);
    let [x, y] = cell.root.map(|texel| texel as f32);
    let left = x - cell.x as f32;
    let right = (cell.x + cell.w) as f32 - x;
    let above = y - cell.y as f32;
    let below = (cell.y + cell.h) as f32 - y;
    let grows = [
        (below, [0.0, -above], [left, right]),
        (left, [right, 0.0], [above, below]),
        (right, [-left, 0.0], [below, above]),
        (above, [0.0, below], [right, left]),
    ];
    let (back, axis, sides) = grows
        .into_iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap_or(grows[0]);
    let long = max(axis[0].abs() + axis[1].abs(), 1.0);
    CoverCell {
        axis: [x / across, y / down, axis[0] / across, axis[1] / down],
        span: [sides[0] / long, sides[1] / long, back / long, across / down],
    }
}

struct Tail<'a> {
    cover: &'a GroundCover,
    first: u32,
    rows: Vec<[u32; 4]>,
    cells: BTreeMap<&'a str, ([u32; 4], f32)>,
    kinds: BTreeMap<&'a str, [u32; 4]>,
}

impl<'a> Tail<'a> {
    fn at(&self) -> u32 {
        self.first + self.rows.len() as u32
    }

    fn cells(&mut self, name: &'a str, look: &CoverLook) -> ([u32; 4], f32) {
        if let Some(&found) = self.cells.get(name) {
            return found;
        }
        let made = self.cells_made(look).unwrap_or(([0; 4], 0.0));
        self.cells.insert(name, made);
        made
    }

    fn cells_made(&mut self, look: &CoverLook) -> Option<([u32; 4], f32)> {
        let plants = look.plants.as_ref()?;
        let cover = self.cover;
        let (bin, atlas) = cover
            .atlases
            .iter()
            .enumerate()
            .find_map(|(bin, (name, atlas))| (*name == plants.atlas).then_some((bin, atlas)))?;
        let start = self.at();
        let mut spans = 0.0;
        let mut count = 0_u32;
        for name in &plants.cells {
            let Some(cell) = atlas.cells.iter().find(|cell| cell.name == *name) else {
                continue;
            };
            let made = cell_rows(atlas, cell);
            self.rows.push(made.axis.map(f32::to_bits));
            self.rows.push(made.span.map(f32::to_bits));
            spans += made.span[0] + made.span[1];
            count += 1;
        }
        let fan = plants.count[0] | (plants.count[1] << 16);
        (count > 0).then(|| ([start, count, bin as u32, fan], spans / count as f32))
    }

    fn dims(&mut self, name: &'a str, look: &CoverLook) -> Dims {
        let (cells, aspect) = self.cells(name, look);
        let mean = mean_height(look);
        let plants = look.plants.as_ref().filter(|_| cells[1] > 0);
        let width = plants.map_or(look.width, |_| mean * aspect);
        let reach = plants.map_or(1.0, |_| 2.0 + aspect);
        let lift = plants.map_or(0.0, |plants| plants.lift[1]);
        Dims {
            width,
            mean,
            span: look.height[1] * (1.0 + look.clump.height) * reach
                + look.tuft.size * look.tuft.gather[1]
                + lift,
            smallest: look.smallest,
            stand: look.tuft.size,
            cards: plants.map_or(0.0, |plants| plants.count[1] as f32),
        }
    }

    fn material(&mut self, name: &'a str, look: &CoverLook) -> CoverMaterial {
        let (cells, _) = self.cells(name, look);
        let dims = self.dims(name, look);
        let mut own = Counted::default();
        own.seen((dims, Grown::of(look)));
        let clump = &look.clump;
        let patch = &look.patch;
        let tuft = &look.tuft;
        let blade = &look.blade;
        let (bend, stand, arc) = look.plants.as_ref().map_or(
            (0.0, [1.0, 1.0, 0.0, 0.0], 1.0),
            |plants| {
                (
                    plants.bend,
                    [
                        plants.rise[0] / 90.0,
                        plants.rise[1] / 90.0,
                        plants.lift[0],
                        plants.lift[1],
                    ],
                    plants.arc / 360.0,
                )
            },
        );
        let shade = [0, 1, 2].map(|at| 0.5 * (clump.short[at] + clump.tall[at]));
        let drift = [0, 1, 2].map(|at| shade[at] - clump.short[at]);
        let (lean, reach) = match look.plants.as_ref().filter(|_| cells[1] > 0) {
            Some(plants) => risen_lean(plants),
            None => blade_lean(&self.cover.shape, look),
        };
        CoverMaterial {
            shape: [look.density, look.height[0], look.height[1], dims.width],
            tint: [look.tint[0], look.tint[1], look.tint[2], look.sway],
            clump: [clump.size, clump.height, clump.lean, clump.fan],
            shade: [shade[0], shade[1], shade[2], 0.0],
            patches: [patch.size, patch.density, patch.height, dims.mean * reach],
            extra: [0.0, look.spread, 0.0, 0.0],
            accents: [0; 4],
            tuft: [tuft.size, tuft.gather[0], tuft.gather[1], look.smallest],
            fans: [tuft.fan[0], tuft.fan[1], arc, Grown::of(look).dense],
            form: [blade.blunt, blade.bent, blade.fold, blade.widths],
            drift: [drift[0], drift[1], drift[2], bend],
            stand,
            cells,
            lean,
            limits: own.limits(self.cover.lush.sqrt()),
        }
    }

    fn kinds(&mut self, name: &'a str, look: &CoverLook) -> [u32; 4] {
        if let Some(&found) = self.kinds.get(name) {
            return found;
        }
        let cover = self.cover;
        let looks = &cover.looks;
        let mut accents = Vec::with_capacity(look.accents.len());
        for accent in &look.accents {
            if let Some((name, kind)) = looks.get_key_value(&accent.look) {
                let mut made = self.material(name, kind);
                made.extra[2] = accent.share;
                accents.push(made);
            }
        }
        let mut beside = Vec::with_capacity(look.beside.len());
        for layer in &look.beside {
            if let Some((name, kind)) = looks.get_key_value(layer) {
                beside.push(self.material(name, kind));
            }
        }
        let accents_at = self.at();
        self.rows.extend(accents.iter().flat_map(rows));
        let beside_at = self.at();
        self.rows.extend(beside.iter().flat_map(rows));
        let made = [
            accents_at,
            accents.len() as u32,
            beside_at,
            beside.len() as u32,
        ];
        self.kinds.insert(name, made);
        made
    }

    fn grown(
        &mut self,
        (name, look): (&'a str, &CoverLook),
        counted: bool,
        slots: &mut Vec<SlotTerms>,
    ) -> CoverMaterial {
        let cover = self.cover;
        let mut made = self.material(name, look);
        made.accents = self.kinds(name, look);
        let grown = Grown::of(look);
        let mut far = Far {
            lowest: f32::MAX,
            leaves: 0.0,
        };
        let mut own = Counted::default();
        let mut rest = 1.0;
        let mut leaf = 0.0;
        for accent in &look.accents {
            let Some((name, kind)) = cover.looks.get_key_value(&accent.look) else {
                continue;
            };
            rest -= accent.share;
            let dims = self.dims(name, kind);
            leaf += accent.share * dims.leaf();
            own.settle(&mut far, (dims, grown), accent.share);
        }
        let dims = self.dims(name, look);
        leaf += max(rest, 0.0) * dims.leaf();
        own.settle(&mut far, (dims, grown), max(rest, 0.0));
        own.join(&mut slots[0], counted);
        let mut layer = 0;
        for beside in &look.beside {
            let Some((name, kind)) = cover.looks.get_key_value(beside) else {
                continue;
            };
            layer += 1;
            if slots.len() <= layer {
                slots.push(SlotTerms::default());
            }
            let dims = self.dims(name, kind);
            if look.density > 0.0 {
                leaf += kind.density / look.density * dims.leaf();
            }
            let mut other = Counted::default();
            other.settle(&mut far, (dims, Grown::of(kind)), 1.0);
            other.join(&mut slots[layer], counted);
        }
        made.shade[3] = far.lowest;
        made.extra[0] = leaf;
        made.extra[3] = far.leaves;
        made.limits = own.limits(cover.lush.sqrt());
        made
    }
}

fn rows(material: &CoverMaterial) -> [[u32; 4]; COVER_MATERIAL_ROWS as usize] {
    [
        material.shape.map(f32::to_bits),
        material.tint.map(f32::to_bits),
        material.clump.map(f32::to_bits),
        material.shade.map(f32::to_bits),
        material.patches.map(f32::to_bits),
        material.extra.map(f32::to_bits),
        material.accents,
        material.tuft.map(f32::to_bits),
        material.fans.map(f32::to_bits),
        material.form.map(f32::to_bits),
        material.drift.map(f32::to_bits),
        material.stand.map(f32::to_bits),
        material.cells,
        material.lean.map(f32::to_bits),
        material.limits.map(f32::to_bits),
    ]
}

impl CoverTable {
    pub(crate) fn slots(&self, index: usize) -> &[SlotTerms] {
        self.slots.get(index).map_or(&[], Vec::as_slice)
    }

    pub(crate) fn base(&self, index: usize) -> u32 {
        self.terrains.get(index).map_or(0, |terrain| terrain.base)
    }

    fn fill(
        &mut self,
        cover: &GroundCover,
        (terrains, palette): (&[TerrainEntry], &[(Entity, Option<u32>)]),
    ) {
        self.terrains.clear();
        self.slots.clear();
        self.materials.clear();
        let mut tail = Tail {
            cover,
            first: COVER_MATERIAL_ROWS * PALETTE_ENTRIES
                + COVER_HEADER_ROWS
                + terrains.len() as u32,
            rows: std::mem::take(&mut self.kinds),
            cells: BTreeMap::new(),
            kinds: BTreeMap::new(),
        };
        tail.rows.clear();
        for terrain in terrains {
            let base = palette
                .iter()
                .find(|(entity, _)| *entity == terrain.terrain)
                .and_then(|&(_, base)| base);
            let mut row = CoverTerrain {
                base: base.unwrap_or(0),
                ..CoverTerrain::default()
            };
            let mut slots = vec![SlotTerms::default()];
            if let Some(base) = base {
                let end = base as usize + terrain.materials.len();
                if self.materials.len() < end {
                    self.materials.resize(end, CoverMaterial::default());
                }
                let counted = !terrain.surroundings;
                for (at, material) in terrain.materials.iter().enumerate() {
                    self.materials[base as usize + at] = cover
                        .looks
                        .get_key_value(&*material.name)
                        .map_or_else(CoverMaterial::default, |(name, look)| {
                            tail.grown((name, look), counted, &mut slots)
                        });
                }
                for slot in &slots {
                    row.densest = max(row.densest, slot.blades + slot.plants);
                    row.widest = max(row.widest, slot.widest);
                    row.tallest = max(row.tallest, slot.tallest);
                }
            }
            self.terrains.push(row);
            self.slots.push(slots);
        }
        self.kinds = tail.rows;
        self.set(cover);
    }

    fn set(&mut self, cover: &GroundCover) {
        let shape = &cover.shape;
        let light = &cover.light;
        let cards = &cover.cards;
        self.header = CoverHeader {
            lush: [cover.lush_tint[0], cover.lush_tint[1], cover.lush_tint[2], cover.lush],
            shading: [cover.density, cover.up[1], light.sheen, cover.roughness],
            layers: [cover.layers, cover.grow_in, cover.up[0], 0.0],
        };
        self.shape = ShapeRows {
            bends: [shape.most_bend, shape.bent_more, shape.bent_most, shape.bent_knee],
            arc: [shape.knee, shape.drop, shape.reach, shape.taper],
            tips: [shape.blunt_width, shape.blunt_round, shape.curl[0], shape.curl[1]],
            steps: [
                shape.straight_pixels,
                shape.flat_pixels,
                shape.tip_color,
                shape.risen,
            ],
        };
        let wide = |along: f32| 1.0 - along.powf(shape.taper);
        self.light = LightRows {
            glow: [
                light.root_light,
                light.tip_light,
                light.stand_fade[0],
                light.stand_fade[1],
            ],
        };
        self.seen = SeenRows {
            along: seen_bands(wide, |along| along),
            tip: seen_bands(wide, |along| eased(along / shape.tip_color)),
        };
        self.cards = CardRows {
            form: [cards.turn, cards.round, cards.root_light, 0.0],
        };
    }
}

pub(crate) fn write_covers(
    cover: Res<GroundCover>,
    (tiles, ground): (Res<TerrainTiles>, Res<GroundTiles>),
    mut table: ResMut<CoverTable>,
    mut spare: Local<CoverTable>,
) {
    if !(cover.is_changed() || tiles.is_changed() || ground.is_changed()) {
        return;
    }
    spare.fill(&cover, (&tiles.terrains, &ground.palette));
    if *spare != *table {
        std::mem::swap(&mut *table, &mut *spare);
    }
}

fn eased(risen: f32) -> f32 {
    let risen = min(risen, 1.0);
    risen * risen * (3.0 - 2.0 * risen)
}

fn seen_bands(wide: impl Fn(f32) -> f32, value: impl Fn(f32) -> f32) -> [f32; 4] {
    let along = |step: u32| (step as f32 + 0.5) / SEEN_STEPS as f32;
    let whole: f32 = (0..SEEN_STEPS).map(|step| wide(along(step))).sum();
    let (mut sums, mut areas) = ([0.0_f32; 4], [0.0_f32; 4]);
    let mut above = 0.0;
    for step in (0..SEEN_STEPS).rev() {
        let width = wide(along(step));
        let share = (above + 0.5 * width) / whole;
        let band = SEEN_ABOVE.iter().filter(|&&edge| share < edge).count();
        sums[band] += width * value(along(step));
        areas[band] += width;
        above += width;
    }
    let mut held = value(1.0);
    let mut bands = [0.0; 4];
    for band in (0..bands.len()).rev() {
        if areas[band] > 0.0 {
            held = sums[band] / areas[band];
        }
        bands[band] = held;
    }
    bands
}

struct Bend {
    share: f32,
    bend: f32,
    forward: f32,
    knee: f32,
}

fn own_lean(look: &CoverLook) -> f32 {
    let clump = &look.clump;
    let tuft = &look.tuft;
    let mut fanned = 0.0;
    if tuft.gather[1] > 0.0 || tuft.fan[1] > 0.0 {
        for step in 0..LEAN_STEPS {
            let share = (step as f32 + 0.5) / LEAN_STEPS as f32;
            let apart = (share * FRAC_1_PI).sqrt();
            let opened = tuft.fan[0] + (tuft.fan[1] - tuft.fan[0]) * min(2.0 * apart, 1.0);
            fanned += opened * opened / LEAN_STEPS as f32;
        }
    }
    (clump.lean * clump.lean / 3.0 + clump.fan * clump.fan * 0.5 * FRAC_1_PI + fanned).sqrt()
}

fn bends(shape: &CoverShape, look: &CoverLook) -> Vec<Bend> {
    let own = own_lean(look);
    let steps = LEAN_STEPS as f32;
    let forms = [
        (1.0 - look.blade.bent, 0.0, shape.most_bend, shape.knee),
        (look.blade.bent, shape.bent_more, shape.bent_most, shape.bent_knee),
    ];
    let mut made = Vec::with_capacity(2 * (LEAN_STEPS * LEAN_STEPS) as usize);
    for (share, more, most, knee) in forms {
        for curl in 0..LEAN_STEPS {
            let curl = shape.curl[0]
                + (shape.curl[1] - shape.curl[0]) * (curl as f32 + 0.5) / steps;
            for turn in 0..LEAN_STEPS {
                let (sine, cosine) = engine_math::sin_cos(PI * (turn as f32 + 0.5) / steps);
                let forward = curl + own * cosine;
                let sideways = own * sine;
                let leaned = (forward * forward + sideways * sideways).sqrt();
                made.push(Bend {
                    share: share / (steps * steps),
                    bend: min(leaned + more, most),
                    forward: if leaned > 1.0e-4 { forward / leaned } else { 1.0 },
                    knee,
                });
            }
        }
    }
    made
}

fn blade_lean(shape: &CoverShape, look: &CoverLook) -> ([f32; 4], f32) {
    let bends = bends(shape, look);
    let wide = |along: f32| 1.0 - along.powf(shape.taper);
    let tilted = |along: f32| {
        bends
            .iter()
            .map(|bent| {
                let across = along * shape.reach * bent.bend * bent.forward;
                let up = (1.0 - along) * bent.knee
                    + along * (1.0 - shape.drop * bent.bend * bent.bend - bent.knee);
                let run = across * across + up * up;
                if run > 0.0 {
                    bent.share * across * across / run
                } else {
                    0.0
                }
            })
            .sum::<f32>()
    };
    let reach: f32 = bends.iter().map(|bent| bent.share * bent.bend).sum();
    (seen_bands(wide, tilted), shape.reach * reach)
}

fn risen_lean(plants: &CoverPlants) -> ([f32; 4], f32) {
    let (mut flat, mut reach) = (0.0, 0.0);
    for step in 0..LEAN_STEPS {
        let share = (step as f32 + 0.5) / LEAN_STEPS as f32;
        let rise = (plants.rise[0] + (plants.rise[1] - plants.rise[0]) * share).to_radians();
        let (_, cosine) = engine_math::sin_cos(rise);
        flat += cosine * cosine / LEAN_STEPS as f32;
        reach += cosine / LEAN_STEPS as f32;
    }
    ([flat; 4], reach)
}

fn mean_height(look: &CoverLook) -> f32 {
    0.5 * (look.height[0] + look.height[1])
}
