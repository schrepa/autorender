//! Tiles' portals of a walk graph: the runs of ground along a tile's
//! edges a gait stands on, and what walking between each two inside the
//! tile costs.

use super::*;

/// A side of a tile: its first or last column of samples (along x), or its
/// first or last row (along z).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    MinX,
    MaxX,
    MinZ,
    MaxZ,
}

tagged!(Side, "invalid tile side" {
    0 => MinX,
    1 => MaxX,
    2 => MinZ,
    3 => MaxZ,
});

impl Side {
    const ALL: [Self; 4] = [Self::MinX, Self::MaxX, Self::MinZ, Self::MaxZ];

    /// Its sample `along` it (0 to 256), within its tile: column and row.
    fn sample(self, along: usize) -> [usize; 2] {
        let last = SIDE - 1;
        match self {
            Self::MinX => [0, along],
            Self::MaxX => [last, along],
            Self::MinZ => [along, 0],
            Self::MaxZ => [along, last],
        }
    }

    /// Whether tile `key`, of a level `[columns, rows]` of tiles, has a
    /// neighbor on it.
    fn neighbored(self, key: TileKey, [columns, rows]: [u32; 2]) -> bool {
        match self {
            Self::MinX => key.x > 0,
            Self::MaxX => u32::from(key.x) + 1 < columns,
            Self::MinZ => key.z > 0,
            Self::MaxZ => u32::from(key.z) + 1 < rows,
        }
    }
}

/// A portal of a tile: its side, its first and last samples along it, and
/// the sample walks cross it at, the middle one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Portal {
    pub side: Side,
    pub first: u16,
    pub last: u16,
    pub at: u16,
}

record!(Portal {
    side,
    first,
    last,
    at,
});

impl Portal {
    /// The sample walks cross it at, within its tile: column and row.
    pub fn sample(&self) -> [usize; 2] {
        self.side.sample(usize::from(self.at))
    }

    /// That sample's index in its tile's ground.
    pub(super) fn index(&self) -> usize {
        let [x, z] = self.sample();
        z * SIDE + x
    }
}

/// A tile's portals as a gait walks it, and what walking inside the tile
/// between each two costs: the walk graph's piece over the tile, a record
/// per tile and gait that hosts keep and the cook ships. Its portals are the
/// runs of samples the gait stands on along the tile's sides with
/// neighbors, a portal each at most [`PORTAL_RUN`] samples of them; a tile
/// and its neighbor find the same portals on the edge they share, whose
/// ground both hold. Costs are in 256ths of the level's spacing: each
/// step's run and rise.
#[derive(Clone, Debug, PartialEq)]
pub struct TilePortals {
    key: TileKey,
    portals: Vec<Portal>,
    // Each two portals' cost, the first's index the lower, row by row;
    // UNREACHED where no walk inside the tile joins them.
    costs: Vec<u32>,
}

impl TilePortals {
    /// The portals of `ground`, a tile of `pyramid`, as `gait` walks it, and
    /// what walking inside the tile between each two costs.
    pub fn new(ground: &WalkGround, pyramid: &Pyramid, gait: &Gait) -> Self {
        let key = ground.key;
        let stands = ground.standing(gait.max_water);
        let tiles = pyramid.tiles(u32::from(key.level));
        let mut portals = Vec::new();
        for side in Side::ALL
            .into_iter()
            .filter(|side| side.neighbored(key, tiles))
        {
            let mut run = None;
            for along in 0..=SIDE {
                let open = along < SIDE && {
                    let [x, z] = side.sample(along);
                    stands[z * SIDE + x]
                };
                match (open, run) {
                    (true, None) => run = Some(along),
                    (false, Some(first)) => {
                        split_run(side, first, along - 1, &mut portals);
                        run = None;
                    }
                    (true, Some(_)) | (false, None) => {}
                }
            }
        }
        let stepper = Stepper::new(ground.spacing, gait.max_slope);
        let samples: Vec<usize> = portals.iter().map(Portal::index).collect();
        let count = portals.len();
        let mut costs = vec![UNREACHED; count * count.saturating_sub(1) / 2];
        let mut flood = Flood::default();
        let (mut whole, mut searched) = (u64::MAX, 0);
        for from in 0..count.saturating_sub(1) {
            flood.start(samples[from], &samples[from + 1..]);
            flood.step(ground, &stands, &stepper, (&mut whole, &mut searched));
            for to in from + 1..count {
                costs[pair(count, from, to)] = flood.cost[samples[to]];
            }
        }
        Self {
            key,
            portals,
            costs,
        }
    }

    pub fn key(&self) -> TileKey {
        self.key
    }

    pub fn portals(&self) -> &[Portal] {
        &self.portals
    }

    /// What walking inside the tile from portal `a` to portal `b` costs;
    /// `None` where no walk inside it joins them.
    pub fn cost(&self, a: usize, b: usize) -> Option<u32> {
        if a == b {
            return Some(0);
        }
        let (low, high) = if a < b { (a, b) } else { (b, a) };
        let cost = self.costs[pair(self.portals.len(), low, high)];
        (cost != UNREACHED).then_some(cost)
    }

    /// The portals walks cross at the sample of its tile `sample`.
    pub(super) fn at(&self, sample: [usize; 2]) -> impl Iterator<Item = usize> + '_ {
        self.portals
            .iter()
            .enumerate()
            .filter(move |(_, portal)| portal.sample() == sample)
            .map(|(index, _)| index)
    }
}

/// Where the cost between portals `low` and `high` (lower first) of `count`
/// lies among a tile's costs.
fn pair(count: usize, low: usize, high: usize) -> usize {
    low * (2 * count - low - 1) / 2 + (high - low - 1)
}

/// Splits the run of samples `first` to `last` along `side` into portals of
/// at most [`PORTAL_RUN`] samples, as even as they divide.
fn split_run(side: Side, first: usize, last: usize, portals: &mut Vec<Portal>) {
    let length = last - first + 1;
    let count = length.div_ceil(PORTAL_RUN);
    for index in 0..count {
        let start = first + index * length / count;
        let end = first + (index + 1) * length / count - 1;
        portals.push(Portal {
            side,
            first: start as u16,
            last: end as u16,
            at: ((start + end) / 2) as u16,
        });
    }
}

/// A tile's portals as its record holds them: its key, its portals and
/// their costs; reading checks every portal lies on its side and that a
/// cost stands for each two.
impl codec::Codec for TilePortals {
    fn write(&self, out: &mut Writer) {
        out.put(&self.key).put(&self.portals).put(&self.costs);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        let key = input.get()?;
        let portals: Vec<Portal> = input.get()?;
        let costs: Vec<u32> = input.get()?;
        let last = (SIDE - 1) as u16;
        let placed = portals.iter().all(|portal| {
            portal.first <= portal.at && portal.at <= portal.last && portal.last <= last
        });
        let count = portals.len();
        if !placed || costs.len() != count * count.saturating_sub(1) / 2 {
            return Err(CodecError("invalid portals"));
        }
        Ok(Self {
            key,
            portals,
            costs,
        })
    }
}

/// The portals of tile `key` over `pyramid` as `gait` walks it, its ground
/// generated from `layers` (the edits reaching it among them) as the tile's
/// record holds it: heights and water at the steps of the height grid
/// tiles travel at. A tile without valid heights has none.
pub fn tile_portals(
    layers: &Arc<Layers>,
    pyramid: &Pyramid,
    key: TileKey,
    gait: &Gait,
) -> Result<TilePortals, String> {
    let mut whole = u64::MAX;
    let tile = Generator::new(layers.clone(), pyramid, key)
        .advance(&mut whole)
        .ok_or("the tile's heights are invalid")?;
    let tile: Tile = codec::decode(&codec::encode(&tile)).map_err(|error| error.to_string())?;
    Ok(TilePortals::new(
        &WalkGround::new(&tile, layers, pyramid),
        pyramid,
        gait,
    ))
}
