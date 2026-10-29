use super::*;
use engine_core::Expected;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    MinX,
    MaxX,
    MinZ,
    MaxZ,
}

tagged!(Side {
    0 => MinX,
    1 => MaxX,
    2 => MinZ,
    3 => MaxZ,
});

impl Side {
    const ALL: [Self; 4] = [Self::MinX, Self::MaxX, Self::MinZ, Self::MaxZ];

    fn sample(self, along: usize) -> [usize; 2] {
        let last = SIDE - 1;
        match self {
            Self::MinX => [0, along],
            Self::MaxX => [last, along],
            Self::MinZ => [along, 0],
            Self::MaxZ => [along, last],
        }
    }

    fn neighbored(self, key: TileKey, [columns, rows]: [u32; 2]) -> bool {
        match self {
            Self::MinX => key.x > 0,
            Self::MaxX => u32::from(key.x) + 1 < columns,
            Self::MinZ => key.z > 0,
            Self::MaxZ => u32::from(key.z) + 1 < rows,
        }
    }
}

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
    pub fn sample(&self) -> [usize; 2] {
        self.side.sample(usize::from(self.at))
    }

    pub(super) fn index(&self) -> usize {
        let [x, z] = self.sample();
        z * SIDE + x
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TilePortals {
    key: TileKey,
    portals: Vec<Portal>,
    costs: Vec<u32>,
}

impl TilePortals {
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

    pub fn cost(&self, a: usize, b: usize) -> Option<u32> {
        if a == b {
            return Some(0);
        }
        let (low, high) = if a < b { (a, b) } else { (b, a) };
        let cost = self.costs[pair(self.portals.len(), low, high)];
        (cost != UNREACHED).then_some(cost)
    }

    pub(super) fn at(&self, sample: [usize; 2]) -> impl Iterator<Item = usize> + '_ {
        self.portals
            .iter()
            .enumerate()
            .filter(move |(_, portal)| portal.sample() == sample)
            .map(|(index, _)| index)
    }
}

fn pair(count: usize, low: usize, high: usize) -> usize {
    low * (2 * count - low - 1) / 2 + (high - low - 1)
}

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

impl codec::Codec for TilePortals {
    fn write(&self, out: &mut Writer) {
        out.put(&self.key).put(&self.portals).put(&self.costs);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
        let key = input.get()?;
        let portals: Vec<Portal> = input.get()?;
        let costs: Vec<u32> = input.get()?;
        let last = (SIDE - 1) as u16;
        let placed = portals.iter().all(|portal| {
            portal.first <= portal.at && portal.at <= portal.last && portal.last <= last
        });
        let count = portals.len();
        if !placed || costs.len() != count * count.saturating_sub(1) / 2 {
            return Err(codec::invalid("$.portals"));
        }
        Ok(Self {
            key,
            portals,
            costs,
        })
    }
}

pub fn tile_portals(
    layers: &Arc<Layers>,
    pyramid: &Pyramid,
    key: TileKey,
    gait: &Gait,
) -> Result<TilePortals, Error> {
    let mut whole = u64::MAX;
    let tile = Generator::new(layers.clone(), pyramid, key)
        .advance(&mut whole)
        .ok_or_else(|| {
            Error::new(ErrorCode::InvalidTerrain, "$.heights").expected(Expected::Finite)
        })?;
    let tile: Tile = codec::decode(&codec::encode(&tile))?;
    Ok(TilePortals::new(
        &WalkGround::new(&tile, layers, pyramid),
        pyramid,
        gait,
    ))
}
