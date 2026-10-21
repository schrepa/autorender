//! The shallow-water equations over a square of cells on the CPU, as the
//! renderer simulates rivers around the camera (`water_sim.wgsl`, whose
//! `SIM_` constants mirror these): Kurganov and Petrova's well-balanced,
//! positivity-preserving central-upwind scheme (2007), second order in space
//! (each cell's water reconstructed linearly, its slopes limited by a
//! generalized minmod and kept from dipping below the ground at its faces)
//! and in time (Heun's two stages), with Manning's friction and velocities
//! desingularized where the water thins out. Each cell holds its water's
//! surface, its flow along x and z (square meters a second) and the level a
//! lake holds it at, if one does: lakes stand still, taking the water that
//! runs into them and giving none. The ground stands at the cells' corners.
//! Water leaves freely over the square's edges and none comes in over them;
//! rivers bring theirs at the cells their sources name. The water carries
//! foam, churned up where it runs faster than its waves and where it slams
//! into slower water, carried off by the current and fading as its bubbles
//! break.
//!
//! Steps run as the live simulation's do, every cell by the same seconds
//! ([`Steps::Live`]), or each cell by as long as its own and its
//! neighbors' fastest waves allow ([`Steps::Local`]): water settling toward
//! its steady state gets there in far fewer steps, and the state it settles
//! to is the same, as each cell's water stops changing when what flows in
//! and out of it balances, whatever its steps. Only cells a few cells from
//! water step: dry ground stays as it is until water comes.
//!
//! The water's surface stands hundreds of meters up while a thin stream's
//! changes a step by hundredths of a millimeter, finer than `f32` holds
//! there: the state and every operation are `f64`, plain arithmetic in one
//! order, so each host steps the same bits and no water rounds away.

use crate::hydraulics::GRAVITY;

/// The generalized minmod's theta.
pub const THETA: f32 = 1.3;
/// Meters of water below which velocities are desingularized.
pub const THIN: f32 = 0.01;
/// Meters of water under which a cell is dry.
pub const DRY: f32 = 1.0e-4;
/// Meters a second the fastest water runs.
pub const FASTEST: f32 = 20.0;
/// The level a cell no lake holds holds as its lake's.
pub const UNHELD: f32 = -1.0e30;
/// Manning's roughness of the ground the water runs over: a clean mountain
/// channel's.
pub const ROUGHNESS: f32 = 0.03;
/// The share of a cell the fastest water and waves may cross in a step:
/// Kurganov and Petrova keep the water's depth positive within a quarter,
/// in two dimensions.
pub const COURANT: f32 = 0.24;
/// Meters a second the fastest water and waves run as the live steps'
/// length counts them: the fastest water and the waves of water five meters
/// deep.
pub const BOUNDING_SPEED: f32 = 27.0;
/// Foam: the Froude number past which rapids churn it up and how fast (a
/// second, per unit past it); how fast water converging churns it (per
/// unit of convergence, a second); the seconds it takes to fade to 1/e; the
/// most it piles up to; and the meters of water under which a cell holds
/// none.
pub const FOAM_FROUDE: f32 = 2.0;
pub const FOAM_RAPIDS: f32 = 0.3;
pub const FOAM_PILING: f32 = 0.25;
pub const FOAM_LIFE: f32 = 6.0;
pub const FOAM_MOST: f32 = 1.5;
pub const FOAM_WET: f32 = 0.005;
/// Meters a second the water rivers bring runs along them as it comes.
pub const SOURCE_SPEED: f32 = 1.5;

/// Cells from water within which cells step: what reaches a cell's water in
/// a step lies within two, and water spreads less than a cell over the
/// [`LISTED_FOR`] steps between listings.
const REACH: usize = 3;
const LISTED_FOR: u32 = 4;
/// Local steps take this share of the most their cells' waves allow, and
/// count no wave slower than [`SLOWEST`] meters a second.
const LOCAL_SHARE: f64 = 0.9;
const SLOWEST: f64 = 1.0;

/// The seconds of a live step over cells `cell` meters across: the longest
/// in which the fastest water and waves cross [`COURANT`] of a cell.
pub fn step_seconds(cell: f32) -> f32 {
    COURANT * cell / BOUNDING_SPEED
}

/// How long cells' steps run: all of them `seconds` alike, as the live
/// simulation's, or each as long as its own waves allow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Steps {
    Live(f32),
    Local,
}

/// Where a river brings water to a cell (its index, row by row): the cubic
/// meters a second it brings there and the way it runs, as a share of
/// [`SOURCE_SPEED`] along x and z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Source {
    pub cell: u32,
    pub rate: f32,
    pub way: [f32; 2],
}

/// How fast water comes and goes over a square, cubic meters a second, as
/// its latest step found it: what its sources bring, what leaves over its
/// edges and what runs into lakes (the mean of the step's two stages, as
/// its water's change is).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rates {
    pub sources: f64,
    pub outflow: f64,
    pub absorbed: f64,
}

/// A square of cells' water as the shallow-water equations run it.
pub struct ShallowWater {
    n: usize,
    cell: f64,
    // The ground at the middles of the faces between columns (n + 1 a row)
    // and between rows (n a row, n + 1 rows), and at the cells' middles.
    ground_x: Vec<f64>,
    ground_z: Vec<f64>,
    bed: Vec<f64>,
    // The step's start and its first stage: each cell's surface and flow
    // along x and z; and the level a lake holds each at.
    start: Vec<[f64; 3]>,
    first: Vec<[f64; 3]>,
    held: Vec<Option<f64>>,
    // Each cell's faces as reconstructed, east, west, north and south: the
    // water's surface and depth there, and its velocity across the face and
    // along it.
    faces: Vec<[[f64; 4]; 4]>,
    // The fluxes through the faces between columns and between rows: mass,
    // and momentum across and along the face; and the fastest wave through
    // each in the step's first stage.
    flux_x: Vec<[f64; 3]>,
    flux_z: Vec<[f64; 3]>,
    wave_x: Vec<f64>,
    wave_z: Vec<f64>,
    // Each cell's seconds this step and the fastest wave through its
    // faces, its foam and the foam it churns.
    seconds: Vec<f64>,
    fastest: Vec<f64>,
    foam: Vec<f64>,
    churned: Vec<f64>,
    sources: Vec<Source>,
    // The cells that step, in order, and which do; steps since they were
    // last listed; and which stood dry beside dry neighbors when last
    // reconstructed, their faces dry since.
    listed: Vec<u32>,
    stepping: Vec<bool>,
    quiet: Vec<bool>,
    since_listed: u32,
    rates: Rates,
}

impl ShallowWater {
    /// `cells` cells along a side, `cell` meters across, over ground at
    /// their corners `corners` (`cells + 1` a row, rows along z), starting
    /// from `start` (each cell's surface, or its ground where dry, and its
    /// flow along x and z), lakes holding the cells `held` gives a level
    /// (NaN where none does), with `foam` and the rivers' `sources`. `None`
    /// where the sizes do not match.
    pub fn new(
        (cells, cell): (usize, f32),
        corners: &[f32],
        (start, held, foam): (&[[f32; 3]], &[f32], &[f32]),
        sources: Vec<Source>,
    ) -> Option<Self> {
        let n = cells;
        if n == 0
            || corners.len() != (n + 1) * (n + 1)
            || start.len() != n * n
            || held.len() != n * n
            || foam.len() != n * n
            || sources.iter().any(|source| source.cell as usize >= n * n)
        {
            return None;
        }
        let corner = |i: usize, j: usize| f64::from(corners[j * (n + 1) + i]);
        let mut ground_x = vec![0.0; (n + 1) * n];
        for j in 0..n {
            for i in 0..=n {
                ground_x[j * (n + 1) + i] = 0.5 * (corner(i, j) + corner(i, j + 1));
            }
        }
        let mut ground_z = vec![0.0; n * (n + 1)];
        for j in 0..=n {
            for i in 0..n {
                ground_z[j * n + i] = 0.5 * (corner(i, j) + corner(i + 1, j));
            }
        }
        let mut bed = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                bed[j * n + i] = 0.5 * (ground_x[j * (n + 1) + i] + ground_x[j * (n + 1) + i + 1]);
            }
        }
        let held: Vec<Option<f64>> = held
            .iter()
            .map(|&level| (!level.is_nan()).then(|| f64::from(level)))
            .collect();
        // Water no deeper than a dry cell's stands at its ground, still.
        let dry = f64::from(DRY);
        let start: Vec<[f64; 3]> = start
            .iter()
            .zip(&held)
            .zip(&bed)
            .map(|((&water, &level), &ground)| match level {
                Some(level) => [level, 0.0, 0.0],
                None if f64::from(water[0]) - ground <= dry => [ground, 0.0, 0.0],
                None => water.map(f64::from),
            })
            .collect();
        let mut water = Self {
            n,
            cell: f64::from(cell),
            ground_x,
            ground_z,
            bed,
            first: start.clone(),
            start,
            held,
            faces: vec![[[0.0; 4]; 4]; n * n],
            flux_x: vec![[0.0; 3]; (n + 1) * n],
            flux_z: vec![[0.0; 3]; n * (n + 1)],
            wave_x: vec![0.0; (n + 1) * n],
            wave_z: vec![0.0; n * (n + 1)],
            seconds: vec![0.0; n * n],
            fastest: vec![0.0; n * n],
            foam: foam.iter().map(|&foam| f64::from(foam)).collect(),
            churned: vec![0.0; n * n],
            sources,
            listed: Vec::new(),
            stepping: vec![false; n * n],
            quiet: vec![false; n * n],
            since_listed: 0,
            rates: Rates::default(),
        };
        // Every cell's faces as it stands, so cells that never step read
        // their dry faces.
        for at in 0..n * n {
            water.reconstruct_cell(0, at);
        }
        water.list();
        Some(water)
    }

    /// Cell `at`'s water (row by row): its surface and flow along x and z,
    /// the level a lake holds it at (`None` where none does), and its foam.
    pub fn water(&self, at: usize) -> ([f32; 3], Option<f32>, f32) {
        (
            self.start[at].map(|value| value as f32),
            self.held[at].map(|level| level as f32),
            self.foam[at] as f32,
        )
    }

    /// Cell `at`'s depth and velocity along x and z where water no lake
    /// holds flows there.
    pub fn flowing(&self, at: usize) -> Option<[f32; 3]> {
        let [surface, x, z] = self.start[at];
        let h = surface - self.bed[at];
        (self.held[at].is_none() && h > f64::from(DRY))
            .then(|| [h as f32, velocity(h, x) as f32, velocity(h, z) as f32])
    }

    /// Cubic meters of water no lake holds.
    pub fn volume(&self) -> f64 {
        let area = self.cell * self.cell;
        self.listed
            .iter()
            .map(|&at| at as usize)
            .filter(|&at| self.held[at].is_none())
            .map(|at| greater(self.start[at][0] - self.bed[at], 0.0) * area)
            .sum()
    }

    /// The cells stepping, in order.
    pub fn listed(&self) -> &[u32] {
        &self.listed
    }

    /// Where rivers bring water, cells in order.
    pub fn sources(&self) -> &[Source] {
        &self.sources
    }

    /// How fast water came and went in the latest step.
    pub fn rates(&self) -> Rates {
        self.rates
    }

    /// Runs a step: its two stages, the rivers bringing their water and the
    /// foam, every cell for as long as `steps` says.
    pub fn step(&mut self, steps: Steps) {
        if self.since_listed >= LISTED_FOR {
            self.list();
        }
        self.since_listed += 1;
        // Heun's step carries the mean of its two stages' fluxes.
        let face = 0.5 * self.cell;
        self.rates.outflow = 0.0;
        self.rates.absorbed = 0.0;
        for stage in 0..2 {
            for index in 0..self.listed.len() {
                self.reconstruct_cell(stage, self.listed[index] as usize);
            }
            let (out, absorbed) = self.exchange(stage == 0);
            self.rates.outflow += out * face;
            self.rates.absorbed += absorbed * face;
            if stage == 0 {
                self.time(steps);
            }
            self.update(stage);
        }
        self.rates.sources = self.bring();
        self.churn();
    }

    /// Lists the cells within [`REACH`] of water no lake holds or of a
    /// river's source, those listed before staying listed, in order.
    fn list(&mut self) {
        self.since_listed = 0;
        let (n, reach) = (self.n, REACH);
        let wet = |water: &Self, at: usize| {
            water.held[at].is_none() && water.start[at][0] > water.bed[at]
        };
        let mut seeds: Vec<usize> = if self.listed.is_empty() {
            (0..n * n).filter(|&at| wet(self, at)).collect()
        } else {
            self.listed
                .iter()
                .map(|&at| at as usize)
                .filter(|&at| wet(self, at))
                .collect()
        };
        seeds.extend(self.sources.iter().map(|source| source.cell as usize));
        let before = self.listed.len();
        for at in seeds {
            let (i, j) = (at % n, at / n);
            for z in j.saturating_sub(reach)..(j + reach + 1).min(n) {
                for x in i.saturating_sub(reach)..(i + reach + 1).min(n) {
                    let near = z * n + x;
                    if !self.stepping[near] {
                        self.stepping[near] = true;
                        self.listed.push(near as u32);
                    }
                }
            }
        }
        if self.listed.len() > before {
            self.listed.sort_unstable();
        }
    }

    /// Cell `at`'s faces from the stage's state (0: the step's start, 1:
    /// its first stage); a dry cell beside dry neighbors keeps the dry faces
    /// it had.
    fn reconstruct_cell(&mut self, stage: usize, at: usize) {
        let n = self.n;
        let state = if stage == 0 { &self.start } else { &self.first };
        let (i, j) = (at % n, at / n);
        let (w, e, s, nn) = (
            if i > 0 { at - 1 } else { at },
            if i + 1 < n { at + 1 } else { at },
            if j > 0 { at - n } else { at },
            if j + 1 < n { at + n } else { at },
        );
        let dry = |cell: usize| state[cell][0] <= self.bed[cell] && self.held[cell].is_none();
        let quiet = dry(at) && dry(w) && dry(e) && dry(s) && dry(nn);
        if quiet && self.quiet[at] {
            return;
        }
        self.quiet[at] = quiet;
        let here = state[at];
        let west = state[w];
        let east = state[e];
        let south = state[s];
        let north = state[nn];
        let ground_x = [
            self.ground_x[j * (n + 1) + i],
            self.ground_x[j * (n + 1) + i + 1],
        ];
        let ground_z = [self.ground_z[at], self.ground_z[at + n]];
        // Across x: east then west, kept above the ground at both faces.
        let slope_x = [0, 1, 2].map(|k| half_slope(west[k], here[k], east[k]));
        let mut east_w = here[0] + slope_x[0];
        let mut west_w = here[0] - slope_x[0];
        if east_w < ground_x[1] {
            east_w = ground_x[1];
            west_w = 2.0 * here[0] - ground_x[1];
        } else if west_w < ground_x[0] {
            west_w = ground_x[0];
            east_w = 2.0 * here[0] - ground_x[0];
        }
        // Across z: north then south, the flow along z across these faces.
        let slope_z = [0, 1, 2].map(|k| half_slope(south[k], here[k], north[k]));
        let mut north_w = here[0] + slope_z[0];
        let mut south_w = here[0] - slope_z[0];
        if north_w < ground_z[1] {
            north_w = ground_z[1];
            south_w = 2.0 * here[0] - ground_z[1];
        } else if south_w < ground_z[0] {
            south_w = ground_z[0];
            north_w = 2.0 * here[0] - ground_z[0];
        }
        self.faces[at] = [
            face(
                east_w,
                ground_x[1],
                here[1] + slope_x[1],
                here[2] + slope_x[2],
            ),
            face(
                west_w,
                ground_x[0],
                here[1] - slope_x[1],
                here[2] - slope_x[2],
            ),
            face(
                north_w,
                ground_z[1],
                here[2] + slope_z[2],
                here[1] + slope_z[1],
            ),
            face(
                south_w,
                ground_z[0],
                here[2] - slope_z[2],
                here[1] - slope_z[1],
            ),
        ];
    }

    /// The fluxes through every face of a listed cell, each found once:
    /// its west and south faces, and its east and north faces where the cell
    /// beyond is not listed. What leaves over the square's edges and runs
    /// into lakes (square meters a second per meter of face), and with
    /// `waves`, the fastest wave through each face.
    fn exchange(&mut self, waves: bool) -> (f64, f64) {
        let n = self.n;
        let (mut out, mut absorbed) = (0.0, 0.0);
        for index in 0..self.listed.len() {
            let at = self.listed[index] as usize;
            let (i, j) = (at % n, at / n);
            let east = i + 1 == n || !self.stepping[at + 1];
            let north = j + 1 == n || !self.stepping[at + n];
            for (face, taken) in [(i, true), (i + 1, east)] {
                if !taken {
                    continue;
                }
                let (flux, wave) = if face == 0 {
                    edge(self.faces[j * n][1], 1.0, &mut out)
                } else if face == n {
                    edge(self.faces[j * n + n - 1][0], -1.0, &mut out)
                } else {
                    let (near, far) = (j * n + face - 1, j * n + face);
                    let held = (self.held[near].is_some(), self.held[far].is_some());
                    between(self.faces[near][0], self.faces[far][1], held, &mut absorbed)
                };
                self.flux_x[j * (n + 1) + face] = flux;
                if waves {
                    self.wave_x[j * (n + 1) + face] = wave;
                }
            }
            for (face, taken) in [(j, true), (j + 1, north)] {
                if !taken {
                    continue;
                }
                let (flux, wave) = if face == 0 {
                    edge(self.faces[i][3], 1.0, &mut out)
                } else if face == n {
                    edge(self.faces[(n - 1) * n + i][2], -1.0, &mut out)
                } else {
                    let (near, far) = ((face - 1) * n + i, face * n + i);
                    let held = (self.held[near].is_some(), self.held[far].is_some());
                    between(self.faces[near][2], self.faces[far][3], held, &mut absorbed)
                };
                self.flux_z[face * n + i] = flux;
                if waves {
                    self.wave_z[face * n + i] = wave;
                }
            }
        }
        (out, absorbed)
    }

    /// Each listed cell's seconds this step: all alike, or as long as the
    /// fastest wave through its faces and its listed neighbors' allows, so
    /// neighbors step alike.
    fn time(&mut self, steps: Steps) {
        let n = self.n;
        if let Steps::Live(seconds) = steps {
            for index in 0..self.listed.len() {
                self.seconds[self.listed[index] as usize] = f64::from(seconds);
            }
            return;
        }
        for index in 0..self.listed.len() {
            let at = self.listed[index] as usize;
            let (i, j) = (at % n, at / n);
            self.fastest[at] = greater(
                greater(
                    self.wave_x[j * (n + 1) + i],
                    self.wave_x[j * (n + 1) + i + 1],
                ),
                greater(self.wave_z[at], self.wave_z[at + n]),
            );
        }
        let longest = LOCAL_SHARE * f64::from(COURANT) * self.cell;
        for index in 0..self.listed.len() {
            let at = self.listed[index] as usize;
            let (i, j) = (at % n, at / n);
            let mut fastest = self.fastest[at];
            for (near, inside) in [
                (at.wrapping_sub(1), i > 0),
                (at + 1, i + 1 < n),
                (at.wrapping_sub(n), j > 0),
                (at + n, j + 1 < n),
            ] {
                if inside && self.stepping[near] {
                    fastest = greater(fastest, self.fastest[near]);
                }
            }
            self.seconds[at] = longest / greater(fastest, SLOWEST);
        }
    }

    /// Each listed cell's water after the stage: from the step's start in
    /// the first stage, Heun's mean of the start and the first stage's step
    /// in the second.
    fn update(&mut self, stage: usize) {
        let n = self.n;
        let dx = self.cell;
        let gravity = f64::from(GRAVITY);
        for index in 0..self.listed.len() {
            let at = self.listed[index] as usize;
            let (i, j) = (at % n, at / n);
            let dt = self.seconds[at];
            let west = self.flux_x[j * (n + 1) + i];
            let east = self.flux_x[j * (n + 1) + i + 1];
            let south = self.flux_z[at];
            let north = self.flux_z[at + n];
            // A quiet cell nothing flows through stays as it is.
            if self.quiet[at] && [west, east, south, north] == [[0.0; 3]; 4] {
                continue;
            }
            let faces = &self.faces[at];
            let depth_x = 0.5 * (faces[0][1] + faces[1][1]);
            let depth_z = 0.5 * (faces[2][1] + faces[3][1]);
            let slope_x =
                (self.ground_x[j * (n + 1) + i + 1] - self.ground_x[j * (n + 1) + i]) / dx;
            let slope_z = (self.ground_z[at + n] - self.ground_z[at]) / dx;
            // The fluxes through faces between rows carry the flow along z
            // across them and the flow along x along them.
            let change = [
                -(east[0] - west[0]) / dx - (north[0] - south[0]) / dx,
                -(east[1] - west[1]) / dx
                    - (north[2] - south[2]) / dx
                    - gravity * depth_x * slope_x,
                -(east[2] - west[2]) / dx
                    - (north[1] - south[1]) / dx
                    - gravity * depth_z * slope_z,
            ];
            let start = self.start[at];
            let next = if stage == 0 {
                [0, 1, 2].map(|k| start[k] + dt * change[k])
            } else {
                let first = self.first[at];
                [0, 1, 2].map(|k| 0.5 * (start[k] + first[k] + dt * change[k]))
            };
            let settled = settle(next, self.bed[at], self.held[at], dt);
            if stage == 0 {
                self.first[at] = settled;
            } else {
                self.start[at] = settled;
            }
        }
    }

    /// The rivers bring their water to their cells over each cell's step;
    /// the cubic meters a second they bring.
    fn bring(&mut self) -> f64 {
        let area = self.cell * self.cell;
        let speed = f64::from(SOURCE_SPEED);
        let mut rate = 0.0;
        for index in 0..self.sources.len() {
            let source = self.sources[index];
            let at = source.cell as usize;
            if self.held[at].is_some() {
                continue;
            }
            let added = f64::from(source.rate) * self.seconds[at] / area;
            let cell = &mut self.start[at];
            cell[0] += added;
            cell[1] += added * f64::from(source.way[0]) * speed;
            cell[2] += added * f64::from(source.way[1]) * speed;
            rate += f64::from(source.rate);
        }
        rate
    }

    /// Cell `(i, j)`'s depth and velocity, or `instead` where it is dry or
    /// outside the square.
    fn moving(&self, i: isize, j: isize, instead: [f64; 3]) -> [f64; 3] {
        let n = self.n as isize;
        if i < 0 || j < 0 || i >= n || j >= n {
            return instead;
        }
        let at = (j * n + i) as usize;
        let cell = self.start[at];
        let h = cell[0] - self.bed[at];
        if h <= f64::from(FOAM_WET) {
            return instead;
        }
        [h, velocity(h, cell[1]), velocity(h, cell[2])]
    }

    /// The foam at cell `p` (fractional, along x and z), bilinearly.
    fn foam_at(&self, p: [f64; 2]) -> f64 {
        let n = self.n;
        let last = (n - 1) as f64;
        let q = [within(p[0], 0.0, last), within(p[1], 0.0, last)];
        let low = [q[0].floor() as usize, q[1].floor() as usize];
        let high = [(low[0] + 1).min(n - 1), (low[1] + 1).min(n - 1)];
        let f = [q[0] - q[0].floor(), q[1] - q[1].floor()];
        let at = |x: usize, z: usize| self.foam[z * n + x];
        let below = mix(at(low[0], low[1]), at(high[0], low[1]), f[0]);
        let above = mix(at(low[0], high[1]), at(high[0], high[1]), f[0]);
        mix(below, above, f[1])
    }

    /// Each listed cell's foam after its step: what the current carried
    /// there (traced back along it), faded, and what the water churned up
    /// there.
    fn churn(&mut self) {
        let n = self.n;
        let dx = self.cell;
        let gravity = f64::from(GRAVITY);
        for index in 0..self.listed.len() {
            let at = self.listed[index] as usize;
            let (i, j) = ((at % n) as isize, (at / n) as isize);
            let here = self.moving(i, j, [0.0; 3]);
            if here[0] <= f64::from(FOAM_WET) {
                self.churned[at] = 0.0;
                continue;
            }
            let dt = self.seconds[at];
            let carried =
                self.foam_at([i as f64 - here[1] * dt / dx, j as f64 - here[2] * dt / dx]);
            let speed = (here[1] * here[1] + here[2] * here[2]).sqrt();
            let froude = speed / (gravity * here[0]).sqrt();
            let rapids = greater(froude - f64::from(FOAM_FROUDE), 0.0) * f64::from(FOAM_RAPIDS);
            let east = self.moving(i + 1, j, here)[1];
            let west = self.moving(i - 1, j, here)[1];
            let north = self.moving(i, j + 1, here)[2];
            let south = self.moving(i, j - 1, here)[2];
            let converging = greater(-(east - west + north - south) / (2.0 * dx), 0.0);
            let churned = carried * fading(dt / f64::from(FOAM_LIFE))
                + dt * (rapids + f64::from(FOAM_PILING) * converging);
            self.churned[at] = lesser(churned, f64::from(FOAM_MOST));
        }
        for index in 0..self.listed.len() {
            let at = self.listed[index] as usize;
            self.foam[at] = self.churned[at];
        }
    }
}

fn lesser(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

fn greater(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

fn within(x: f64, low: f64, high: f64) -> f64 {
    lesser(greater(x, low), high)
}

/// `a` blended `t` of the way to `b`.
fn mix(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}

/// The least of `a`, `b` and `c` by size where all three share a sign,
/// with that sign; else zero.
fn minmod(a: f64, b: f64, c: f64) -> f64 {
    let positive = (a > 0.0) & (b > 0.0) & (c > 0.0);
    let negative = (a < 0.0) & (b < 0.0) & (c < 0.0);
    let least = lesser(a.abs(), lesser(b.abs(), c.abs()));
    let sign = f64::from(u8::from(positive)) - f64::from(u8::from(negative));
    sign * least
}

/// Half a cell's change across it, from the values either side of it,
/// limited.
fn half_slope(before: f64, middle: f64, after: f64) -> f64 {
    let theta = f64::from(THETA);
    0.5 * minmod(
        theta * (middle - before),
        0.5 * (after - before),
        theta * (after - middle),
    )
}

/// The velocity of `q` square meters a second of flow `h` meters deep,
/// desingularized as the water thins out.
fn velocity(h: f64, q: f64) -> f64 {
    q * per_flow(h)
}

/// Meters a second of velocity per square meter a second of flow `h`
/// meters deep: its depth's reciprocal, desingularized as the water thins
/// out.
fn per_flow(h: f64) -> f64 {
    let h4 = h * h * h * h;
    let thin = f64::from(THIN);
    let thin4 = thin * thin * thin * thin;
    std::f64::consts::SQRT_2 * h / (h4 + greater(h4, thin4)).sqrt()
}

/// A face's water from the cell's surface `w` reconstructed there, the
/// ground at the face, and its flow across and along the face: no faster
/// than [`FASTEST`], so no wave outruns the steps' length.
fn face(w: f64, ground: f64, across: f64, along: f64) -> [f64; 4] {
    let h = greater(w - ground, 0.0);
    if h <= 0.0 {
        return [w, 0.0, 0.0, 0.0];
    }
    let per = per_flow(h);
    let fastest = f64::from(FASTEST);
    [
        w,
        h,
        within(across * per, -fastest, fastest),
        within(along * per, -fastest, fastest),
    ]
}

/// The central-upwind flux through a face whose water stands as `l` on its
/// near side and `r` on its far one: its mass and its momentum across and
/// along the face, and the fastest wave through it.
fn flux(l: [f64; 4], r: [f64; 4]) -> ([f64; 3], f64) {
    // Two dry faces pass nothing, and no wave crosses them.
    if l[1] <= 0.0 && r[1] <= 0.0 {
        return ([0.0; 3], 0.0);
    }
    let gravity = f64::from(GRAVITY);
    let cl = (gravity * l[1]).sqrt();
    let cr = (gravity * r[1]).sqrt();
    let fastest = greater(greater(l[2] + cl, r[2] + cr), 0.0);
    let slowest = lesser(lesser(l[2] - cl, r[2] - cr), 0.0);
    let span = fastest - slowest;
    let wave = greater(fastest, -slowest);
    if span < 1.0e-6 {
        return ([0.0; 3], wave);
    }
    let fl = [
        l[1] * l[2],
        l[1] * l[2] * l[2] + 0.5 * gravity * l[1] * l[1],
        l[1] * l[2] * l[3],
    ];
    let fr = [
        r[1] * r[2],
        r[1] * r[2] * r[2] + 0.5 * gravity * r[1] * r[1],
        r[1] * r[2] * r[3],
    ];
    let ul = [l[0], l[1] * l[2], l[1] * l[3]];
    let ur = [r[0], r[1] * r[2], r[1] * r[3]];
    let diffusion = fastest * slowest / span;
    (
        [0, 1, 2].map(|k| (fastest * fl[k] - slowest * fr[k]) / span + diffusion * (ur[k] - ul[k])),
        wave,
    )
}

/// The flux through the face from a cell standing as `l` to one standing
/// as `r`, whether each is `held` by a lake: none where it would carry
/// water out of a lake; what runs into one adds to `absorbed`.
fn between(
    l: [f64; 4],
    r: [f64; 4],
    (near, far): (bool, bool),
    absorbed: &mut f64,
) -> ([f64; 3], f64) {
    let (flux, wave) = flux(l, r);
    if (flux[0] > 0.0 && near) || (flux[0] < 0.0 && far) {
        return ([0.0; 3], wave);
    }
    if far && !near {
        *absorbed += flux[0];
    } else if near && !far {
        *absorbed -= flux[0];
    }
    (flux, wave)
}

/// The flux through a face on the square's edge whose inside stands as
/// `face`, `inward` the way into the square across it (+1 or -1): water
/// leaves as it runs, the inside standing on both sides, and none comes in;
/// what leaves adds to `out`.
fn edge(face: [f64; 4], inward: f64, out: &mut f64) -> ([f64; 3], f64) {
    let (flux, wave) = flux(face, face);
    if flux[0] * inward > 0.0 {
        return ([0.0; 3], wave);
    }
    *out -= flux[0] * inward;
    (flux, wave)
}

/// e to the power of `-x`, for `x` a step's share of foam's life (well
/// under one): its series to the fourth power.
fn fading(x: f64) -> f64 {
    1.0 - x * (1.0 - x * (0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0))))
}

/// The cube root of `x`, positive: a first guess from its bits refined by
/// Newton's steps, in plain arithmetic.
fn cube_root(x: f64) -> f64 {
    let guess = f64::from_bits(x.to_bits() / 3 + 0x2a9f_7893_782d_a1ce);
    let refine = |y: f64| (2.0 * y + x / (y * y)) * (1.0 / 3.0);
    refine(refine(refine(refine(guess))))
}

/// A cell's water `water` (surface and flow) settled over `ground` after a
/// stage of `dt` seconds: never below the ground, slowed by friction, no
/// faster than [`FASTEST`], reset where it stopped being finite, and held
/// at its lake's level `held` where one holds it.
fn settle(water: [f64; 3], ground: f64, held: Option<f64>, dt: f64) -> [f64; 3] {
    if let Some(level) = held {
        return [level, 0.0, 0.0];
    }
    let mut next = water;
    if !(next[0].abs() < 1.0e20 && next[1].abs() < 1.0e20 && next[2].abs() < 1.0e20) {
        next = [ground, 0.0, 0.0];
    }
    next[0] = greater(next[0], ground);
    let h = next[0] - ground;
    if h <= f64::from(DRY) {
        return [next[0], 0.0, 0.0];
    }
    let mut flow = [next[1], next[2]];
    let length = (flow[0] * flow[0] + flow[1] * flow[1]).sqrt();
    let roughness = f64::from(ROUGHNESS);
    let friction = f64::from(GRAVITY) * roughness * roughness * (length / h) / (h * cube_root(h));
    let slowed = 1.0 + dt * friction;
    flow = [flow[0] / slowed, flow[1] / slowed];
    let speed = (flow[0] * flow[0] + flow[1] * flow[1]).sqrt() / h;
    let fastest = f64::from(FASTEST);
    if speed > fastest {
        let scale = fastest / speed;
        flow = [flow[0] * scale, flow[1] * scale];
    }
    [next[0], flow[0], flow[1]]
}
