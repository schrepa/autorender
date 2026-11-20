use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use cook::{Kind, Refusal, Step};
use engine::math::{cos, sin};
use engine_visible_sets::VisibleSets;
use serde::Deserialize;
use serde_json::{Value, json};

use cook_rays::{Gpu, NONE, Occluders};

pub static VISIBLE_SETS: Kind = Kind {
    name: "visible_sets",
    version: 2,
    run: visible_sets,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetsPlan {
    level: String,
    cell_m: f32,
    eye_inset_m: f32,
    eye_heights_m: Vec<f32>,
    top_m: f32,
    padding_cells: u32,
    face_step_m: f32,
    probe_m: f32,
    target_gap_m: f32,
    ground: String,
    occluders: OccluderPlan,
    see_through: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OccluderPlan {
    names: Vec<String>,
    surfaces: Vec<String>,
    shrink_m: f32,
    #[serde(default)]
    starts_m: BTreeMap<String, f32>,
}

pub struct Occluder {
    pub id: String,
    pub material: String,
    pub opaque: bool,
    pub two_sided: bool,
    pub positions: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

struct Chunk {
    id: String,
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
    occludes: bool,
    two_sided: bool,
}

fn put(out: &mut Vec<u8>, words: impl IntoIterator<Item = [u8; 4]>) {
    for word in words {
        out.extend_from_slice(&word);
    }
}

fn put_text(out: &mut Vec<u8>, text: &str) {
    let length = u32::try_from(text.len()).expect("names_fit_u32");
    put(out, [length.to_le_bytes()]);
    out.extend_from_slice(text.as_bytes());
}

pub fn occluders_file(occluders: &[Occluder]) -> Vec<u8> {
    let count = |held: usize| u32::try_from(held).expect("counts_fit_u32").to_le_bytes();
    let mut out = Vec::new();
    put(&mut out, [count(occluders.len())]);
    for held in occluders {
        put_text(&mut out, &held.id);
        put_text(&mut out, &held.material);
        out.extend_from_slice(&[u8::from(held.opaque), u8::from(held.two_sided)]);
        put(&mut out, [count(held.positions.len())]);
        put(&mut out, held.positions.iter().flatten().map(|value| value.to_le_bytes()));
        put(&mut out, [count(held.indices.len())]);
        put(&mut out, held.indices.iter().map(|value| value.to_le_bytes()));
    }
    out
}

struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, length: usize) -> Result<&'a [u8], String> {
        let (held, rest) = self
            .bytes
            .split_at_checked(length)
            .ok_or("the occluder file ends early")?;
        self.bytes = rest;
        Ok(held)
    }

    fn word(&mut self) -> Result<[u8; 4], String> {
        let held = self.take(4)?;
        Ok([held[0], held[1], held[2], held[3]])
    }

    fn count(&mut self) -> Result<usize, String> {
        Ok(u32::from_le_bytes(self.word()?) as usize)
    }

    fn text(&mut self) -> Result<String, String> {
        let length = self.count()?;
        String::from_utf8(self.take(length)?.to_vec()).map_err(|error| error.to_string())
    }
}

fn chunks(bytes: &[u8], see_through: &[String]) -> Result<Vec<Chunk>, String> {
    let mut reader = Reader { bytes };
    let mut chunks = Vec::new();
    let mut materials = Vec::new();
    for _ in 0..reader.count()? {
        let id = reader.text()?;
        let material = reader.text()?;
        let flags = reader.take(2)?;
        let mut positions = Vec::new();
        for _ in 0..reader.count()? {
            let [x, y, z] = [reader.word()?, reader.word()?, reader.word()?];
            positions.push([x, y, z].map(f32::from_le_bytes));
        }
        let mut indices = Vec::new();
        for _ in 0..reader.count()? {
            indices.push(u32::from_le_bytes(reader.word()?));
        }
        let see_through = see_through.contains(&material);
        chunks.push(Chunk {
            id,
            positions,
            indices,
            occludes: !see_through && flags[0] == 1,
            two_sided: flags[1] == 1,
        });
        materials.push(material);
    }
    if !reader.bytes.is_empty() {
        return Err("the occluder file runs on past its chunks".to_owned());
    }
    match see_through.iter().find(|wanted| !materials.contains(wanted)) {
        Some(wanted) => Err(format!(
            "see_through `{wanted}`: no drawn material has this name"
        )),
        None => Ok(chunks),
    }
}

fn visible_sets(step: &mut Step<'_>) -> Result<(), Refusal> {
    let plan: SetsPlan = step.params()?;
    let refuse = |detail: String| Refusal::new("VisibleSets", &plan.level, detail);
    let frame = step.input("frame.json")?;
    let data: Value =
        serde_json::from_slice(frame.bytes).map_err(|error| refuse(error.to_string()))?;
    let held = step.input(&format!("{}.occluders.bin", plan.level))?;
    let chunks = chunks(held.bytes, &plan.see_through).map_err(refuse)?;
    let built = build(&plan, &data, &chunks).map_err(refuse)?;
    let file = format!("{}.sets.bin", plan.level);
    let id = format!("{}_sets", plan.level);
    step.write(&file, built.bytes);
    step.item("entities", &id, json!({"id": id, "level_sets": {"file": file}}));
    step.note(json!({ "visible_sets": built.summary }));
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    low: [f32; 3],
    high: [f32; 3],
}

impl Bounds {
    fn empty() -> Self {
        Self {
            low: [f32::INFINITY; 3],
            high: [f32::NEG_INFINITY; 3],
        }
    }

    fn grow(&mut self, point: [f32; 3]) {
        for ((low, high), value) in self.low.iter_mut().zip(&mut self.high).zip(point) {
            if value < *low {
                *low = value;
            }
            if value > *high {
                *high = value;
            }
        }
    }

    fn join(&mut self, other: &Self) {
        self.grow(other.low);
        self.grow(other.high);
    }

    fn overlaps(&self, other: &Self) -> bool {
        (0..3).all(|axis| self.low[axis] <= other.high[axis] && other.low[axis] <= self.high[axis])
    }

    fn holds(&self, point: [f32; 3]) -> bool {
        (0..3).all(|axis| self.low[axis] <= point[axis] && point[axis] <= self.high[axis])
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn crosses(from: [f32; 3], to: [f32; 3], held: &Bounds) -> bool {
    let (mut near, mut far) = (0.0_f32, 1.0_f32);
    for axis in 0..3 {
        let step = to[axis] - from[axis];
        if step == 0.0 {
            if from[axis] <= held.low[axis] || from[axis] >= held.high[axis] {
                return false;
            }
            continue;
        }
        let (mut enter, mut leave) = (
            (held.low[axis] - from[axis]) / step,
            (held.high[axis] - from[axis]) / step,
        );
        if enter > leave {
            std::mem::swap(&mut enter, &mut leave);
        }
        if enter > near {
            near = enter;
        }
        if leave < far {
            far = leave;
        }
        if near >= far {
            return false;
        }
    }
    true
}

enum Solid {
    Block(Bounds),
    Prism { corners: Vec<[f32; 2]>, z: [f32; 2] },
}

impl Solid {
    fn holds(&self, point: [f32; 3]) -> bool {
        match self {
            Self::Block(bounds) => bounds.holds(point),
            Self::Prism { corners, z } => {
                if !(z[0] <= point[2] && point[2] <= z[1]) {
                    return false;
                }
                let mut sign = 0.0_f32;
                for (at, corner) in corners.iter().enumerate() {
                    let next = corners[(at + 1) % corners.len()];
                    let side = (next[0] - corner[0]) * (point[1] - corner[1])
                        - (next[1] - corner[1]) * (point[0] - corner[0]);
                    if sign != 0.0 && side * sign < 0.0 {
                        return false;
                    }
                    if side != 0.0 {
                        sign = side;
                    }
                }
                true
            }
        }
    }
}

struct Floor {
    rect: [f32; 4],
    height: Option<f32>,
    ramp: Option<(char, f32, f32)>,
}

impl Floor {
    fn at(&self, x: f32, y: f32) -> Option<f32> {
        let [x0, y0, x1, y1] = self.rect;
        if !(x0 <= x && x <= x1 && y0 <= y && y <= y1) {
            return None;
        }
        if let Some(height) = self.height {
            return Some(height);
        }
        let (up, from, to) = self.ramp?;
        let along = match up {
            'x' => (x - x0) / (x1 - x0),
            'X' => (x1 - x) / (x1 - x0),
            'y' => (y - y0) / (y1 - y0),
            _ => (y1 - y) / (y1 - y0),
        };
        Some(from + (to - from) * along)
    }
}

struct World {
    origin: [f32; 2],
    cells: [u32; 2],
    ground: f32,
    floors: Vec<Floor>,
    solids: Vec<Solid>,
    occluders: Vec<Bounds>,
    occluder_names: usize,
}

fn number(value: &Value, at: &str) -> Result<f32, String> {
    value
        .as_f64()
        .map(|held| held as f32)
        .ok_or_else(|| format!("{at} is not a number"))
}

fn numbers<const N: usize>(value: &Value, at: &str) -> Result<[f32; N], String> {
    let list = value
        .as_array()
        .filter(|list| list.len() == N)
        .ok_or_else(|| format!("{at} is not a list of {N} numbers"))?;
    let mut out = [0.0; N];
    for (slot, held) in out.iter_mut().zip(list) {
        *slot = number(held, at)?;
    }
    Ok(out)
}

fn matches(pattern: &str, name: &str) -> bool {
    if let Some(prefix) = pattern.strip_suffix('*') {
        return name.starts_with(prefix);
    }
    if let Some(prefix) = pattern.strip_suffix('#') {
        return name.strip_prefix(prefix).is_some_and(|rest| {
            !rest.is_empty() && rest.bytes().all(|byte| byte.is_ascii_digit())
        });
    }
    pattern == name
}

fn game(low: [f32; 3], high: [f32; 3]) -> Bounds {
    Bounds {
        low: [low[0], low[2], -high[1]],
        high: [high[0], high[2], -low[1]],
    }
}

fn floor_of(held: &Value, at: &str) -> Result<Vec<Floor>, String> {
    let rect = numbers::<4>(&held["rect"], &format!("{at}.rect"))?;
    let mut out = vec![Floor {
        rect,
        height: Some(number(&held["floor"], &format!("{at}.floor"))?),
        ramp: None,
    }];
    let ramp = &held["ramp"];
    if !ramp.is_null() {
        let up = match ramp["up"].as_str() {
            Some("+x") => 'x',
            Some("-x") => 'X',
            Some("+y") => 'y',
            Some("-y") => 'Y',
            other => return Err(format!("{at}.ramp.up is {other:?}, not +x, -x, +y or -y")),
        };
        out.push(Floor {
            rect: numbers::<4>(&ramp["rect"], &format!("{at}.ramp.rect"))?,
            height: None,
            ramp: Some((
                up,
                number(&ramp["from"], &format!("{at}.ramp.from"))?,
                number(&ramp["to"], &format!("{at}.ramp.to"))?,
            )),
        });
    }
    Ok(out)
}

fn world(plan: &SetsPlan, data: &Value) -> Result<World, String> {
    let frame = data["frame"].as_str().unwrap_or_default();
    if !frame.starts_with("x east, y north, z up") {
        return Err(format!(
            "frame `{frame}`: the sets read the data's boxes as x east, y north, z up"
        ));
    }
    let bounds = numbers::<4>(&data["bounds"], "bounds")?;
    let cells = [bounds[2] - bounds[0], bounds[3] - bounds[1]]
        .map(|span| (span / plan.cell_m).ceil() as u32);
    let mut floors = Vec::new();
    for (at, held) in data["raised"].as_array().into_iter().flatten().enumerate() {
        floors.extend(floor_of(held, &format!("raised[{at}]"))?);
    }
    let mut ground = None;
    let mut solids = Vec::new();
    let mut occluders = Vec::new();
    let mut occluder_names = 0;
    let mut started: Vec<String> = Vec::new();
    let listed = data["collision"]
        .as_array()
        .ok_or("the data holds no `collision` list")?;
    for (at, held) in listed.iter().enumerate() {
        let name = held["name"].as_str().unwrap_or_default();
        let place = format!("collision[{at}] `{name}`");
        match held["type"].as_str() {
            Some("box") => {
                let low = numbers::<3>(&held["min"], &format!("{place}.min"))?;
                let high = numbers::<3>(&held["max"], &format!("{place}.max"))?;
                if name == plan.ground {
                    ground = Some(high[2]);
                    continue;
                }
                solids.push(Solid::Block(Bounds { low, high }));
                let surface = held["surface"].as_str().unwrap_or_default();
                if plan
                    .occluders
                    .names
                    .iter()
                    .any(|pattern| matches(pattern, name))
                    && plan.occluders.surfaces.iter().any(|held| held == surface)
                {
                    occluder_names += 1;
                    let mut shrunk = game(low, high);
                    if let Some(start) = plan.occluders.starts_m.get(name) {
                        started.push(name.to_owned());
                        if *start > shrunk.low[1] {
                            shrunk.low[1] = *start;
                        }
                    }
                    for axis in [0, 2] {
                        shrunk.low[axis] += plan.occluders.shrink_m;
                        shrunk.high[axis] -= plan.occluders.shrink_m;
                    }
                    if (0..3).all(|axis| shrunk.low[axis] < shrunk.high[axis]) {
                        occluders.push(shrunk);
                    }
                }
            }
            Some("prism") => {
                let centre = numbers::<2>(&held["center"], &format!("{place}.center"))?;
                let radius = number(&held["radius"], &format!("{place}.radius"))?;
                let rotation = number(&held["rot"], &format!("{place}.rot"))?;
                let z = numbers::<2>(&held["z"], &format!("{place}.z"))?;
                let sides = held["sides"]
                    .as_u64()
                    .filter(|sides| *sides >= 3)
                    .ok_or_else(|| format!("{place}.sides is not a count of 3 or more"))?;
                let corners = (0..sides)
                    .map(|side| {
                        let angle = rotation + std::f32::consts::TAU * side as f32 / sides as f32;
                        [
                            centre[0] + radius * cos(angle),
                            centre[1] + radius * sin(angle),
                        ]
                    })
                    .collect();
                solids.push(Solid::Prism { corners, z });
            }
            Some("ramp") => {}
            other => return Err(format!("{place}.type is {other:?}")),
        }
    }
    let ground = ground.ok_or_else(|| format!("no collision box is named `{}`", plan.ground))?;
    if let Some(unused) = plan
        .occluders
        .starts_m
        .keys()
        .find(|name| !started.contains(name))
    {
        return Err(format!(
            "occluders.starts_m names `{unused}`, which is not one of the occluder boxes"
        ));
    }
    Ok(World {
        origin: [bounds[0], -bounds[3]],
        cells,
        ground,
        floors,
        solids,
        occluders,
        occluder_names,
    })
}

fn checked(plan: &SetsPlan) -> Result<(), String> {
    let positive = |value: f32| value.is_finite() && value > 0.0;
    let wrong = [
        ("cell_m", positive(plan.cell_m)),
        ("top_m", positive(plan.top_m)),
        ("face_step_m", positive(plan.face_step_m)),
        ("probe_m", positive(plan.probe_m)),
        ("target_gap_m", positive(plan.target_gap_m)),
        (
            "eye_inset_m",
            plan.eye_inset_m.is_finite()
                && plan.eye_inset_m >= 0.0
                && plan.eye_inset_m * 2.0 < plan.cell_m,
        ),
        (
            "eye_heights_m",
            !plan.eye_heights_m.is_empty()
                && plan
                    .eye_heights_m
                    .iter()
                    .all(|height| positive(*height) && *height <= plan.top_m),
        ),
        (
            "occluders.shrink_m",
            plan.occluders.shrink_m.is_finite() && plan.occluders.shrink_m >= 0.0,
        ),
    ];
    match wrong.iter().find(|(_, good)| !good) {
        Some((name, _)) => Err(format!("visible_sets.{name} is out of range")),
        None => Ok(()),
    }
}

fn workers<T: Send>(count: usize, work: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let next = AtomicUsize::new(0);
    let (work, next) = (&work, &next);
    let mut done: Vec<(usize, T)> = std::thread::scope(|scope| {
        let parts: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(move || {
                    let mut held = Vec::new();
                    loop {
                        let at = next.fetch_add(1, Ordering::Relaxed);
                        if at >= count {
                            break held;
                        }
                        held.push((at, work(at)));
                    }
                })
            })
            .collect();
        parts
            .into_iter()
            .flat_map(|part| {
                part.join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect()
    });
    done.sort_by_key(|(at, _)| *at);
    done.into_iter().map(|(_, held)| held).collect()
}

fn face_points(bounds: &Bounds, step: f32) -> Vec<[f32; 3]> {
    let size = sub(bounds.high, bounds.low);
    let marks = |axis: usize| -> Vec<f32> {
        let parts = (size[axis] / step).ceil() as usize;
        if parts == 0 {
            return vec![bounds.low[axis]];
        }
        (0..=parts)
            .map(|at| bounds.low[axis] + size[axis] * at as f32 / parts as f32)
            .collect()
    };
    let mut points = Vec::new();
    for axis in 0..3 {
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        for side in [bounds.low[axis], bounds.high[axis]] {
            for &along in &marks(u) {
                for &across in &marks(v) {
                    let mut point = [0.0; 3];
                    point[axis] = side;
                    point[u] = along;
                    point[v] = across;
                    points.push(point);
                }
            }
        }
    }
    points
}

struct Cell {
    floor: f32,
    eyes: Vec<(usize, [f32; 3])>,
}

fn cell_of(plan: &SetsPlan, world: &World, x: u32, z: u32) -> Cell {
    let low = [
        world.origin[0] + x as f32 * plan.cell_m,
        world.origin[1] + z as f32 * plan.cell_m,
    ];
    let half = plan.cell_m / 2.0;
    let (near, far) = (plan.eye_inset_m, plan.cell_m - plan.eye_inset_m);
    let centre = [low[0] + half, low[1] + half];
    let (bx, by) = (centre[0], -centre[1]);
    let mut heights: Vec<f32> = std::iter::once(world.ground)
        .chain(world.floors.iter().filter_map(|floor| floor.at(bx, by)))
        .collect();
    heights.sort_by(|a, b| b.total_cmp(a));
    let spots = [
        centre,
        [low[0] + near, low[1] + near],
        [low[0] + far, low[1] + near],
        [low[0] + near, low[1] + far],
        [low[0] + far, low[1] + far],
    ];
    let free = |spot: [f32; 2], height: f32| {
        let held = [spot[0], -spot[1], height];
        !world.solids.iter().any(|solid| solid.holds(held))
    };
    let floor = heights
        .into_iter()
        .find(|height| spots.iter().any(|spot| free(*spot, height + plan.probe_m)))
        .unwrap_or(f32::NAN);
    if floor.is_nan() {
        return Cell {
            floor,
            eyes: Vec::new(),
        };
    }
    let eyes: Vec<(usize, [f32; 3])> = plan
        .eye_heights_m
        .iter()
        .flat_map(|height| spots.map(|spot| [spot[0], floor + height, spot[1]]))
        .enumerate()
        .filter(|(_, eye)| free([eye[0], eye[2]], eye[1]))
        .collect();
    Cell {
        floor: if eyes.is_empty() { f32::NAN } else { floor },
        eyes,
    }
}

struct Addition {
    cell: [u32; 2],
    chunk: usize,
    eye: usize,
    eye_at: [f32; 3],
    triangle: usize,
    centre: [f32; 3],
}

struct Built {
    bytes: Vec<u8>,
    summary: Value,
}

fn by_boxes(cell: &Cell, world: &World, targets: &[(Bounds, Vec<[f32; 3]>)]) -> (Vec<bool>, u64) {
    let mut rays = 0_u64;
    let mut hull = Bounds::empty();
    for (_, eye) in &cell.eyes {
        hull.grow(*eye);
    }
    let kept = targets
        .iter()
        .map(|(bounds, points)| {
            if cell.eyes.is_empty() {
                return false;
            }
            let mut reach = hull;
            reach.join(bounds);
            let near: Vec<&Bounds> = world
                .occluders
                .iter()
                .filter(|held| held.overlaps(&reach))
                .collect();
            for (_, eye) in &cell.eyes {
                for point in points {
                    rays += 1;
                    if !near.iter().any(|held| crosses(*eye, *point, held)) {
                        return true;
                    }
                }
            }
            false
        })
        .collect();
    (kept, rays)
}

fn by_triangles(
    (plan, gpu, occluders): (&SetsPlan, &Gpu, &Occluders),
    (chunks, cells, kept): (&[Chunk], &[Cell], &[Vec<bool>]),
    place: impl Fn(usize) -> [u32; 2],
) -> Result<(Vec<Addition>, u64), String> {
    let mut centres = Vec::new();
    let mut spans = Vec::new();
    for chunk in chunks {
        let start = centres.len();
        for triangle in chunk.indices.as_chunks::<3>().0 {
            let [a, b, c] = triangle.map(|corner| chunk.positions[corner as usize]);
            centres.push([0, 1, 2].map(|axis| (a[axis] + b[axis] + c[axis]) / 3.0));
        }
        spans.push(start..centres.len());
    }
    let mut eyes = Vec::new();
    let mut first_eye = Vec::new();
    for cell in cells {
        first_eye.push(eyes.len());
        eyes.extend(cell.eyes.iter().map(|(_, eye)| *eye));
    }
    let (mut work, mut items) = (Vec::new(), Vec::new());
    for (chunk, span) in spans.iter().enumerate() {
        for (at, cell) in cells.iter().enumerate() {
            if kept[at][chunk] || span.is_empty() {
                continue;
            }
            for eye in 0..cell.eyes.len() {
                work.push((at, chunk, eye));
                items.push(
                    [first_eye[at] + eye, span.start, span.len()]
                        .map(|value| u32::try_from(value).expect("work_fits_u32")),
                );
            }
        }
    }
    let answers = gpu.first_unblocked(occluders, &centres, &eyes, &items, plan.target_gap_m)?;
    let mut rays = 0_u64;
    let mut found: Vec<Addition> = Vec::new();
    for ((at, chunk, eye), (item, answer)) in work.into_iter().zip(items.iter().zip(answers)) {
        rays += u64::from(if answer == NONE { item[2] } else { answer + 1 });
        let cell = place(at);
        if answer == NONE
            || found
                .last()
                .is_some_and(|held| held.cell == cell && held.chunk == chunk)
        {
            continue;
        }
        let (eye, eye_at) = cells[at].eyes[eye];
        let triangle = answer as usize;
        found.push(Addition {
            cell,
            chunk,
            eye,
            eye_at,
            triangle,
            centre: centres[spans[chunk].start + triangle],
        });
    }
    found.sort_by_key(|held| (held.cell[1], held.cell[0], held.chunk));
    Ok((found, rays))
}

fn padded(
    world: &World,
    cells: &[Cell],
    boxed: &[(Vec<bool>, u64)],
    padding: u32,
) -> Vec<Vec<bool>> {
    let [cells_x, cells_z] = world.cells.map(i64::from);
    let reach = i64::from(padding);
    let width = boxed.first().map_or(0, |(row, _)| row.len());
    (0..cells.len())
        .map(|at| {
            let (x, z) = (at as i64 % cells_x, at as i64 / cells_x);
            let mut row = vec![false; width];
            if cells[at].eyes.is_empty() {
                return row;
            }
            for dz in -reach..=reach {
                for dx in -reach..=reach {
                    let (nx, nz) = (x + dx, z + dz);
                    if nx < 0 || nz < 0 || nx >= cells_x || nz >= cells_z {
                        continue;
                    }
                    let (own, _) = &boxed[(nz * cells_x + nx) as usize];
                    for (slot, held) in row.iter_mut().zip(own) {
                        *slot |= *held;
                    }
                }
            }
            row
        })
        .collect()
}

struct Counted {
    from_boxes: Vec<usize>,
    box_rays: u64,
    truth_rays: u64,
    occluding: usize,
    adapter: (String, &'static str),
}

fn summary(
    (plan, world, chunks): (&SetsPlan, &World, &[Chunk]),
    (cells, kept): (&[Cell], &[Vec<bool>]),
    additions: &[Addition],
    (counted, file_bytes): (&Counted, usize),
) -> Value {
    let walkable: Vec<usize> = (0..cells.len())
        .filter(|at| !cells[*at].eyes.is_empty())
        .collect();
    let triangles: Vec<usize> = chunks.iter().map(|chunk| chunk.indices.len() / 3).collect();
    let sizes: Vec<(usize, usize)> = walkable
        .iter()
        .map(|at| {
            let row = &kept[*at];
            (
                row.iter().filter(|held| **held).count(),
                row.iter()
                    .zip(&triangles)
                    .filter(|(held, _)| **held)
                    .map(|(_, count)| count)
                    .sum(),
            )
        })
        .collect();
    let share = |total: usize| total as f64 / walkable.len().max(1) as f64;
    let mean = |pick: fn(&(usize, usize)) -> usize| share(sizes.iter().map(pick).sum());
    let largest = |pick: fn(&(usize, usize)) -> usize| sizes.iter().map(pick).max().unwrap_or(0);
    json!({
        "cells": world.cells,
        "origin": world.origin,
        "walkable_cells": walkable.len(),
        "eyes_per_cell": plan.eye_heights_m.len() * 5,
        "chunks": chunks.len(),
        "occluders": {"named": world.occluder_names, "kept_after_shrink": world.occluders.len()},
        "occluding_triangles": counted.occluding,
        "set_chunks": {"mean_from_boxes": share(walkable.iter().map(|at| counted.from_boxes[*at]).sum()), "mean": mean(|size| size.0), "largest": largest(|size| size.0)},
        "set_triangles": {"mean": mean(|size| size.1), "largest": largest(|size| size.1), "map": triangles.iter().sum::<usize>()},
        "box_rays": counted.box_rays,
        "truth_rays": counted.truth_rays,
        "added_pairs": additions.len(),
        "additions": additions.iter().map(|held| json!({"cell": held.cell, "chunk": chunks[held.chunk].id, "eye": held.eye, "eye_position": held.eye_at, "triangle": held.triangle, "triangle_centre": held.centre})).collect::<Vec<Value>>(),
        "file_bytes": file_bytes,
        "adapter": counted.adapter.0,
        "backend": counted.adapter.1,
    })
}

fn place_index(cell: [u32; 2], cells_x: usize) -> usize {
    cell[1] as usize * cells_x + cell[0] as usize
}

fn build(plan: &SetsPlan, data: &Value, chunks: &[Chunk]) -> Result<Built, String> {
    checked(plan)?;
    let gpu = Gpu::new()?;
    let world = world(plan, data)?;
    let cells_x = world.cells[0] as usize;
    let count = cells_x * world.cells[1] as usize;
    let place = |at: usize| [(at % cells_x) as u32, (at / cells_x) as u32];
    let cells: Vec<Cell> = (0..count)
        .map(|at| {
            let [x, z] = place(at);
            cell_of(plan, &world, x, z)
        })
        .collect();
    let targets: Vec<(Bounds, Vec<[f32; 3]>)> = chunks
        .iter()
        .map(|chunk| {
            let mut bounds = Bounds::empty();
            for point in &chunk.positions {
                bounds.grow(*point);
            }
            (bounds, face_points(&bounds, plan.face_step_m))
        })
        .collect();
    let boxed = workers(count, |at| by_boxes(&cells[at], &world, &targets));
    let mut kept = padded(&world, &cells, &boxed, plan.padding_cells);
    let occluders = Occluders::new(
        chunks
            .iter()
            .filter(|chunk| chunk.occludes)
            .flat_map(|chunk| {
                (chunk.indices.as_chunks::<3>().0.iter()).map(|triangle| {
                    (
                        triangle.map(|corner| chunk.positions[corner as usize]),
                        chunk.two_sided,
                    )
                })
            }),
        plan.target_gap_m,
    );
    let (additions, truth_rays) =
        by_triangles((plan, &gpu, &occluders), (chunks, &cells, &kept), place)?;
    let counted = Counted {
        from_boxes: kept
            .iter()
            .map(|row| row.iter().filter(|held| **held).count())
            .collect(),
        box_rays: boxed.iter().map(|(_, rays)| rays).sum(),
        truth_rays,
        occluding: occluders.count(),
        adapter: gpu.adapter(),
    };
    for addition in &additions {
        kept[place_index(addition.cell, cells_x)][addition.chunk] = true;
    }
    let floors: Vec<f32> = cells.iter().map(|cell| cell.floor).collect();
    let ids = chunks.iter().map(|chunk| chunk.id.clone()).collect();
    let bytes = VisibleSets::new(
        world.origin,
        plan.cell_m,
        world.cells,
        plan.top_m,
        ids,
        floors,
        &kept,
    )?
    .write();
    let summary = summary(
        (plan, &world, chunks),
        (&cells, &kept),
        &additions,
        (&counted, bytes.len()),
    );
    Ok(Built { bytes, summary })
}
