use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use cook::{Kind, Refusal, Step};
use engine::math::{cos, sin};
use serde::Deserialize;
use serde_json::{Value, json};

const MAGIC: &[u8; 4] = b"LVS1";

pub static VISIBLE_SETS: Kind = Kind {
    name: "visible_sets",
    version: 1,
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

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
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

fn wide(value: [f32; 3]) -> [f64; 3] {
    value.map(f64::from)
}

fn sub_wide(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross_wide(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot_wide(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[derive(Clone, Copy)]
struct Triangle {
    corner: [f64; 3],
    first: [f64; 3],
    second: [f64; 3],
    two_sided: bool,
}

impl Triangle {
    fn hit(&self, from: [f64; 3], step: [f64; 3], reach: f64) -> bool {
        let p = cross_wide(step, self.second);
        let det = dot_wide(self.first, p);
        if det == 0.0 || (!self.two_sided && det < 0.0) {
            return false;
        }
        let inverse = 1.0 / det;
        let s = sub_wide(from, self.corner);
        let u = dot_wide(s, p) * inverse;
        if !(0.0..=1.0).contains(&u) {
            return false;
        }
        let q = cross_wide(s, self.first);
        let v = dot_wide(step, q) * inverse;
        if v < 0.0 || u + v > 1.0 {
            return false;
        }
        let t = dot_wide(self.second, q) * inverse;
        t > 0.0 && t < reach
    }
}

struct Node {
    bounds: Bounds,
    start: u32,
    count: u32,
}

struct Tree {
    nodes: Vec<Node>,
    triangles: Vec<Triangle>,
}

struct Item {
    bounds: Bounds,
    centre: [f32; 3],
    triangle: Triangle,
}

fn build_node(items: &mut [Item], first: usize, nodes: &mut Vec<Node>, at: usize) {
    let mut bounds = Bounds::empty();
    let mut centres = Bounds::empty();
    for item in items.iter() {
        bounds.join(&item.bounds);
        centres.grow(item.centre);
    }
    let count = u32::try_from(items.len()).expect("tree_items_fit_u32");
    let start = u32::try_from(first).expect("tree_items_fit_u32");
    if items.len() <= 4 {
        nodes[at] = Node {
            bounds,
            start,
            count,
        };
        return;
    }
    let size = sub(centres.high, centres.low);
    let axis = if size[0] >= size[1] && size[0] >= size[2] {
        0
    } else if size[1] >= size[2] {
        1
    } else {
        2
    };
    let middle = items.len() / 2;
    items.select_nth_unstable_by(middle, |a, b| a.centre[axis].total_cmp(&b.centre[axis]));
    let left = nodes.len();
    for _ in 0..2 {
        nodes.push(Node {
            bounds,
            start: 0,
            count: 0,
        });
    }
    nodes[at] = Node {
        bounds,
        start: u32::try_from(left).expect("tree_nodes_fit_u32"),
        count: 0,
    };
    let (low, high) = items.split_at_mut(middle);
    build_node(low, first, nodes, left);
    build_node(high, first + middle, nodes, left + 1);
}

impl Tree {
    fn new(chunks: &[Chunk], pad: f32) -> Self {
        let mut items: Vec<Item> = Vec::new();
        for chunk in chunks.iter().filter(|chunk| chunk.occludes) {
            for triangle in chunk.indices.as_chunks::<3>().0 {
                let [a, b, c] = triangle.map(|corner| chunk.positions[corner as usize]);
                let mut bounds = Bounds::empty();
                for point in [a, b, c] {
                    bounds.grow(point);
                }
                for axis in 0..3 {
                    bounds.low[axis] -= pad;
                    bounds.high[axis] += pad;
                }
                let corner = wide(a);
                items.push(Item {
                    bounds,
                    centre: [0, 1, 2].map(|axis| (a[axis] + b[axis] + c[axis]) / 3.0),
                    triangle: Triangle {
                        corner,
                        first: sub_wide(wide(b), corner),
                        second: sub_wide(wide(c), corner),
                        two_sided: chunk.two_sided,
                    },
                });
            }
        }
        let mut nodes = vec![Node {
            bounds: Bounds::empty(),
            start: 0,
            count: 0,
        }];
        if !items.is_empty() {
            build_node(&mut items, 0, &mut nodes, 0);
        }
        Self {
            nodes,
            triangles: items.into_iter().map(|item| item.triangle).collect(),
        }
    }

    fn enters(bounds: &Bounds, from: [f32; 3], inverse: [f32; 3], reach: f32) -> bool {
        let (mut near, mut far) = (0.0_f32, reach);
        for axis in 0..3 {
            let mut enter = (bounds.low[axis] - from[axis]) * inverse[axis];
            let mut leave = (bounds.high[axis] - from[axis]) * inverse[axis];
            if enter > leave {
                std::mem::swap(&mut enter, &mut leave);
            }
            if enter > near {
                near = enter;
            }
            if leave < far {
                far = leave;
            }
        }
        near <= far
    }

    fn blocked(&self, from: [f32; 3], to: [f32; 3], reach: f32) -> bool {
        if self.triangles.is_empty() {
            return false;
        }
        let step = sub(to, from);
        let inverse = step.map(|value| 1.0 / value);
        let (start, ray, limit) = (wide(from), sub_wide(wide(to), wide(from)), f64::from(reach));
        let mut stack = [0_u32; 64];
        let mut depth = 1;
        while depth > 0 {
            depth -= 1;
            let node = &self.nodes[stack[depth] as usize];
            if !Self::enters(&node.bounds, from, inverse, reach) {
                continue;
            }
            if node.count == 0 {
                stack[depth] = node.start;
                stack[depth + 1] = node.start + 1;
                depth += 2;
                continue;
            }
            let first = node.start as usize;
            let held = &self.triangles[first..first + node.count as usize];
            if held.iter().any(|triangle| triangle.hit(start, ray, limit)) {
                return true;
            }
        }
        false
    }
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
    (plan, tree, chunks): (&SetsPlan, &Tree, &[Chunk]),
    cell: &Cell,
    at: [u32; 2],
    kept: &[bool],
) -> (Vec<Addition>, u64) {
    let mut rays = 0_u64;
    let mut found = Vec::new();
    for (index, chunk) in chunks.iter().enumerate() {
        if kept[index] {
            continue;
        }
        'chunk: for &(eye_at, eye) in &cell.eyes {
            for (triangle_at, triangle) in chunk.indices.as_chunks::<3>().0.iter().enumerate() {
                let [a, b, c] = triangle.map(|corner| chunk.positions[corner as usize]);
                let centre = [0, 1, 2].map(|axis| (a[axis] + b[axis] + c[axis]) / 3.0);
                let length = dot(sub(centre, eye), sub(centre, eye)).sqrt();
                rays += 1;
                if length <= plan.target_gap_m
                    || !tree.blocked(eye, centre, 1.0 - plan.target_gap_m / length)
                {
                    found.push(Addition {
                        cell: at,
                        chunk: index,
                        eye: eye_at,
                        eye_at: eye,
                        triangle: triangle_at,
                        centre,
                    });
                    break 'chunk;
                }
            }
        }
    }
    (found, rays)
}

fn bits(row: &[bool]) -> Vec<u8> {
    let mut out = vec![0_u8; row.len().div_ceil(8)];
    for (at, held) in row.iter().enumerate() {
        if *held {
            out[at / 8] |= 1 << (at % 8);
        }
    }
    out
}

fn written(
    plan: &SetsPlan,
    world: &World,
    chunks: &[Chunk],
    (floors, kept): (&[f32], &[Vec<bool>]),
) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    for value in world.cells {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for value in [world.origin[0], world.origin[1], plan.cell_m, plan.top_m] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    let count = u32::try_from(chunks.len()).map_err(|_| "too many chunks".to_owned())?;
    out.extend_from_slice(&count.to_le_bytes());
    for chunk in chunks {
        let length = u16::try_from(chunk.id.len())
            .map_err(|_| format!("chunk id `{}` is longer than 65535 bytes", chunk.id))?;
        out.extend_from_slice(&length.to_le_bytes());
        out.extend_from_slice(chunk.id.as_bytes());
    }
    out.resize(out.len().div_ceil(4) * 4, 0);
    for floor in floors {
        out.extend_from_slice(&floor.to_le_bytes());
    }
    for row in kept {
        out.extend(bits(row));
    }
    Ok(out)
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

struct Timed {
    started: Instant,
    from_boxes: Vec<usize>,
    box_rays: u64,
    box_seconds: f64,
    truth_rays: u64,
    truth_seconds: f64,
    occluding: usize,
}

fn summary(
    (plan, world, chunks): (&SetsPlan, &World, &[Chunk]),
    (cells, kept): (&[Cell], &[Vec<bool>]),
    additions: &[Addition],
    (timed, file_bytes): (&Timed, usize),
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
        "occluding_triangles": timed.occluding,
        "set_chunks": {"mean_from_boxes": share(walkable.iter().map(|at| timed.from_boxes[*at]).sum()), "mean": mean(|size| size.0), "largest": largest(|size| size.0)},
        "set_triangles": {"mean": mean(|size| size.1), "largest": largest(|size| size.1), "map": triangles.iter().sum::<usize>()},
        "box_rays": timed.box_rays,
        "box_seconds": timed.box_seconds,
        "truth_rays": timed.truth_rays,
        "truth_seconds": timed.truth_seconds,
        "added_pairs": additions.len(),
        "additions": additions.iter().map(|held| json!({"cell": held.cell, "chunk": chunks[held.chunk].id, "eye": held.eye, "eye_position": held.eye_at, "triangle": held.triangle, "triangle_centre": held.centre})).collect::<Vec<Value>>(),
        "file_bytes": file_bytes,
        "seconds": timed.started.elapsed().as_secs_f64(),
    })
}

fn place_index(cell: [u32; 2], cells_x: usize) -> usize {
    cell[1] as usize * cells_x + cell[0] as usize
}

fn build(plan: &SetsPlan, data: &Value, chunks: &[Chunk]) -> Result<Built, String> {
    checked(plan)?;
    let started = Instant::now();
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
    let box_seconds = started.elapsed().as_secs_f64();
    let truth_started = Instant::now();
    let tree = Tree::new(chunks, plan.target_gap_m);
    let checked_rows = workers(count, |at| {
        by_triangles((plan, &tree, chunks), &cells[at], place(at), &kept[at])
    });
    let timed = Timed {
        started,
        from_boxes: kept
            .iter()
            .map(|row| row.iter().filter(|held| **held).count())
            .collect(),
        box_rays: boxed.iter().map(|(_, rays)| rays).sum(),
        box_seconds,
        truth_rays: checked_rows.iter().map(|(_, rays)| rays).sum(),
        truth_seconds: truth_started.elapsed().as_secs_f64(),
        occluding: tree.triangles.len(),
    };
    let mut additions = Vec::new();
    for (at, (found, _)) in checked_rows.into_iter().enumerate() {
        for addition in found {
            kept[at][addition.chunk] = true;
            additions.push(addition);
        }
    }
    if let Some(missing) = additions
        .iter()
        .find(|held| !kept[place_index(held.cell, cells_x)][held.chunk])
    {
        return Err(format!(
            "cell {:?}: chunk `{}` is seen by the triangle check but not kept",
            missing.cell, chunks[missing.chunk].id
        ));
    }
    let floors: Vec<f32> = cells.iter().map(|cell| cell.floor).collect();
    let bytes = written(plan, &world, chunks, (&floors, &kept))?;
    let summary = summary(
        (plan, &world, chunks),
        (&cells, &kept),
        &additions,
        (&timed, bytes.len()),
    );
    Ok(Built { bytes, summary })
}
