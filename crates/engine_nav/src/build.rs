use std::collections::BTreeMap;

use engine_codec::{Error, ErrorCode};
use glam::{Vec2, Vec3};

use super::{BuildReport, Grid, NavMesh, Polygon, Portal, flat, turn};

type EdgeMap = BTreeMap<(u32, u32), Vec<(usize, usize)>>;

fn at(polygon: usize) -> String {
    format!("$.polygons.{polygon}")
}

fn refused(polygon: usize) -> Error {
    Error::new(ErrorCode::InvalidMesh, at(polygon))
}

pub(super) fn assemble(
    positions: &[Vec3],
    lists: Vec<Vec<u32>>,
    weld: f32,
) -> Result<NavMesh, Error> {
    check_indices(positions, &lists)?;
    let (vertices, remap) = weld_vertices(positions, weld);
    let mut lists = remapped(&lists, &remap);
    orient(&vertices, &mut lists, weld)?;
    split_t_junctions(&vertices, &mut lists, weld);
    for (index, list) in lists.iter().enumerate() {
        if !convex(&vertices, list, weld) {
            return Err(refused(index).got("not_convex"));
        }
    }
    let mut report = BuildReport::default();
    let (mut polygons, portals) = link(&vertices, lists, &mut report);
    report.islands = mark_islands(&mut polygons, &portals);
    let grid = grid(&polygons);
    Ok(NavMesh {
        vertices,
        polygons,
        portals,
        grid,
        weld,
        report,
    })
}

pub(super) fn assemble_triangles(
    positions: &[Vec3],
    indices: &[u32],
    weld: f32,
) -> Result<NavMesh, Error> {
    let lists: Vec<Vec<u32>> = indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|t| t.to_vec())
        .collect();
    check_indices(positions, &lists)?;
    let (vertices, remap) = weld_vertices(positions, weld);
    let mut lists = remapped(&lists, &remap);
    let triangles = lists.len();
    lists.retain(|list| list.len() == 3 && signed_area(&vertices, list).abs() > weld * weld);
    let empty_triangles = triangles - lists.len();
    orient(&vertices, &mut lists, weld)?;
    let merged = merge_convex(&vertices, lists, weld);
    let mut mesh = assemble(&vertices, merged, weld)?;
    mesh.report.empty_triangles = empty_triangles;
    Ok(mesh)
}

fn check_indices(positions: &[Vec3], lists: &[Vec<u32>]) -> Result<(), Error> {
    for (polygon, list) in lists.iter().enumerate() {
        if let Some(&index) = list.iter().find(|&&i| i as usize >= positions.len()) {
            return Err(Error::new(ErrorCode::OutOfRange, at(polygon)).got(index));
        }
    }
    Ok(())
}

fn weld_vertices(positions: &[Vec3], weld: f32) -> (Vec<Vec3>, Vec<u32>) {
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut by_x: Vec<(f32, u32)> = Vec::new();
    let mut remap = Vec::with_capacity(positions.len());
    for &position in positions {
        let low = by_x.partition_point(|&(x, _)| x < position.x - weld);
        let high = by_x.partition_point(|&(x, _)| x <= position.x + weld);
        let found = by_x[low..high]
            .iter()
            .map(|&(_, id)| id)
            .filter(|&id| vertices[id as usize].distance(position) <= weld)
            .min();
        let id = found.unwrap_or_else(|| {
            let id = vertices.len() as u32;
            vertices.push(position);
            let at = by_x.partition_point(|&(x, _)| x <= position.x);
            by_x.insert(at, (position.x, id));
            id
        });
        remap.push(id);
    }
    (vertices, remap)
}

fn remapped(lists: &[Vec<u32>], remap: &[u32]) -> Vec<Vec<u32>> {
    lists
        .iter()
        .map(|list| {
            let mut out: Vec<u32> = Vec::with_capacity(list.len());
            for &index in list {
                let id = remap[index as usize];
                if out.last() != Some(&id) {
                    out.push(id);
                }
            }
            while out.len() > 1 && out.first() == out.last() {
                out.pop();
            }
            out
        })
        .collect()
}

fn signed_area(vertices: &[Vec3], list: &[u32]) -> f32 {
    let mut twice = 0.0;
    for (i, &a) in list.iter().enumerate() {
        let b = list[(i + 1) % list.len()];
        twice += flat(vertices[a as usize]).perp_dot(flat(vertices[b as usize]));
    }
    twice * 0.5
}

fn orient(vertices: &[Vec3], lists: &mut [Vec<u32>], weld: f32) -> Result<(), Error> {
    for (polygon, list) in lists.iter_mut().enumerate() {
        if list.len() < 3 {
            return Err(refused(polygon).got("degenerate"));
        }
        let area = signed_area(vertices, list);
        if area.abs() <= weld * weld {
            return Err(refused(polygon).got("degenerate"));
        }
        if area < 0.0 {
            list.reverse();
        }
    }
    Ok(())
}

fn split_t_junctions(vertices: &[Vec3], lists: &mut [Vec<u32>], weld: f32) {
    let mut used: Vec<u32> = lists.iter().flatten().copied().collect();
    used.sort_unstable();
    used.dedup();
    used.sort_by(|&a, &b| {
        vertices[a as usize]
            .x
            .total_cmp(&vertices[b as usize].x)
            .then(a.cmp(&b))
    });
    for list in lists.iter_mut() {
        let mut out = Vec::with_capacity(list.len());
        for i in 0..list.len() {
            let a = list[i];
            let b = list[(i + 1) % list.len()];
            out.push(a);
            out.extend(vertices_on_edge(vertices, &used, list, a, b, weld));
        }
        *list = out;
    }
}

fn vertices_on_edge(
    vertices: &[Vec3],
    used: &[u32],
    list: &[u32],
    a: u32,
    b: u32,
    weld: f32,
) -> Vec<u32> {
    let pa = vertices[a as usize];
    let pb = vertices[b as usize];
    let along = pb - pa;
    let length = along.length();
    let low =
        used.partition_point(|&v| vertices[v as usize].x < engine_math::min(pa.x, pb.x) - weld);
    let high =
        used.partition_point(|&v| vertices[v as usize].x <= engine_math::max(pa.x, pb.x) + weld);
    let mut found: Vec<(f32, u32)> = used[low..high]
        .iter()
        .filter(|v| !list.contains(v))
        .filter_map(|&v| {
            let point = vertices[v as usize];
            let t = (point - pa).dot(along) / (length * length);
            let inside = t * length > weld && (1.0 - t) * length > weld;
            (inside && (pa + along * t).distance(point) <= weld).then_some((t, v))
        })
        .collect();
    found.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
    found.into_iter().map(|(_, v)| v).collect()
}

fn convex(vertices: &[Vec3], list: &[u32], weld: f32) -> bool {
    let count = list.len();
    (0..count).all(|i| {
        let previous = flat(vertices[list[(i + count - 1) % count] as usize]);
        let here = flat(vertices[list[i] as usize]);
        let next = flat(vertices[list[(i + 1) % count] as usize]);
        let tolerance = weld * (here.distance(previous) + here.distance(next));
        turn(previous, here, next) >= -tolerance
    })
}

fn edge_map(lists: &[Vec<u32>]) -> EdgeMap {
    let mut map = EdgeMap::new();
    for (polygon, list) in lists.iter().enumerate() {
        for edge in 0..list.len() {
            let a = list[edge];
            let b = list[(edge + 1) % list.len()];
            map.entry((a.min(b), a.max(b)))
                .or_default()
                .push((polygon, edge));
        }
    }
    map
}

fn link(
    vertices: &[Vec3],
    lists: Vec<Vec<u32>>,
    report: &mut BuildReport,
) -> (Vec<Polygon>, Vec<Portal>) {
    let map = edge_map(&lists);
    let mut links: Vec<Vec<Option<u32>>> = lists.iter().map(|l| vec![None; l.len()]).collect();
    let mut walls = vec![false; vertices.len()];
    let mut joins = Vec::new();
    for (&(low, high), users) in &map {
        let opposite = match users.as_slice() {
            [first, second] => {
                let start = |&(p, e): &(usize, usize)| lists[p][e];
                first.0 != second.0 && start(first) != start(second)
            }
            _ => false,
        };
        if opposite {
            joins.push((users[0], users[1]));
        } else {
            walls[low as usize] = true;
            walls[high as usize] = true;
        }
    }
    let mut portals = Vec::with_capacity(joins.len());
    for ((p0, e0), (p1, e1)) in joins {
        let a = lists[p0][e0];
        let b = lists[p0][(e0 + 1) % lists[p0].len()];
        let id = portals.len() as u32;
        links[p0][e0] = Some(id);
        links[p1][e1] = Some(id);
        let (pa, pb) = (vertices[a as usize], vertices[b as usize]);
        portals.push(Portal {
            a: pa,
            b: pb,
            polygons: [p0 as u32, p1 as u32],
            width: pa.distance(pb),
            walls: [walls[a as usize], walls[b as usize]],
        });
    }
    report.portals = portals.len();
    let polygons = lists
        .into_iter()
        .zip(links)
        .map(|(corners, links)| shape(vertices, corners, links))
        .collect();
    (polygons, portals)
}

fn shape(vertices: &[Vec3], corners: Vec<u32>, links: Vec<Option<u32>>) -> Polygon {
    let points: Vec<Vec3> = corners.iter().map(|&v| vertices[v as usize]).collect();
    let center = points.iter().copied().sum::<Vec3>() / points.len() as f32;
    let mut normal = Vec3::ZERO;
    for (i, &a) in points.iter().enumerate() {
        let b = points[(i + 1) % points.len()];
        normal += (a - center).cross(b - center);
    }
    let normal = if normal.y < 0.0 { -normal } else { normal }.normalize();
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    let mut low = f32::INFINITY;
    let mut high = f32::NEG_INFINITY;
    for point in &points {
        min = min.min(flat(*point));
        max = max.max(flat(*point));
        low = engine_math::min(low, point.y);
        high = engine_math::max(high, point.y);
    }
    Polygon {
        corners,
        links,
        normal,
        offset: normal.dot(center),
        min,
        max,
        low,
        high,
        island: u32::MAX,
    }
}

fn mark_islands(polygons: &mut [Polygon], portals: &[Portal]) -> usize {
    let mut islands = 0;
    for seed in 0..polygons.len() {
        if polygons[seed].island != u32::MAX {
            continue;
        }
        let id = islands as u32;
        let mut stack = vec![seed];
        polygons[seed].island = id;
        while let Some(current) = stack.pop() {
            let next: Vec<usize> = polygons[current]
                .links
                .iter()
                .flatten()
                .map(|&portal| portals[portal as usize].other(current as u32) as usize)
                .collect();
            for other in next {
                if polygons[other].island == u32::MAX {
                    polygons[other].island = id;
                    stack.push(other);
                }
            }
        }
        islands += 1;
    }
    islands
}

fn grid(polygons: &[Polygon]) -> Grid {
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    let mut spans = 0.0;
    for polygon in polygons {
        min = min.min(polygon.min);
        max = max.max(polygon.max);
        spans += (polygon.max - polygon.min).max_element();
    }
    let cell = spans / polygons.len().max(1) as f32;
    let cell = if cell > 0.0 { cell } else { 1.0 };
    let columns = ((max.x - min.x) / cell).floor() as usize + 1;
    let rows = ((max.y - min.y) / cell).floor() as usize + 1;
    let mut grid = Grid {
        origin: min,
        cell,
        columns,
        rows,
        starts: vec![0; columns * rows + 1],
        items: Vec::new(),
    };
    let mut cells: Vec<Vec<u32>> = vec![Vec::new(); columns * rows];
    for (id, polygon) in polygons.iter().enumerate() {
        let (c0, r0) = grid.cell(polygon.min);
        let (c1, r1) = grid.cell(polygon.max);
        for row in r0..=r1 {
            for column in c0..=c1 {
                cells[row * columns + column].push(id as u32);
            }
        }
    }
    for (index, cell) in cells.into_iter().enumerate() {
        grid.items.extend(cell);
        grid.starts[index + 1] = grid.items.len() as u32;
    }
    grid
}

impl Grid {
    pub(super) fn cell(&self, point: Vec2) -> (usize, usize) {
        let to_cell = |value: f32, origin: f32, count: usize| {
            (engine_math::max((value - origin) / self.cell, 0.0).floor() as usize).min(count - 1)
        };
        (
            to_cell(point.x, self.origin.x, self.columns),
            to_cell(point.y, self.origin.y, self.rows),
        )
    }
}

fn merge_convex(vertices: &[Vec3], lists: Vec<Vec<u32>>, weld: f32) -> Vec<Vec<u32>> {
    let mut alive: Vec<Option<Vec<u32>>> = lists.into_iter().map(Some).collect();
    loop {
        let current: Vec<Vec<u32>> = alive
            .iter()
            .map(|p| p.clone().unwrap_or_default())
            .collect();
        let map = edge_map(&current);
        let mut touched = vec![false; alive.len()];
        let mut merged_any = false;
        for users in map.values() {
            let [(p0, e0), (p1, e1)] = users.as_slice() else {
                continue;
            };
            let (p0, e0, p1, e1) = (*p0, *e0, *p1, *e1);
            if p0 == p1 || touched[p0] || touched[p1] || alive[p0].is_none() || alive[p1].is_none()
            {
                continue;
            }
            if let Some(union) = try_union(vertices, &current[p0], e0, &current[p1], e1, weld) {
                alive[p0] = Some(union);
                alive[p1] = None;
                touched[p0] = true;
                touched[p1] = true;
                merged_any = true;
            }
        }
        if !merged_any {
            return alive.into_iter().flatten().collect();
        }
    }
}

fn try_union(
    vertices: &[Vec3],
    first: &[u32],
    first_edge: usize,
    second: &[u32],
    second_edge: usize,
    weld: f32,
) -> Option<Vec<u32>> {
    let a = first[first_edge];
    let b = first[(first_edge + 1) % first.len()];
    if second[second_edge] != b || second[(second_edge + 1) % second.len()] != a {
        return None;
    }
    let mut union: Vec<u32> = (0..first.len())
        .map(|i| first[(first_edge + 1 + i) % first.len()])
        .collect();
    union.extend((2..second.len()).map(|i| second[(second_edge + i) % second.len()]));
    if !convex(vertices, &union, weld) {
        return None;
    }
    let plane = shape(vertices, first.to_vec(), vec![None; first.len()]);
    let flat_enough = union
        .iter()
        .all(|&v| (plane.normal.dot(vertices[v as usize]) - plane.offset).abs() <= weld);
    flat_enough.then_some(union)
}
