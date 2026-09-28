//! Cluster LOD hierarchies from meshoptimizer's `clusterlod` (decision 044):
//! the groups of clusters each level simplifies into the next, every cluster
//! with the group it belongs to and the finer group it came from, and a
//! bounding volume hierarchy over the groups of each level.

use std::ffi::c_int;
use std::ops::Range;

use crate::ffi;

/// Triangles per cluster at most: vertex shaders draw 3 vertices of each.
pub const CLUSTER_TRIANGLES: usize = 128;
/// Vertices per cluster at most, addressed by 8-bit local indices.
pub const CLUSTER_VERTICES: usize = 255;
/// Children per hierarchy node.
pub const NODE_WIDTH: usize = 8;

/// A sphere in mesh space and the simplification error it bounds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bounds {
    pub center: [f32; 3],
    pub radius: f32,
    /// Mesh-space error; `f32::MAX` for groups never simplified further.
    pub error: f32,
}

impl From<ffi::ClodBounds> for Bounds {
    fn from(bounds: ffi::ClodBounds) -> Self {
        Self {
            center: bounds.center,
            radius: bounds.radius,
            error: bounds.error,
        }
    }
}

impl From<Bounds> for ffi::ClodBounds {
    fn from(bounds: Bounds) -> Self {
        Self {
            center: bounds.center,
            radius: bounds.radius,
            error: bounds.error,
        }
    }
}

/// Up to [`CLUSTER_TRIANGLES`] triangles over up to [`CLUSTER_VERTICES`]
/// vertices.
#[derive(Clone, Debug)]
pub struct Cluster {
    /// The group it belongs to: when that group's simplification is precise
    /// enough, a coarser cluster replaces it.
    pub group: u32,
    /// The finer group whose simplification made it (its own error), or
    /// `None` for original geometry.
    pub refined: Option<u32>,
    /// Culling sphere (its error is not monotonic across the hierarchy).
    pub bounds: Bounds,
    /// Normal cone: axis and cutoff (cosine of half its angle) as 8-bit snorm;
    /// a cutoff of 127 disables it.
    pub cone: [i8; 4],
    /// Source vertex of each local vertex.
    pub vertices: Vec<u32>,
    /// Local vertex indices, three per triangle.
    pub triangles: Vec<u8>,
}

/// Clusters simplified together into the next level's clusters.
#[derive(Clone, Debug)]
pub struct Group {
    /// Level of the hierarchy (0 holds the original geometry).
    pub depth: u32,
    /// Bounds and error of its simplification: the error of the clusters it
    /// made, and of every cluster in it when drawn instead of those.
    pub simplified: Bounds,
    /// Its clusters in [`Dag::clusters`].
    pub clusters: Range<u32>,
}

/// A node of the bounding volume hierarchy over one level's groups.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NodeKind {
    Group(u32),
    Children { first: u32, count: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct Node {
    /// Bounds of its subtree, with the largest simplified error in it.
    pub bounds: Bounds,
    pub kind: NodeKind,
}

/// A mesh's cluster LOD hierarchy.
#[derive(Clone, Debug)]
pub struct Dag {
    pub clusters: Vec<Cluster>,
    pub groups: Vec<Group>,
    /// Nodes `0..levels` are the roots of each level's tree.
    pub nodes: Vec<Node>,
    pub levels: u32,
}

/// Triangles with per-vertex normals and optional texture coordinates.
pub struct Source<'a> {
    pub positions: &'a [[f32; 3]],
    pub normals: &'a [[f32; 3]],
    pub uvs: Option<&'a [[f32; 2]]>,
    pub indices: &'a [u32],
}

/// Builds the cluster LOD hierarchy of `source`: whole triangles, a normal
/// (and texture coordinates, if any) per vertex, finite positions, indices in
/// range.
pub fn build(source: &Source<'_>) -> Result<Dag, String> {
    let count = source.positions.len();
    if source.indices.len() < 3 || !source.indices.len().is_multiple_of(3) {
        return Err("indices must hold whole triangles".into());
    }
    if source.normals.len() != count || source.uvs.is_some_and(|uvs| uvs.len() != count) {
        return Err("every vertex needs a normal (and texture coordinates if any)".into());
    }
    if source.indices.iter().any(|&index| index as usize >= count) {
        return Err("an index is out of range".into());
    }
    if !source
        .positions
        .iter()
        .flatten()
        .all(|value| value.is_finite())
    {
        return Err("positions must be finite".into());
    }
    // Normals steer simplification; texture seams stay where they are.
    let attributes: Vec<f32> = (0..count)
        .flat_map(|vertex| {
            let [nx, ny, nz] = source.normals[vertex];
            let [u, v] = source.uvs.map_or([0.0; 2], |uvs| uvs[vertex]);
            [nx, ny, nz, u, v]
        })
        .collect();
    let config = ffi::ClodConfig {
        max_vertices: CLUSTER_VERTICES,
        ..ffi::default_config(CLUSTER_TRIANGLES)
    };
    let locks = border_locks(source);
    let input = ffi::MeshInput {
        positions: source.positions,
        attributes: &attributes,
        attribute_stride: 5,
        weights: &[0.5; 3],
        protect_mask: if source.uvs.is_some() {
            (1 << 3) | (1 << 4)
        } else {
            0
        },
        indices: source.indices,
        locks: &locks,
    };
    let mut clusters: Vec<Cluster> = Vec::new();
    let mut groups: Vec<Group> = Vec::new();
    let completed = ffi::build(config, &input, &mut |group, views| {
        let index = groups.len() as u32;
        let first = clusters.len() as u32;
        for view in views {
            let (vertices, triangles) = ffi::local_indices(view.indices);
            let bounds = ffi::cluster_bounds(view.indices, source.positions);
            let [x, y, z] = bounds.cone_axis_s8;
            clusters.push(Cluster {
                group: index,
                refined: u32::try_from(view.refined).ok(),
                // Tight culling spheres; `clusterlod`'s are its LOD spheres.
                bounds: Bounds {
                    center: bounds.center,
                    radius: bounds.radius,
                    error: 0.0,
                },
                cone: [x, y, z, bounds.cone_cutoff_s8],
                vertices,
                triangles,
            });
        }
        groups.push(Group {
            depth: group.depth.max(0) as u32,
            simplified: group.simplified.into(),
            clusters: first..clusters.len() as u32,
        });
        index as c_int
    });
    if !completed || groups.is_empty() {
        return Err("cluster LOD construction failed".into());
    }
    let levels = groups.iter().map(|group| group.depth).max().unwrap_or(0) + 1;
    let clod_groups: Vec<ffi::ClodGroup> = groups
        .iter()
        .map(|group| ffi::ClodGroup {
            depth: group.depth as c_int,
            simplified: group.simplified.into(),
        })
        .collect();
    let nodes = ffi::build_hierarchy(&clod_groups, NODE_WIDTH, levels as usize)
        .iter()
        .map(|node| Node {
            bounds: node.bounds.into(),
            kind: match u32::try_from(node.group) {
                Ok(group) => NodeKind::Group(group),
                Err(_) => NodeKind::Children {
                    first: node.child_offset,
                    count: node.child_count,
                },
            },
        })
        .collect();
    Ok(Dag {
        clusters,
        groups,
        nodes,
        levels,
    })
}

/// `meshopt_SimplifyVertex_Lock` on every vertex of an open border: an edge
/// of one triangle, compared by position so texture seams are not borders.
/// Borders then stay put at every level, and separately placed meshes that
/// meet there (terrain tiles) never crack apart.
/// Each vertex's first vertex at the same position, and how many triangles
/// use each edge between such positions.
fn edge_uses(source: &Source<'_>) -> (Vec<u32>, std::collections::HashMap<(u32, u32), u32>) {
    let mut first = std::collections::HashMap::new();
    let remap: Vec<u32> = source
        .positions
        .iter()
        .enumerate()
        .map(|(vertex, position)| {
            *first
                .entry(position.map(f32::to_bits))
                .or_insert(vertex as u32)
        })
        .collect();
    let mut edges: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
    for triangle in source.indices.as_chunks::<3>().0 {
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let (a, b) = (remap[triangle[a] as usize], remap[triangle[b] as usize]);
            *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    (remap, edges)
}

/// Whether `source` is closed: every edge between positions joins exactly
/// two triangles, so its surfaces facing a direction hide those facing away.
pub fn is_closed(source: &Source<'_>) -> bool {
    edge_uses(source).1.values().all(|&uses| uses == 2)
}

fn border_locks(source: &Source<'_>) -> Vec<u8> {
    let (remap, edges) = edge_uses(source);
    let mut border = vec![false; source.positions.len()];
    for ((a, b), uses) in edges {
        if uses == 1 {
            border[a as usize] = true;
            border[b as usize] = true;
        }
    }
    remap
        .iter()
        .map(|&position| u8::from(border[position as usize]))
        .collect()
}

/// A viewpoint for selecting clusters: its position in mesh space, `cot(fovy
/// / 2)`, the near plane distance and the largest error, as a fraction of the
/// screen height, a drawn cluster may have.
#[derive(Clone, Copy, Debug)]
pub struct View {
    pub eye: [f32; 3],
    pub projection: f32,
    pub near: f32,
    pub threshold: f32,
}

impl View {
    /// `bounds.error` projected at the sphere's nearest point, as a fraction
    /// of the screen height; monotonic in nested spheres and errors.
    pub fn error(&self, bounds: &Bounds) -> f32 {
        let [x, y, z] = bounds.center;
        let [ex, ey, ez] = self.eye;
        let distance =
            ((x - ex).powi(2) + (y - ey).powi(2) + (z - ez).powi(2)).sqrt() - bounds.radius;
        let distance = if distance > self.near {
            distance
        } else {
            self.near
        };
        bounds.error / distance * (self.projection * 0.5)
    }
}

impl Dag {
    /// The clusters `view` draws: the least detailed of each part of the
    /// surface whose error is at most the threshold (the reference the GPU
    /// selection follows).
    pub fn select(&self, view: &View) -> Vec<u32> {
        let mut selected = Vec::new();
        let mut stack: Vec<u32> = (0..self.levels).collect();
        while let Some(index) = stack.pop() {
            let node = &self.nodes[index as usize];
            if view.error(&node.bounds) <= view.threshold {
                continue;
            }
            match node.kind {
                NodeKind::Children { first, count } => stack.extend(first..first + count),
                NodeKind::Group(group) => {
                    for cluster in self.groups[group as usize].clusters.clone() {
                        let refined = self.clusters[cluster as usize].refined;
                        if refined.is_none_or(|refined| {
                            view.error(&self.groups[refined as usize].simplified) <= view.threshold
                        }) {
                            selected.push(cluster);
                        }
                    }
                }
            }
        }
        selected
    }
}
