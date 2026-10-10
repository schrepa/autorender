//! Packs a cluster LOD hierarchy into the `EVG1` virtual mesh format
//! (`engine_assets::VirtualMesh`): quantized vertices and 8-bit local
//! triangles in pages of [`PAGE_WORDS`], the pinned groups first, then the
//! coarsest levels first.

pub use engine_assets::PAGE_WORDS;
use engine_assets::{
    CLUSTER_WORDS, DEFAULT_FALLBACK_TRIANGLES, FLAG_CLOSED, FLAG_UVS, GROUP_WORDS, HEADER_WORDS,
    NODE_WORDS, VERTEX_WORDS, VIRTUAL_MESH_MAGIC, VIRTUAL_MESH_VERSION, VIRTUAL_NONE,
    octahedral_encode, virtual_mesh_header as header,
};

use crate::dag::{Dag, NodeKind, Source, is_closed};

/// Triangles of the cut at the pin error at most: the pinned pages hold it
/// and every coarser group, which renderers draw before any page streams
/// in, and fallback meshes of up to this many triangles.
pub const PINNED_TRIANGLES: u32 = DEFAULT_FALLBACK_TRIANGLES;

/// The smallest error whose cut (every cluster whose group simplifies with
/// more error and whose own error is at most it) has at most `triangles`
/// triangles; cuts shrink as the error grows.
pub fn pin_error(dag: &Dag, triangles: u32) -> f32 {
    let error = |group: u32| dag.groups[group as usize].simplified.error;
    let cut_triangles = |at: f32| -> u32 {
        dag.clusters
            .iter()
            .filter(|cluster| {
                error(cluster.group) > at
                    && cluster.refined.is_none_or(|refined| error(refined) <= at)
            })
            .map(|cluster| (cluster.triangles.len() / 3) as u32)
            .sum()
    };
    let mut errors: Vec<f32> = dag
        .groups
        .iter()
        .map(|group| group.simplified.error)
        .filter(|error| *error < f32::MAX)
        .collect();
    errors.push(0.0);
    errors.sort_by(f32::total_cmp);
    errors.dedup();
    let index = errors.partition_point(|&at| cut_triangles(at) > triangles);
    errors.get(index).copied().unwrap_or(f32::MAX / 2.0)
}

/// Quantization of one axis: the value of step zero and the step size over
/// `low..=high` in 16 bits.
fn axis(low: f32, high: f32) -> (f32, f32) {
    let step = (high - low) / 65535.0;
    (low, if step > 0.0 { step } else { 0.0 })
}

fn quantize(value: f32, (low, step): (f32, f32)) -> u32 {
    if step > 0.0 {
        let steps = ((value - low) / step).round();
        if steps <= 0.0 {
            0
        } else if steps >= 65535.0 {
            65535
        } else {
            steps as u32
        }
    } else {
        0
    }
}

/// The `EVG1` words of `dag`, built from `source`.
pub fn pack(dag: &Dag, source: &Source<'_>) -> Vec<u32> {
    let mut low = [f32::INFINITY; 3];
    let mut high = [f32::NEG_INFINITY; 3];
    for position in source.positions {
        for axis in 0..3 {
            if position[axis] < low[axis] {
                low[axis] = position[axis];
            }
            if position[axis] > high[axis] {
                high[axis] = position[axis];
            }
        }
    }
    let positions: [(f32, f32); 3] = std::array::from_fn(|index| axis(low[index], high[index]));
    let (mut uv_low, mut uv_high) = ([0.0_f32; 2], [0.0_f32; 2]);
    if let Some(uvs) = source.uvs {
        uv_low = [f32::INFINITY; 2];
        uv_high = [f32::NEG_INFINITY; 2];
        for uv in uvs {
            for axis in 0..2 {
                if uv[axis] < uv_low[axis] {
                    uv_low[axis] = uv[axis];
                }
                if uv[axis] > uv_high[axis] {
                    uv_high[axis] = uv[axis];
                }
            }
        }
    }
    let uvs: [(f32, f32); 2] = std::array::from_fn(|index| axis(uv_low[index], uv_high[index]));

    // Pages: the pinned groups' clusters (every group simplified with more
    // error than the pin error), then the coarsest levels first, whole
    // clusters per page.
    let pin = pin_error(dag, PINNED_TRIANGLES);
    let pinned_group = |group: usize| dag.groups[group].simplified.error > pin;
    let mut order: Vec<usize> = (0..dag.clusters.len()).collect();
    order.sort_by_key(|&cluster| {
        let group = dag.clusters[cluster].group as usize;
        (
            !pinned_group(group),
            std::cmp::Reverse(dag.groups[group].depth),
            group,
            cluster,
        )
    });
    let mut pages: Vec<Vec<u32>> = vec![Vec::new()];
    // Page, vertex offset and triangle offset of each cluster.
    let mut placement = vec![(0u32, 0u32, 0u32); dag.clusters.len()];
    for &index in &order {
        let cluster = &dag.clusters[index];
        let triangle_words = cluster.triangles.len().div_ceil(4);
        let size = cluster.vertices.len() * VERTEX_WORDS + triangle_words;
        if pages
            .last()
            .is_some_and(|page| page.len() + size > PAGE_WORDS)
        {
            pages.push(Vec::new());
        }
        let page = pages.last_mut().expect("pushed above");
        let vertex_offset = page.len() as u32;
        for &vertex in &cluster.vertices {
            let vertex = vertex as usize;
            let position = source.positions[vertex];
            let [x, y, z] = std::array::from_fn(|axis| quantize(position[axis], positions[axis]));
            let [ox, oy] =
                octahedral_encode(source.normals[vertex]).map(|value| value as u16 as u32);
            let uv = source.uvs.map_or([0.0; 2], |uvs| uvs[vertex]);
            let [u, v] = std::array::from_fn(|axis| quantize(uv[axis], uvs[axis]));
            page.extend([x | y << 16, z | ox << 16, oy | u << 16, v]);
        }
        let triangle_offset = page.len() as u32;
        page.extend(cluster.triangles.chunks(4).map(|chunk| {
            let mut bytes = [0u8; 4];
            bytes[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(bytes)
        }));
        placement[index] = ((pages.len() - 1) as u32, vertex_offset, triangle_offset);
    }
    let pinned_pages = order
        .iter()
        .filter(|&&cluster| pinned_group(dag.clusters[cluster].group as usize))
        .map(|&cluster| placement[cluster].0 + 1)
        .max()
        .unwrap_or(1);

    // Depth of the hierarchy: nodes on the longest root-to-leaf path. Roots
    // come first; `clodBuildHierarchy` writes the other nodes bottom-up.
    let mut depth = 0u32;
    let mut stack: Vec<(u32, u32)> = (0..dag.levels).map(|root| (root, 1)).collect();
    while let Some((node, level)) = stack.pop() {
        depth = depth.max(level);
        if let NodeKind::Children { first, count } = dag.nodes[node as usize].kind {
            stack.extend((first..first + count).map(|child| (child, level + 1)));
        }
    }
    let triangles: usize = dag
        .clusters
        .iter()
        .map(|cluster| cluster.triangles.len() / 3)
        .sum();

    let mut words = vec![0u32; HEADER_WORDS];
    words[header::MAGIC] = u32::from_le_bytes(VIRTUAL_MESH_MAGIC);
    words[header::VERSION] = VIRTUAL_MESH_VERSION;
    words[header::FLAGS] = if source.uvs.is_some() { FLAG_UVS } else { 0 }
        | if is_closed(source) { FLAG_CLOSED } else { 0 };
    words[header::LEVELS] = dag.levels;
    words[header::NODES] = dag.nodes.len() as u32;
    words[header::GROUPS] = dag.groups.len() as u32;
    words[header::CLUSTERS] = dag.clusters.len() as u32;
    words[header::PAGES] = pages.len() as u32;
    words[header::TRIANGLES] = triangles as u32;
    words[header::SOURCE_TRIANGLES] = (source.indices.len() / 3) as u32;
    words[header::DEPTH] = depth;
    for axis in 0..3 {
        words[header::POSITION_MIN + axis] = positions[axis].0.to_bits();
        words[header::POSITION_STEP + axis] = positions[axis].1.to_bits();
        words[header::BOUNDS_MIN + axis] = low[axis].to_bits();
        words[header::BOUNDS_MAX + axis] = high[axis].to_bits();
    }
    for axis in 0..2 {
        words[header::UV_MIN + axis] = uvs[axis].0.to_bits();
        words[header::UV_STEP + axis] = uvs[axis].1.to_bits();
    }
    words[header::PINNED_PAGES] = pinned_pages;
    words[header::PIN_ERROR] = pin.to_bits();
    let sphere = |center: [f32; 3], radius: f32, error: f32| {
        [
            center[0].to_bits(),
            center[1].to_bits(),
            center[2].to_bits(),
            radius.to_bits(),
            error.to_bits(),
        ]
    };
    for node in &dag.nodes {
        words.extend(sphere(
            node.bounds.center,
            node.bounds.radius,
            node.bounds.error,
        ));
        words.extend(match node.kind {
            NodeKind::Group(group) => [group, 0, 0],
            NodeKind::Children { first, count } => [VIRTUAL_NONE, first, count],
        });
    }
    debug_assert_eq!(words.len(), HEADER_WORDS + dag.nodes.len() * NODE_WORDS);
    for group in &dag.groups {
        words.extend(sphere(
            group.simplified.center,
            group.simplified.radius,
            group.simplified.error,
        ));
        words.extend([
            group.clusters.start,
            group.clusters.end - group.clusters.start,
            group.depth,
        ]);
    }
    debug_assert_eq!(words.len() % GROUP_WORDS, 0);
    for (index, cluster) in dag.clusters.iter().enumerate() {
        let (page, vertex_offset, triangle_offset) = placement[index];
        let cone = u32::from_le_bytes(cluster.cone.map(|value| value as u8));
        words.extend([
            cluster.bounds.center[0].to_bits(),
            cluster.bounds.center[1].to_bits(),
            cluster.bounds.center[2].to_bits(),
            cluster.bounds.radius.to_bits(),
            cone,
            cluster.refined.unwrap_or(VIRTUAL_NONE),
            cluster.group,
            page,
            vertex_offset,
            triangle_offset,
            cluster.vertices.len() as u32 | ((cluster.triangles.len() / 3) as u32) << 16,
            0,
        ]);
    }
    debug_assert_eq!(words.len() % 4, 0);
    debug_assert_eq!(CLUSTER_WORDS, 12);
    let mut offset = 0u32;
    for page in &pages {
        words.extend([offset, page.len() as u32]);
        offset += page.len() as u32;
    }
    // The prefix: metadata and the pinned pages.
    words[30] = (words.len()
        + pages[..pinned_pages as usize]
            .iter()
            .map(Vec::len)
            .sum::<usize>()) as u32;
    for page in pages {
        words.extend(page);
    }
    words
}
