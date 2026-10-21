//! The triangles shadow maps draw of each level of a mesh: shadow maps
//! cull no faces, so a face drawn from its back too (two triangles wound
//! both ways over the same vertices) would rasterize the same texels at the
//! same depths twice; casters draw one of each pair.

use engine_math::collections::HashMap;

/// The triangles of index list `triangles` less the second of each pair
/// wound both ways over the same vertices. `None` when no triangle has
/// one.
fn without_back_faces(triangles: &[u32]) -> Option<Vec<u32>> {
    // A triangle's corners from its least index on, keeping its winding.
    let wound = |[a, b, c]: [u32; 3]| {
        if a <= b && a <= c {
            [a, b, c]
        } else if b <= c {
            [b, c, a]
        } else {
            [c, a, b]
        }
    };
    // Triangles kept whose other winding has not come yet.
    let mut open: HashMap<[u32; 3], u32> = HashMap::default();
    let mut kept = Vec::with_capacity(triangles.len());
    for &[a, b, c] in triangles.as_chunks::<3>().0 {
        if let Some(waiting) = open.get_mut(&wound([a, c, b]))
            && *waiting > 0
        {
            *waiting -= 1;
            continue;
        }
        *open.entry(wound([a, b, c])).or_insert(0) += 1;
        kept.extend_from_slice(&[a, b, c]);
    }
    (kept.len() < triangles.len()).then_some(kept)
}

/// The triangles shadow maps draw of each level of a mesh whose `indices`
/// hold its own `index_count` and then its coarser `lods` (each one's first
/// index there, index count and error): each level's first index there and
/// index count, its own where it has no back faces
/// ([`without_back_faces`]), the others' appended to `indices`. Empty when
/// the mesh's own triangles have none: its coarser levels are then not
/// searched.
pub(crate) fn caster_ranges(
    indices: &mut Vec<u32>,
    index_count: u32,
    lods: &[(u32, u32, f32)],
) -> Vec<(u32, u32)> {
    let Some(own) = without_back_faces(&indices[..index_count as usize]) else {
        return Vec::new();
    };
    let append = |indices: &mut Vec<u32>, kept: Vec<u32>| {
        let first = indices.len() as u32;
        indices.extend_from_slice(&kept);
        (first, kept.len() as u32)
    };
    let mut ranges = Vec::with_capacity(1 + lods.len());
    ranges.push(append(indices, own));
    for &(first, count, _) in lods {
        let level = first as usize..(first + count) as usize;
        ranges.push(match without_back_faces(&indices[level]) {
            Some(kept) => append(indices, kept),
            None => (first, count),
        });
    }
    ranges
}
