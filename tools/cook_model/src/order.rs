const CACHE: usize = 32;
const VALENCES: usize = 64;

fn tables() -> ([f32; CACHE], [f32; VALENCES]) {
    let cache = std::array::from_fn(|at| {
        if at < 3 {
            0.75
        } else {
            let scaled = 1.0 - (at - 3) as f32 / (CACHE - 3) as f32;
            scaled.powf(1.5)
        }
    });
    let valence = std::array::from_fn(|count| {
        if count == 0 {
            0.0
        } else {
            2.0 / (count as f32).sqrt()
        }
    });
    (cache, valence)
}

pub fn triangles(indices: &[u32], vertices: usize) -> Vec<u32> {
    let count = indices.len() / 3;
    let (cache_score, valence_score) = tables();
    let mut starts = vec![0usize; vertices + 1];
    for &index in indices {
        starts[index as usize + 1] += 1;
    }
    for at in 0..vertices {
        starts[at + 1] += starts[at];
    }
    let mut fill = starts.clone();
    let mut adjacent = vec![0u32; indices.len()];
    for (triangle, corners) in indices.as_chunks::<3>().0.iter().enumerate() {
        for &corner in corners {
            let slot = &mut fill[corner as usize];
            adjacent[*slot] = u32::try_from(triangle).expect("triangles_fit_u32");
            *slot += 1;
        }
    }
    let mut remaining: Vec<usize> = (0..vertices)
        .map(|at| starts[at + 1] - starts[at])
        .collect();
    let mut position = vec![usize::MAX; vertices];
    let score_of = |held: usize, at: usize| {
        if held == 0 {
            return -1.0;
        }
        let near = if at < CACHE { cache_score[at] } else { 0.0 };
        near + valence_score[held.min(VALENCES - 1)]
    };
    let mut score: Vec<f32> = (0..vertices)
        .map(|at| score_of(remaining[at], usize::MAX))
        .collect();
    let mut emitted = vec![false; count];
    let triangle_score = |score: &[f32], triangle: usize| {
        indices[triangle * 3..triangle * 3 + 3]
            .iter()
            .map(|&corner| score[corner as usize])
            .sum::<f32>()
    };
    let mut cache: Vec<u32> = Vec::with_capacity(CACHE + 3);
    let mut out = Vec::with_capacity(indices.len());
    let mut cursor = 0;
    let mut best: Option<usize> = None;
    while out.len() < indices.len() {
        let chosen = match best {
            Some(triangle) => triangle,
            None => {
                while cursor < count && emitted[cursor] {
                    cursor += 1;
                }
                let Some(first) = (cursor..count).find(|&triangle| !emitted[triangle]) else {
                    break;
                };
                first
            }
        };
        emitted[chosen] = true;
        let corners = [
            indices[chosen * 3],
            indices[chosen * 3 + 1],
            indices[chosen * 3 + 2],
        ];
        out.extend_from_slice(&corners);
        for &corner in &corners {
            let vertex = corner as usize;
            remaining[vertex] -= 1;
            let list = &mut adjacent[starts[vertex]..starts[vertex + 1]];
            if let Some(at) = list.iter().position(|&held| held as usize == chosen) {
                let last = remaining[vertex];
                list.swap(at, last);
            }
        }
        let mut next: Vec<u32> = corners.to_vec();
        for &held in &cache {
            if !corners.contains(&held) {
                next.push(held);
            }
        }
        for &dropped in next.iter().skip(CACHE) {
            position[dropped as usize] = usize::MAX;
        }
        next.truncate(CACHE);
        for (at, &held) in next.iter().enumerate() {
            position[held as usize] = at;
        }
        for &dropped in cache.iter().filter(|held| !next.contains(held)) {
            score[dropped as usize] = score_of(remaining[dropped as usize], usize::MAX);
        }
        cache = next;
        for &held in &cache {
            let vertex = held as usize;
            score[vertex] = score_of(remaining[vertex], position[vertex]);
        }
        best = None;
        let mut top = f32::MIN;
        for &held in &cache {
            let vertex = held as usize;
            for &triangle in &adjacent[starts[vertex]..starts[vertex] + remaining[vertex]] {
                let triangle = triangle as usize;
                let value = triangle_score(&score, triangle);
                if value > top {
                    top = value;
                    best = Some(triangle);
                }
            }
        }
    }
    out
}

pub fn first_use(indices: &mut [u32], vertices: usize) -> Vec<u32> {
    let mut new_of = vec![u32::MAX; vertices];
    let mut old_of = Vec::with_capacity(vertices);
    for index in indices.iter_mut() {
        let slot = &mut new_of[*index as usize];
        if *slot == u32::MAX {
            *slot = u32::try_from(old_of.len()).expect("vertices_fit_u32");
            old_of.push(*index);
        }
        *index = *slot;
    }
    for (old, slot) in new_of.iter().enumerate() {
        if *slot == u32::MAX {
            old_of.push(u32::try_from(old).expect("vertices_fit_u32"));
        }
    }
    old_of
}

pub fn permuted<T: Copy>(values: &[T], old_of: &[u32]) -> Vec<T> {
    old_of.iter().map(|&old| values[old as usize]).collect()
}
