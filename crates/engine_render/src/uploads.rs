#[derive(Clone, Copy, Default)]
pub struct UploadRange {
    pub start: usize,
    pub end: usize,
}

pub fn listed_ranges(
    nodes: &[u32],
    len: usize,
    bits: &mut Vec<u64>,
    ranges: &mut Vec<UploadRange>,
) {
    ranges.clear();
    let words = len.div_ceil(64);
    if bits.len() < words {
        bits.resize(words, 0);
    }
    for &node in nodes {
        let node = node as usize;
        debug_assert!(node < len, "changed_node_in_array");
        bits[node / 64] |= 1 << (node % 64);
    }
    for (word, &value) in bits[..words].iter().enumerate() {
        let mut value = value;
        while value != 0 {
            let first = value.trailing_zeros() as usize;
            let run = (value >> first).trailing_ones() as usize;
            let (start, end) = (word * 64 + first, word * 64 + first + run);
            match ranges.last_mut() {
                Some(range) if range.end == start => range.end = end,
                _ => ranges.push(UploadRange { start, end }),
            }
            value = if first + run == 64 {
                0
            } else {
                value & !(((1 << run) - 1) << first)
            };
        }
    }
    for &node in nodes {
        bits[node as usize / 64] = 0;
    }
}
