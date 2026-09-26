//! Upload planning for GPU presentation inputs (decision 014). Core stamps each
//! gathered record with the gather sequence that last changed it; the renderer
//! uploads records newer than the sequence it last uploaded, coalescing small
//! clean gaps with the same byte/call heuristics as retained uploads.

use crate::uploads::UploadRange;

const MAX_RANGES: usize = 64;
// Include clean gaps up to this many bytes instead of another queue call.
const WRITE_CALL_BYTES: usize = 16 * 1024;

/// Whether a stream of `len` records with `stamped` of them changed uploads
/// whole instead of by records: at least three quarters changed.
pub(crate) fn whole_stream(stamped: usize, len: usize) -> bool {
    stamped * 4 >= len * 3
}

/// Ranges of records whose stamp exceeds `since`. Returns false, with a single
/// full range, when the plan is fragmented or covers at least 75% of records.
pub(crate) fn changed_ranges(
    stamps: &[u64],
    since: u64,
    stride: usize,
    ranges: &mut Vec<UploadRange>,
) -> bool {
    ranges.clear();
    let gap = WRITE_CALL_BYTES.div_ceil(stride);
    let mut covered = 0;
    for (index, &stamp) in stamps.iter().enumerate() {
        if stamp <= since {
            continue;
        }
        match ranges.last_mut() {
            Some(range) if index - range.end <= gap => {
                covered += index + 1 - range.end;
                range.end = index + 1;
            }
            _ => {
                if ranges.len() == MAX_RANGES {
                    return full(stamps.len(), ranges);
                }
                ranges.push(UploadRange {
                    start: index,
                    end: index + 1,
                });
                covered += 1;
            }
        }
    }
    if whole_stream(covered, stamps.len()) && !ranges.is_empty() {
        return full(stamps.len(), ranges);
    }
    true
}

/// [`changed_ranges`] for the records listed in `nodes` (each below `len`,
/// in any order) instead of those with newer stamps: the same ranges, from
/// the listed records alone. `bits` is scratch, left cleared.
pub(crate) fn listed_ranges(
    nodes: &[u32],
    len: usize,
    stride: usize,
    bits: &mut Vec<u64>,
    ranges: &mut Vec<UploadRange>,
) -> bool {
    ranges.clear();
    let words = len.div_ceil(64);
    if bits.len() < words {
        bits.resize(words, 0);
    }
    for &node in nodes {
        let node = node as usize;
        debug_assert!(node < len);
        bits[node / 64] |= 1 << (node % 64);
    }
    let gap = WRITE_CALL_BYTES.div_ceil(stride);
    let mut covered = 0;
    let mut partial = true;
    // Runs of listed records in ascending order, merged exactly as
    // `changed_ranges` merges records one at a time.
    'words: for (word, &value) in bits[..words].iter().enumerate() {
        let mut value = value;
        while value != 0 {
            let first = value.trailing_zeros() as usize;
            let run = (value >> first).trailing_ones() as usize;
            let (start, end) = (word * 64 + first, word * 64 + first + run);
            match ranges.last_mut() {
                Some(range) if start - range.end <= gap => {
                    covered += end - range.end;
                    range.end = end;
                }
                _ => {
                    if ranges.len() == MAX_RANGES {
                        partial = false;
                        break 'words;
                    }
                    ranges.push(UploadRange { start, end });
                    covered += end - start;
                }
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
    if !partial || (whole_stream(covered, len) && !ranges.is_empty()) {
        return full(len, ranges);
    }
    true
}

fn full(len: usize, ranges: &mut Vec<UploadRange>) -> bool {
    ranges.clear();
    if len != 0 {
        ranges.push(UploadRange { start: 0, end: len });
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(stamps: &[u64], since: u64, stride: usize) -> (bool, Vec<(usize, usize)>) {
        let mut ranges = Vec::new();
        let partial = changed_ranges(stamps, since, stride, &mut ranges);
        (
            partial,
            ranges
                .iter()
                .map(|range| (range.start, range.end))
                .collect(),
        )
    }

    #[test]
    fn newer_stamps_coalesce_small_gaps_and_dense_plans_become_full() {
        let mut stamps = vec![1_u64; 10_000];
        assert_eq!(plan(&stamps, 1, 80), (true, vec![]));
        stamps[10] = 2;
        stamps[12] = 2;
        stamps[9_000] = 3;
        // A 16 KiB gap allowance at 80-byte records spans 205 records.
        assert_eq!(plan(&stamps, 1, 80), (true, vec![(10, 13), (9_000, 9_001)]));
        assert_eq!(plan(&stamps, 2, 80), (true, vec![(9_000, 9_001)]));
        for stamp in &mut stamps[..8_000] {
            *stamp = 4;
        }
        assert_eq!(plan(&stamps, 3, 80), (false, vec![(0, 10_000)]));
        // Fragmented plans upload everything.
        let sparse: Vec<u64> = (0..100_000)
            .map(|index| u64::from(index % 1_000 == 0) + 1)
            .collect();
        assert_eq!(plan(&sparse, 1, 8), (false, vec![(0, 100_000)]));
        assert_eq!(plan(&[], 0, 8), (true, vec![]));
    }

    #[test]
    fn listed_records_plan_the_same_ranges_as_newer_stamps() {
        let mut state = 0x9e37_79b9_u32;
        let mut random = move |bound: u32| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state % bound
        };
        let mut bits = Vec::new();
        for case in 0..400 {
            let len = [0, 1, 63, 64, 65, 1_000, 4_096, 20_000][case % 8];
            let stride = [8, 40, 48, 80][case % 4];
            let mut stamps = vec![1_u64; len];
            // Sparse scatter, contiguous runs, dense blocks and near-full streams.
            let density = [1, 20, 200, 900, 1_000][case % 5];
            for stamp in &mut stamps {
                if random(1_000) < density {
                    *stamp = 2;
                }
            }
            if len > 0 && case % 3 == 0 {
                let start = random(len as u32) as usize;
                let end = (start + random(3_000) as usize).min(len);
                stamps[start..end].fill(2);
            }
            let mut nodes: Vec<u32> = (0..len as u32)
                .filter(|&index| stamps[index as usize] == 2)
                .collect();
            // Lists come in any order.
            for index in (1..nodes.len()).rev() {
                nodes.swap(index, random(index as u32 + 1) as usize);
            }
            let mut ranges = Vec::new();
            let partial = listed_ranges(&nodes, len, stride, &mut bits, &mut ranges);
            let listed: Vec<(usize, usize)> = ranges
                .iter()
                .map(|range| (range.start, range.end))
                .collect();
            assert_eq!((partial, listed), plan(&stamps, 1, stride), "case {case}");
            assert!(bits.iter().all(|&word| word == 0), "scratch left cleared");
        }
    }

    #[test]
    fn streams_upload_whole_from_three_quarters_stamped() {
        assert!(whole_stream(75, 100));
        assert!(!whole_stream(74, 100));
        assert!(whole_stream(0, 0));
        assert!(!whole_stream(25_000, 250_000));
    }
}
