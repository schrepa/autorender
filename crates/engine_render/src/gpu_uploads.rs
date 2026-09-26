//! Upload planning for GPU presentation inputs (decision 014). Core stamps each
//! gathered record with the gather sequence that last changed it; the renderer
//! uploads records newer than the sequence it last uploaded, coalescing small
//! clean gaps with the same byte/call heuristics as retained uploads.

use crate::uploads::UploadRange;

const MAX_RANGES: usize = 64;
// Include clean gaps up to this many bytes instead of another queue call.
const WRITE_CALL_BYTES: usize = 16 * 1024;

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
    if covered * 4 >= stamps.len() * 3 && !ranges.is_empty() {
        return full(stamps.len(), ranges);
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
}
