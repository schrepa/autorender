//! Upload planning for GPU presentation inputs (decision 014). Core stamps each
//! gathered record with the gather sequence that last changed it; the renderer
//! uploads records newer than the sequence it last uploaded, coalescing small
//! clean gaps with the same byte/call heuristics as retained uploads.

pub(crate) use crate::uploads::whole_stream;
use crate::uploads::{UploadRange, WRITE_CALL_BYTES};

const MAX_RANGES: usize = 64;

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
