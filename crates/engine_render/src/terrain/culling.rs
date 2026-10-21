//! Culling the camera's terrain patches within its frustum: those whose
//! every triangle faces away from the eye, by the disk of slopes each
//! patch's tile was made with ([`engine_core::Patch::slopes`]); and those
//! nearer ground hides, by an occlusion horizon around the eye. Both hold
//! only what they can prove: a patch drops where no point of it could show.
//!
//! The horizon holds, in each of its azimuth bins around the eye, the
//! steepest sight it has found blocked by the quarters of nearer patches. A
//! quarter's lowest ground (its floor) blocks every sight across it that
//! passes below that floor somewhere over it, which, along every azimuth its
//! square spans whole, holds for sights rising less than the floor does from
//! the eye to the quarter's farthest point (nearest, for a floor below the
//! eye). A patch is hidden where, in every bin it touches, its highest point
//! rises less than what is blocked there. Patches go in order of their
//! nearest point, a quarter blocking once the patches tested lie wholly
//! past it; patches with holes block nothing.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::Range;

use engine_core::glam::Vec3;
use engine_core::math::{max, min};

/// Azimuth bins around the eye.
const BINS: usize = 1024;
/// Meters a floor is taken lower, and a patch's top higher, than they are:
/// what rounding and the vertex stage's arithmetic may move them by.
const MARGIN: f32 = 0.05;
/// How much wider than its cooked radius a patch's disk of slopes is taken:
/// its vertices morph toward their parent's at slightly different shares.
const SPREAD_MARGIN: f32 = 0.05;
/// Meters within which the eye lies over a patch: such a patch is never
/// hidden, and blocks nothing.
const OVER: f32 = 1.0;

/// A patch within the camera's frustum, as culling reads it: its box less
/// the eye (world meters), its disk of slopes, its quarters' floors less
/// the eye's height, and whether it has holes.
#[derive(Clone, Copy, Debug)]
pub(super) struct Seen {
    pub(super) min: Vec3,
    pub(super) max: Vec3,
    pub(super) slopes: [f32; 3],
    pub(super) floors: [f32; 4],
    pub(super) holes: bool,
}

/// Whether every triangle of `patch` faces away from the eye: from every
/// corner of its box, the eye lies behind the plane of every slope its disk
/// holds. How far it lies behind is concave over the box, so its corners
/// bound it.
pub(super) fn back_facing(patch: &Seen) -> bool {
    let [x, z, spread] = patch.slopes;
    let spread = spread * (1.0 + SPREAD_MARGIN) + SPREAD_MARGIN;
    (0..8).all(|corner: u32| {
        let pick = |bit: u32, low: f32, high: f32| if corner & bit == 0 { low } else { high };
        let (cx, cy, cz) = (
            pick(1, patch.min.x, patch.max.x),
            pick(2, patch.min.y, patch.max.y),
            pick(4, patch.min.z, patch.max.z),
        );
        cy - cx * x - cz * z - spread * (cx * cx + cz * cz).sqrt() > 0.0
    })
}

/// Marks in `hidden` which of `patches` nearer ground hides from the eye.
pub(super) fn hidden(patches: &[Seen], hidden: &mut Vec<bool>) {
    hidden.clear();
    hidden.resize(patches.len(), false);
    let reaches: Vec<(f32, f32)> = patches
        .iter()
        .map(|patch| reach(patch.min, patch.max))
        .collect();
    let mut order: Vec<usize> = (0..patches.len()).collect();
    order.sort_by(|&a, &b| reaches[a].0.total_cmp(&reaches[b].0).then(a.cmp(&b)));
    let mut bins = vec![f32::NEG_INFINITY; BINS];
    // Quarters waiting to block until the patches tested lie wholly past
    // them: their farthest reach (its bits, as reaches are positive), their
    // patch and which quarter.
    let mut waiting: BinaryHeap<Reverse<(u32, usize, usize)>> = BinaryHeap::new();
    for index in order {
        let (near, far) = reaches[index];
        while let Some(&Reverse((farthest, patch, quarter))) = waiting.peek() {
            if f32::from_bits(farthest) > near {
                break;
            }
            waiting.pop();
            block(&mut bins, &patches[patch], quarter);
        }
        let patch = &patches[index];
        if near > OVER {
            let top = patch.max.y + MARGIN;
            let rise = if top >= 0.0 { top / near } else { top / far };
            let (first, last) = span(patch.min, patch.max);
            hidden[index] = touched(first, last).all(|bin| bins[bin % BINS] > rise);
        }
        if !patch.holes {
            for quarter in 0..4 {
                let (low, high) = quarter_box(patch, quarter);
                waiting.push(Reverse((reach(low, high).1.to_bits(), index, quarter)));
            }
        }
    }
}

/// `patch`'s quarter `quarter` (rows along z): its footprint's least and
/// greatest corners at its floor.
fn quarter_box(patch: &Seen, quarter: usize) -> (Vec3, Vec3) {
    let half = (patch.max - patch.min) * 0.5;
    let (qx, qz) = ((quarter % 2) as f32, (quarter / 2) as f32);
    let low = Vec3::new(
        patch.min.x + qx * half.x,
        patch.floors[quarter],
        patch.min.z + qz * half.z,
    );
    (low, Vec3::new(low.x + half.x, low.y, low.z + half.z))
}

/// Blocks, in every bin `patch`'s quarter `quarter` spans whole, the sights
/// rising less than its floor does from the eye across it.
fn block(bins: &mut [f32], patch: &Seen, quarter: usize) {
    let (low, high) = quarter_box(patch, quarter);
    let (near, far) = reach(low, high);
    if near <= OVER {
        return;
    }
    let floor = low.y - MARGIN;
    let rise = if floor >= 0.0 {
        floor / far
    } else {
        floor / near
    };
    let (first, last) = span(low, high);
    for bin in covered(first, last) {
        let held = &mut bins[bin % BINS];
        *held = max(*held, rise);
    }
}

/// The nearest and farthest horizontal distances from the eye to the
/// footprint from `low` to `high`.
fn reach(low: Vec3, high: Vec3) -> (f32, f32) {
    let nearest = |a: f32, b: f32| max(max(a, -b), 0.0);
    let farthest = |a: f32, b: f32| max(a.abs(), b.abs());
    let (nx, nz) = (nearest(low.x, high.x), nearest(low.z, high.z));
    let (fx, fz) = (farthest(low.x, high.x), farthest(low.z, high.z));
    ((nx * nx + nz * nz).sqrt(), (fx * fx + fz * fz).sqrt())
}

/// The azimuths (in quarter turns from 0 to 4, the second unwrapped past
/// the first) the footprint from `low` to `high` spans from the eye, which
/// lies outside it.
fn span(low: Vec3, high: Vec3) -> (f32, f32) {
    let middle = turn(0.5 * (low.x + high.x), 0.5 * (low.z + high.z));
    let (mut first, mut last) = (0.0_f32, 0.0_f32);
    for (x, z) in [(low.x, low.z), (high.x, low.z), (low.x, high.z), (high.x, high.z)] {
        // Every corner lies within half a turn of the middle.
        let mut offset = turn(x, z) - middle;
        if offset > 2.0 {
            offset -= 4.0;
        } else if offset < -2.0 {
            offset += 4.0;
        }
        first = min(first, offset);
        last = max(last, offset);
    }
    let start = middle + first;
    let start = if start < 0.0 { start + 4.0 } else { start };
    (start, start + (last - first))
}

/// The direction `(x, z)` in quarter turns from +x toward +z, 0 to 4,
/// growing with its angle though not in proportion, half a turn apart from
/// its opposite.
fn turn(x: f32, z: f32) -> f32 {
    let sum = x.abs() + z.abs();
    if sum <= 0.0 {
        return 0.0;
    }
    let along = x / sum;
    if z >= 0.0 { 1.0 - along } else { 3.0 + along }
}

/// The bins (unwrapped) azimuths from `first` to `last` fall in.
fn touched(first: f32, last: f32) -> std::ops::RangeInclusive<usize> {
    let width = BINS as f32 / 4.0;
    (first * width) as usize..=(last * width) as usize
}

/// The bins (unwrapped) all of whose azimuths lie from `first` to `last`.
fn covered(first: f32, last: f32) -> Range<usize> {
    let width = BINS as f32 / 4.0;
    (first * width).ceil() as usize..(last * width).floor() as usize
}
