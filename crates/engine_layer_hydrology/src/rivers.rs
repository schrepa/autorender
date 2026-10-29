use engine_math::{max, min};
use engine_terrain::rivers::{RIVER_STEP, RiverLine};
use glam::Vec2;

pub(crate) struct Channels<'a> {
    pub(crate) area: &'a [f32],
    pub(crate) receivers: &'a [u32],
    pub(crate) lakes: &'a [bool],
    pub(crate) rank: &'a [u32],
    pub(crate) within: Option<&'a [bool]>,
    pub(crate) continued: &'a [(u32, Vec2)],
}

pub(crate) fn trace(
    channels: &Channels<'_>,
    (width, spacing): (usize, f32),
    channel_area: f32,
    widths: impl Fn(f32) -> f32,
) -> Vec<RiverLine> {
    let Channels {
        area,
        receivers,
        lakes,
        rank,
        within,
        continued,
    } = *channels;
    let count = area.len();
    let given = |sample: usize| within.is_some_and(|within| within[sample]);
    let channel = |sample: usize| area[sample] >= channel_area && !lakes[sample] && !given(sample);
    let mut inflows = vec![0_u8; count];
    for (sample, &receiver) in receivers.iter().enumerate() {
        if channel(sample) && receiver != u32::MAX {
            let entry = &mut inflows[receiver as usize];
            *entry = entry.saturating_add(1);
        }
    }
    for &(sample, _) in continued {
        let entry = &mut inflows[sample as usize];
        *entry = entry.saturating_add(1);
    }
    let lead = |sample: usize| {
        continued
            .binary_search_by_key(&(sample as u32), |&(at, _)| at)
            .ok()
            .map(|found| continued[found].1)
    };
    let mut starts: Vec<usize> = (0..count)
        .filter(|&sample| channel(sample) && (inflows[sample] != 1 || lead(sample).is_some()))
        .collect();
    starts.sort_by_key(|&sample| std::cmp::Reverse(rank[sample]));
    let place = |sample: usize| {
        Vec2::new(
            (sample % width) as f32 * spacing,
            (sample / width) as f32 * spacing,
        )
    };
    let mut lines = Vec::with_capacity(starts.len());
    for start in starts {
        let mut course = vec![start];
        let mut current = start;
        let mut joins = false;
        loop {
            let receiver = receivers[current];
            if receiver == u32::MAX {
                break;
            }
            let next = receiver as usize;
            course.push(next);
            if lakes[next] || !channel(next) || inflows[next] >= 2 {
                joins = true;
                break;
            }
            current = next;
        }
        if course.len() < 2 {
            continue;
        }
        let last = course.len() - 1;
        let drains = |index: usize| {
            let at = if joins && index == last {
                index - 1
            } else {
                index
            };
            area[course[at]]
        };
        let mut shaping = vec![0];
        for index in 1..last {
            let from = place(course[shaping[shaping.len() - 1]]);
            if from.distance(place(course[index])) >= widths(drains(index)) {
                shaping.push(index);
            }
        }
        shaping.push(last);
        let mut points: Vec<Vec2> = shaping.iter().map(|&index| place(course[index])).collect();
        let mut drained: Vec<f32> = shaping.iter().map(|&index| drains(index)).collect();
        if let Some(lead) = lead(start) {
            points.insert(0, lead);
            drained.insert(0, drained[0]);
        }
        let (points, area) = smooth_line(&points, &drained);
        lines.push(RiverLine {
            points,
            area,
            head: inflows[start] == 0,
            given: None,
        });
    }
    lines
}

pub(crate) const MEASURED: usize = 8;

fn spline_weights(t: f32) -> [f32; 4] {
    let s = 1.0 - t;
    let (t2, t3) = (t * t, t * t * t);
    [
        s * s * s / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (3.0 * (t + t2 - t3) + 1.0) / 6.0,
        t3 / 6.0,
    ]
}

pub(crate) fn smooth_line(points: &[Vec2], carried: &[f32]) -> (Vec<Vec2>, Vec<f32>) {
    let last = points.len() - 1;
    let mut control: Vec<(Vec2, f32)> = Vec::with_capacity(last + 3);
    control.push((2.0 * points[0] - points[1], 2.0 * carried[0] - carried[1]));
    control.extend(points.iter().copied().zip(carried.iter().copied()));
    control.push((
        2.0 * points[last] - points[last - 1],
        2.0 * carried[last] - carried[last - 1],
    ));
    let at = |u: f32| {
        let segment = (u.floor() as usize).min(last - 1);
        let (mut point, mut value) = (Vec2::ZERO, 0.0_f32);
        for (offset, weight) in spline_weights(u - segment as f32).into_iter().enumerate() {
            let (place, carries) = control[segment + offset];
            point += place * weight;
            value += carries * weight;
        }
        (point, value)
    };
    let mut measured: Vec<(f32, f32)> = Vec::with_capacity(last * MEASURED + 1);
    measured.push((0.0, 0.0));
    let (mut length, mut previous) = (0.0_f32, points[0]);
    for place in 1..=last * MEASURED {
        let u = place as f32 / MEASURED as f32;
        let (point, _) = at(u);
        length += point.distance(previous);
        previous = point;
        measured.push((u, length));
    }
    let steps = max((length / RIVER_STEP).ceil(), 1.0) as usize;
    let (mut line, mut values) = (Vec::with_capacity(steps + 1), Vec::with_capacity(steps + 1));
    let mut interval = 0;
    for step in 0..=steps {
        let wanted = length * step as f32 / steps as f32;
        while interval + 2 < measured.len() && measured[interval + 1].1 < wanted {
            interval += 1;
        }
        let ((from, before), (to, after)) = (measured[interval], measured[interval + 1]);
        let share = if after > before {
            min(max((wanted - before) / (after - before), 0.0), 1.0)
        } else {
            0.0
        };
        let (point, value) = at(from + (to - from) * share);
        line.push(point);
        values.push(value);
    }
    line[0] = points[0];
    values[0] = carried[0];
    line[steps] = points[last];
    values[steps] = carried[last];
    (line, values)
}
