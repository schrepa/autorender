#[derive(Default)]
pub(crate) struct Nearest {
    keys: Vec<u64>,
}

const POSITION: u64 = u32::MAX as u64;

impl Nearest {
    pub(crate) fn nearest_first(
        &mut self,
        distances: impl IntoIterator<Item = f32>,
        kept: usize,
    ) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.select(distances, kept);
        self.keys.sort_unstable();
        self.positions()
    }

    fn select(&mut self, distances: impl IntoIterator<Item = f32>, kept: usize) {
        self.keys.clear();
        self.keys.extend(
            distances
                .into_iter()
                .enumerate()
                .map(|(position, distance)| {
                    u64::from(total_order(distance)) << 32 | position as u64
                }),
        );
        if kept < self.keys.len() {
            self.keys.select_nth_unstable(kept);
            self.keys.truncate(kept);
        }
    }

    fn positions(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.keys.iter().map(|key| (key & POSITION) as usize)
    }
}

fn total_order(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits >> 31 == 1 {
        !bits
    } else {
        bits | 1 << 31
    }
}
