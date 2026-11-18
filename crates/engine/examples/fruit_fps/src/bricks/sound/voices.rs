use engine_core::VoiceId;
use glam::Vec3;

use crate::play::sound::Anchor;

use super::table::Group;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Voice {
    pub door: Option<VoiceId>,
    pub sound: String,
    pub priority: u32,
    pub base: f32,
    pub gain: f32,
    pub pan: f32,
    pub rate: f32,
    pub sent_gain: f32,
    pub sent_pan: f32,
    pub anchor: Anchor,
    pub at: Vec3,
    pub source: Option<u32>,
    pub group: Group,
    pub tag: Option<u64>,
    pub occlusion: f32,
    pub looping: bool,
    pub ends: u64,
}

pub(super) enum Admit {
    Free,
    Steal(usize),
    Drop,
}

pub(super) fn admit(voices: &[Voice], budget: usize, priority: u32, gain: f32) -> Admit {
    if voices.len() < budget {
        return Admit::Free;
    }
    let mut lowest: Option<(usize, &Voice)> = None;
    for (index, voice) in voices.iter().enumerate() {
        let lower = lowest.is_none_or(|(_, held)| {
            voice.priority < held.priority
                || (voice.priority == held.priority && voice.gain < held.gain)
        });
        if lower {
            lowest = Some((index, voice));
        }
    }
    match lowest {
        Some((index, held))
            if priority > held.priority || (priority == held.priority && gain > held.gain) =>
        {
            Admit::Steal(index)
        }
        Some(_) | None => Admit::Drop,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        mixed ^ (mixed >> 31)
    }

    pub fn below(&mut self, count: usize) -> usize {
        (self.next() % count as u64) as usize
    }

    pub fn signed(&mut self) -> f32 {
        let unit = (self.next() >> 40) as f32 / (1_u64 << 24) as f32;
        unit * 2.0 - 1.0
    }
}

pub(super) fn variation(rng: &mut Rng, count: usize, last: Option<usize>) -> usize {
    match last {
        Some(last) if count > 1 => {
            let pick = rng.below(count - 1);
            if pick >= last { pick + 1 } else { pick }
        }
        Some(_) | None => rng.below(count),
    }
}
