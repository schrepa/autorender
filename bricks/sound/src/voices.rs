use engine_core::VoiceId;
use engine_math::Random;
use glam::Vec3;

use crate::Anchor;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Voice {
    pub door: Option<VoiceId>,
    pub sound: usize,
    pub priority: u32,
    pub base: f32,
    pub gain: f32,
    pub pan: f32,
    pub rate: f32,
    pub sent_gain: f32,
    pub sent_pan: f32,
    pub owed: bool,
    pub anchor: Anchor,
    pub at: Vec3,
    pub source: Option<u64>,
    pub tag: Option<u64>,
    pub started: u64,
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

fn below(rng: &mut Random, count: usize) -> usize {
    (rng.next_u64() % count as u64) as usize
}

pub(super) fn signed(rng: &mut Random) -> f32 {
    rng.unit() * 2.0 - 1.0
}

pub(super) fn variation(rng: &mut Random, count: usize, last: Option<usize>) -> usize {
    match last {
        Some(last) if count > 1 => {
            let pick = below(rng, count - 1);
            if pick >= last { pick + 1 } else { pick }
        }
        Some(_) | None => below(rng, count),
    }
}
