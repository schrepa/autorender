use std::f32::consts::{FRAC_2_PI, TAU};

use engine_math::{Random, clamp, max, min, ticks};
use glam::Vec3;

use super::numbers::{CombatNumbers, GunNumbers, RecoilNumbers, SpreadDeg};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stance {
    Stand,
    Crouch,
    Walk,
    Run,
    Air,
    CrouchMove,
    Land,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Footing {
    pub grounded: bool,
    pub crouched: bool,
    pub speed_mps: f32,
    pub landing: bool,
}

pub fn stance_of(combat: &CombatNumbers, run_speed_mps: f32, footing: Footing) -> Stance {
    let still = footing.speed_mps <= combat.deadzone_share * run_speed_mps;
    if !footing.grounded {
        Stance::Air
    } else if footing.landing {
        Stance::Land
    } else if still && footing.crouched {
        Stance::Crouch
    } else if still {
        Stance::Stand
    } else if footing.crouched {
        Stance::CrouchMove
    } else if footing.speed_mps <= combat.walk_share * run_speed_mps {
        Stance::Walk
    } else {
        Stance::Run
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Aim {
    pub eye: Vec3,
    pub yaw_radians: f32,
    pub pitch_radians: f32,
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Kick {
    pub pitch_radians: f32,
    pub yaw_radians: f32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Shot {
    pub origin: Vec3,
    pub direction: Vec3,
    pub spread_deg: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
struct Recovery {
    from_shots: u32,
    from_pitch: f32,
    from_yaw: f32,
    elapsed: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Recoil {
    pub shots: u32,
    pub pitch_radians: f32,
    pub yaw_radians: f32,
    recovery: Recovery,
}

pub fn bloomed_deg(gun: &GunNumbers, stance: Stance, scope_level: usize, shots: u32) -> f32 {
    spread_deg(gun, stance, scope_level) + bloom_deg(gun, shots)
}

pub fn bloom_deg(gun: &GunNumbers, shots: u32) -> f32 {
    min(gun.spread_bloom.per_shot_deg * shots as f32, gun.spread_bloom.max_deg)
}

pub fn continuous_spread_deg(
    combat: &CombatNumbers,
    run_speed_mps: f32,
    gun: &GunNumbers,
    scope_level: usize,
    footing: Footing,
    land_share: f32,
) -> f32 {
    let table = gun
        .scope
        .as_ref()
        .filter(|_| scope_level > 0)
        .map_or(&gun.spread_deg, |scope| &scope.spread_deg);
    if !footing.grounded {
        return table.air;
    }
    let run = max(run_speed_mps, 0.01);
    let base = if footing.crouched {
        let dead = run * 0.14;
        let full = run * 0.42;
        let t = clamp((footing.speed_mps - dead) / max(full - dead, 0.01), 0.0, 1.0);
        table.crouch + (table.crouch_move - table.crouch) * (t * t)
    } else {
        let dead = combat.deadzone_share * run;
        let walk = combat.walk_share * run;
        if footing.speed_mps <= dead {
            table.stand
        } else if footing.speed_mps <= walk {
            let u = clamp((footing.speed_mps - dead) / max(walk - dead, 0.01), 0.0, 1.0);
            let curve = u * u * (3.0 - 2.0 * u);
            table.stand + (table.walk - table.stand) * curve
        } else {
            let v = clamp((footing.speed_mps - walk) / max(run - walk, 0.01), 0.0, 1.0);
            let curve = v * (1.6 - 0.6 * v);
            table.walk + (table.run - table.walk) * curve
        }
    };
    let land = clamp(land_share, 0.0, 1.0);
    base + max(table.land - base, 0.0) * (land * land)
}

pub fn spread_deg(gun: &GunNumbers, stance: Stance, scope_level: usize) -> f32 {
    let table = gun
        .scope
        .as_ref()
        .filter(|_| scope_level > 0)
        .map_or(&gun.spread_deg, |scope| &scope.spread_deg);
    stance_spread(table, stance)
}

fn stance_spread(table: &SpreadDeg, stance: Stance) -> f32 {
    match stance {
        Stance::Stand => table.stand,
        Stance::Crouch => table.crouch,
        Stance::Walk => table.walk,
        Stance::Run => table.run,
        Stance::Air => table.air,
        Stance::CrouchMove => table.crouch_move,
        Stance::Land => table.land,
    }
}

fn triangle(step: u32, swing: u32) -> f32 {
    if swing == 0 {
        return 0.0;
    }
    let phase = (step + swing) % (4 * swing);
    (swing as f32) - (phase as f32 - 2.0 * swing as f32).abs()
}

pub fn pattern_deg(recoil: &RecoilNumbers, shots: u32) -> (f32, f32) {
    let up = min(shots as f32 * recoil.up_deg_per_shot, recoil.max_up_deg);
    let swaying = shots.saturating_sub(recoil.side_starts_after_shots);
    let side = recoil.side_deg * triangle(swaying, recoil.side_swing_shots);
    (up, side)
}

pub fn fire(
    recoil_numbers: &RecoilNumbers,
    recoil: &mut Recoil,
    aim: Aim,
    spread_deg: f32,
    rng: &mut Random,
) -> (Shot, Kick) {
    let radius_draw = rng.unit();
    let turn_draw = rng.unit();
    let jitter_draw = rng.unit();
    let deviation = spread_deg.to_radians() * radius_draw.sqrt();
    let direction = cone(
        aim.yaw_radians + recoil.yaw_radians,
        aim.pitch_radians + recoil.pitch_radians,
        deviation,
        turn_draw * TAU,
    );
    recoil.shots = recoil.shots.saturating_add(1);
    let (up, side) = pattern_deg(recoil_numbers, recoil.shots);
    let jitter = (jitter_draw * 2.0 - 1.0) * recoil_numbers.side_jitter_deg;
    let pitch_after = up.to_radians();
    let yaw_after = -(side + jitter).to_radians();
    let kick = Kick {
        pitch_radians: pitch_after - recoil.pitch_radians,
        yaw_radians: yaw_after - recoil.yaw_radians,
    };
    recoil.pitch_radians = pitch_after;
    recoil.yaw_radians = yaw_after;
    recoil.recovery = Recovery {
        from_shots: recoil.shots,
        from_pitch: pitch_after,
        from_yaw: yaw_after,
        elapsed: 0,
    };
    let shot = Shot {
        origin: aim.eye,
        direction,
        spread_deg,
    };
    (shot, kick)
}

pub fn recover(recoil_numbers: &RecoilNumbers, recoil: &mut Recoil, ticks_per_second: u32) -> Kick {
    let total = ticks(recoil_numbers.recover_seconds, ticks_per_second);
    let recovery = recoil.recovery;
    if recovery.elapsed >= total && recoil.shots == 0 {
        return Kick::default();
    }
    let elapsed = (recovery.elapsed + 1).min(total);
    let left = total - elapsed;
    let share = if total == 0 {
        0.0
    } else {
        left as f32 / total as f32
    };
    let eased = share * (0.55 + 0.45 * share);
    let pitch = recovery.from_pitch * eased;
    let yaw = recovery.from_yaw * eased;
    let kick = Kick {
        pitch_radians: pitch - recoil.pitch_radians,
        yaw_radians: yaw - recoil.yaw_radians,
    };
    recoil.pitch_radians = pitch;
    recoil.yaw_radians = yaw;
    recoil.shots = (recovery.from_shots * left).div_ceil(total.max(1));
    recoil.recovery.elapsed = elapsed;
    kick
}

pub fn forward(yaw: f32, pitch: f32) -> Vec3 {
    let (yaw_sin, yaw_cos) = sin_cos(yaw);
    let (pitch_sin, pitch_cos) = sin_cos(pitch);
    Vec3::new(-yaw_sin * pitch_cos, pitch_sin, -yaw_cos * pitch_cos)
}

fn cone(yaw: f32, pitch: f32, deviation: f32, turn: f32) -> Vec3 {
    let ahead = forward(yaw, pitch);
    let (yaw_sin, yaw_cos) = sin_cos(yaw);
    let right = Vec3::new(yaw_cos, 0.0, -yaw_sin);
    let up = right.cross(ahead);
    let (deviation_sin, deviation_cos) = sin_cos(deviation);
    let (turn_sin, turn_cos) = sin_cos(turn);
    let aside = right * turn_cos + up * turn_sin;
    (ahead * deviation_cos + aside * deviation_sin).normalize()
}

pub fn sin_cos(angle: f32) -> (f32, f32) {
    let quadrant = (angle * FRAC_2_PI).round();
    let reduced =
        ((angle - quadrant * 1.570_312_5) - quadrant * 4.837_513e-4) - quadrant * 7.549_79e-8;
    let square = reduced * reduced;
    let sin = reduced
        + reduced * square * (-1.666_665_5e-1 + square * (8.332_161e-3 + square * -1.951_529_6e-4));
    let cos = 1.0 - 0.5 * square
        + square * square * (4.166_664_6e-2 + square * (-1.388_731_6e-3 + square * 2.443_315_7e-5));
    match (quadrant as i64).rem_euclid(4) {
        0 => (sin, cos),
        1 => (cos, -sin),
        2 => (-sin, -cos),
        _ => (-cos, sin),
    }
}
