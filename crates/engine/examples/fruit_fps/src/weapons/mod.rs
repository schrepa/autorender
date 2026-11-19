mod gun;
mod hits;
mod ledger;
mod loadout;
mod numbers;
mod recoil;

pub use gun::{Gun, GunInput, GunTick, GunView, Pace};
pub use hits::{
    Blocker, BodyPart, Capsule, CharacterId, HitJoints, HitShapes, MeleeHit, Sphere, Struck,
    Vitals, Wound, gun_damage, hit_shapes, melee, melee_damage, trace, wound,
};
pub use ledger::{DamageLedger, joints_from_body};
pub use loadout::{
    Fired, HandMotion, HandMoving, Inspecting, Loadout, LoadoutInput, LoadoutTick, LoadoutView, Slot,
    Swung, actor_rng,
};
pub use numbers::{
    Absorb, ArmsNumbers, ClassMeters, CombatNumbers, DamageBand, DeviceHand, GunKind, GunNumbers,
    HitShapeNumbers, KeyDrop, MeleeNumbers, PlayerShape, RecoilNumbers, ScopeNumbers, SpreadBloom,
    SpreadDeg, StandJoints, StrikeSeconds, SurfaceNumbers, WeaponNumbers, ticks,
};
pub use recoil::{
    Aim, Footing, Kick, Recoil, Shot, ShotRng, Stance, fire, forward, pattern_deg, recover,
    sin_cos, spread_deg, stance_of,
};
