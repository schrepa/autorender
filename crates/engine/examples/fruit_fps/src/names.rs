use engine_core::TargetName;

pub const COLOR: TargetName = TargetName("color");
pub const DEPTH: TargetName = TargetName("depth");

pub const COLOR_PASS: &str = "color";
pub const COMPOSITE_PASS: &str = "composite";

pub const CAMERA: &str = "camera";

pub const BONE_ROOT: &str = "BoneRoot";
pub const HIP: &str = "Hip";
pub const SPINE: &str = "Spine01";
pub const HEAD: &str = "Head";
pub const RIGHT_HAND: &str = "R_Hand";
pub const LEFT_HAND: &str = "L_Hand";
pub const BONES: [&str; 6] = [BONE_ROOT, HIP, SPINE, HEAD, RIGHT_HAND, LEFT_HAND];

pub const IDLE: &str = "idle";
pub const WALK: &str = "walk";
pub const RUN: &str = "run";
pub const JUMP: &str = "jump";
pub const FALL: &str = "fall";
pub const TURN: &str = "turn";
pub const SHOOT: &str = "shoot";
pub const FIRE: &str = "fire";
pub const HURT: &str = "hurt";
pub const HIT_TO_HEAD: &str = "hit_to_head";
pub const HIT_TO_BODY: &str = "hit_to_body";
pub const DEFEAT_A: &str = "defeat_a";
pub const DEFEAT_B: &str = "defeat_b";
pub const CAST: &str = "cast";
pub const DIG: &str = "dig";
pub const CLIPS: [&str; 15] = [
    IDLE,
    WALK,
    RUN,
    JUMP,
    FALL,
    TURN,
    SHOOT,
    FIRE,
    HURT,
    HIT_TO_HEAD,
    HIT_TO_BODY,
    DEFEAT_A,
    DEFEAT_B,
    CAST,
    DIG,
];

pub const PITCH_LIMIT_DEGREES: f32 = 89.0;
pub const NEAR: f32 = 0.05;
pub const FAR: f32 = 300.0;
