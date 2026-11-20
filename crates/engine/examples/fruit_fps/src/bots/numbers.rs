use serde::Deserialize;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    Easy,
    Normal,
    Hard,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct LevelNumbers {
    pub reaction_seconds: f32,
    pub first_shot_error_deg: f32,
    pub tracking_error_deg: f32,
    pub settle_seconds: f32,
    pub spray_settle_seconds: f32,
    pub turn_deg_per_second: f32,
    pub move_error: f32,
    pub head_share: f32,
    pub fire_tolerance: f32,
    pub hearing_loudness: f32,
    pub hearing_error_m: f32,
    pub memory_seconds: f32,
    pub ability_skill: f32,
    pub checks_corners: bool,
    pub run_and_gun: f32,
    pub notice_wire: f32,
    pub post_close: f32,
    pub post_peek: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Levels {
    pub easy: LevelNumbers,
    pub normal: LevelNumbers,
    pub hard: LevelNumbers,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct PlayNumbers {
    pub reaction_reset_seconds: f32,
    pub reaction_jitter: f32,
    pub preaimed_reaction: f32,
    pub preaim_cone_deg: f32,
    pub aim_ease_deg: f32,
    pub aim_min_turn_share: f32,
    pub error_wander_seconds: f32,
    pub blind_turn: f32,
    pub spray_m: f32,
    pub sweep_quiet_seconds: f32,
    pub pickup_detour_m: f32,
    pub spray_hold_share: f32,
    pub blind_sight: f32,
    pub flash_mate_deg: f32,
    pub reveal_shift_m: f32,
    pub orb_detour_m: f32,
    pub orb_stand_m: f32,
    pub object_shoot_m: f32,
    pub object_block_m: f32,
    pub wall_stuck_seconds: f32,
    pub post_clear_m: f32,
    pub cover_plant_m: f32,
    pub burst_pause_seconds: f32,
    pub over_peek: f32,
    pub hesitate_seconds: [f32; 2],
    pub peek_wait_seconds: [f32; 2],
    pub shoot_through_smoke: bool,
    pub safe_margin_m: f32,
    pub crossfire_deg: f32,
    pub retake_group: u32,
    pub retake_down: u32,
    pub retake_smoke_m: f32,
    pub handoff_m: f32,
    pub line_up_m: f32,
    pub throw_pitch_deg: [f32; 2],
    pub throw_steps: u32,
    pub throw_yaw_deg: f32,
    pub throw_work_per_tick: u32,
    pub throw_aim_deg: f32,
    pub throw_error_deg: f32,
    pub throw_moved_m: f32,
    pub strafe_seconds: [f32; 2],
    pub head_radius_m: f32,
    pub body_radius_m: f32,
    pub mate_clearance_m: f32,
    pub low_ammo_share: f32,
    pub retreat_health: f32,
    pub cover_search_m: f32,
    pub walk_near_m: f32,
    pub preaim_m: f32,
    pub angle_look_seconds: f32,
    pub angle_forget_seconds: f32,
    pub angle_scan_seconds: f32,
    pub arrive_m: f32,
    pub repath_m: f32,
    pub stuck_seconds: f32,
    pub stuck_progress_m: f32,
    pub unstick_m: f32,
    pub hold_scan_seconds: [f32; 2],
    pub hold_scan_deg: f32,
    pub heard_seconds: f32,
    pub call_repeat_seconds: f32,
    pub hurt_call_health: f32,
    pub stage_back_m: f32,
    pub gather_seconds: [f32; 2],
    pub go_stagger_seconds: f32,
    pub utility_lead_seconds: f32,
    pub default_pick_share: [f32; 2],
    pub approach_weights: [f32; 3],
    pub late_round_seconds: f32,
    pub save_gun_cost: u32,
    pub save_spend: u32,
    pub follow_human_m: f32,
    pub knife_path_m: f32,
    pub cast_clear_seconds: f32,
    pub cast_clear_m: f32,
    pub buy_late_seconds: f32,
    pub line_of_fire_m: f32,
    pub line_of_fire_deg: f32,
    pub line_of_fire_seconds: f32,
    pub ask_device_m: f32,
    pub ask_device_deg: f32,
    pub sniper_spare: u32,
    pub eco_gun_cost: u32,
    pub rotate_seen: u32,
    pub info_seconds: f32,
    pub site_near_m: f32,
    pub clear_seconds: f32,
    pub defuse_margin_seconds: f32,
    pub plan_seconds: f32,
    pub heal_below: f32,
    pub flash_mate_m: f32,
    pub flash_inside_m: f32,
    pub smoke_open_step_m: f32,
    pub utility_aim_deg: f32,
    pub utility_seconds: f32,
    pub lose_odds: [u32; 2],
    pub smoke_spray_seconds: f32,
    pub blind_retreat: f32,
    pub mid_share: f32,
    pub reach_m: f32,
    pub merge_m: f32,
    pub gate_step_m: f32,
    pub spacing_m: f32,
    pub idle_turn_share: f32,
    pub burst_pause_stretch: f32,
    pub burst_hold_share: f32,
    pub preaim_front: f32,
    pub hold_lost_share: f32,
    pub push_count: u32,
    pub utility_repeat_seconds: f32,
    pub utility_patience: f32,
    pub flash_shift_m: f32,
    pub flash_avert_seconds: f32,
    pub near_m: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct BotPlayer {
    pub run_speed_mps: f32,
    pub eye_height_m: f32,
    pub crouch_eye_height_m: f32,
    pub radius_m: f32,
    pub footstep_heard_m: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct BotRound {
    pub plant_seconds: f32,
    pub defuse_seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct BotsTable {
    pub levels: Levels,
    pub play: PlayNumbers,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct BotCombat {
    pub deadzone_share: f32,
}

#[derive(Deserialize)]
struct NumbersFile {
    bots: BotsTable,
    player: BotPlayer,
    #[serde(rename = "match")]
    round: BotRound,
    combat: BotCombat,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BotNumbers {
    pub levels: Levels,
    pub play: PlayNumbers,
    pub player: BotPlayer,
    pub round: BotRound,
    pub combat: BotCombat,
    pub ticks_per_second: u32,
}

impl BotNumbers {
    pub fn from_numbers(
        whole: &Value,
        ticks_per_second: u32,
    ) -> Result<BotNumbers, serde_json::Error> {
        let file = NumbersFile::deserialize(whole)?;
        Ok(BotNumbers {
            levels: file.bots.levels,
            play: file.bots.play,
            player: file.player,
            round: file.round,
            combat: file.combat,
            ticks_per_second,
        })
    }

    pub fn level(&self, level: Level) -> &LevelNumbers {
        match level {
            Level::Easy => &self.levels.easy,
            Level::Normal => &self.levels.normal,
            Level::Hard => &self.levels.hard,
        }
    }

    pub fn ticks(&self, seconds: f32) -> u64 {
        (engine_math::max(seconds, 0.0) * self.ticks_per_second as f32).round() as u64
    }

    pub fn seconds(&self, ticks: u64) -> f32 {
        ticks as f32 / self.ticks_per_second as f32
    }

    pub fn dt(&self) -> f32 {
        1.0 / self.ticks_per_second as f32
    }
}
