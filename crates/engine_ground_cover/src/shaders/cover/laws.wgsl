const STRAIGHT_JOINTS: u32 = 3u;
const FLAT_JOINTS: u32 = 2u;
const CARD_JOINTS: u32 = 4u;

const COVER_DRAW_VERTICES: u32 = 64u;

const BLADE_SIZE: u32 = 0x7fff7fffu;
const BLADE_HALF: u32 = 0x80000000u;
const BLADE_UPPER: u32 = 0x00008000u;

const FOLD_STEPS: f32 = 63.0;
const FOLD_WIDEST: f32 = 2.0;

const FORM_FOLD: u32 = 0x3fu;
const FORM_BLUNT: u32 = 0x40u;
const FORM_FLAT: u32 = 0x80u;

fn blade_taper(pointed: f32, blunt: vec2<f32>, form: u32, along: f32) -> f32 {
    if (form & FORM_FLAT) != 0u {
        return max(1.0 - 2.0 * along / (1.0 + pointed), 0.0);
    }
    if (form & FORM_BLUNT) != 0u {
        return 1.0 - (1.0 - blunt.x) * pow(along, blunt.y);
    }
    return 1.0 - pow(along, pointed);
}

const LIT_LEAST: f32 = 1.0 / 255.0;

fn blade_above(pointed: f32, along: f32) -> f32 {
    let whole = 1.0 + pointed;
    return (whole * (1.0 - along) + pow(along, whole) - 1.0) / pointed;
}

const CARD_MIRRORED: u32 = 0x80000000u;
const CARD_CELL: u32 = 0x7fffffffu;
const CARD_TINT_RANGE: f32 = 0.5;
