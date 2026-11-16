import package::cover::order::hash;
import package::engine::math::unit;

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

fn grounded_tilt(normal: vec3<f32>) -> vec2<u32> {
    return vec2<u32>(round((clamp(normal.xz, vec2<f32>(-1.0), vec2<f32>(1.0)) * 0.5
        + 0.5) * 255.0));
}

fn unorm8(value: f32) -> u32 {
    return u32(floor(0.5 + 255.0 * saturate(value)));
}

fn pack_color(color: vec3<f32>, w: f32) -> u32 {
    return pack4x8unorm(vec4<f32>(sqrt(saturate(color)), saturate(w)));
}

fn blade_form(form: vec4<f32>, dice: u32) -> u32 {
    let blunt = u32(unit(dice) < form.x);
    let bent = u32(unit(hash(dice ^ 0x2545f491u)) < form.y);
    let fold = u32(round(saturate(form.z / FOLD_WIDEST) * FOLD_STEPS));
    return blunt | (bent << 1u) | (fold << 2u);
}
