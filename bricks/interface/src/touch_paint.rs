use engine::math;
use engine_touch::{ButtonVisual, StickVisual, TouchState};

use super::draw::{Align, Area, InterfaceDraw, Plate, Rgba, Stroke, TextStyle, Weight};

const BASE_FILL: Rgba = Rgba([0.07, 0.10, 0.16, 0.34]);
const BASE_ACTIVE_FILL: Rgba = Rgba([0.09, 0.14, 0.22, 0.48]);
const RING_STROKE: Rgba = Rgba([0.92, 0.96, 1.0, 0.38]);
const RING_ACTIVE_STROKE: Rgba = Rgba([0.48, 0.90, 0.78, 0.80]);
const KNOB_FILL: Rgba = Rgba([0.92, 0.96, 1.0, 0.56]);
const KNOB_ACTIVE_FILL: Rgba = Rgba([0.48, 0.90, 0.78, 0.88]);
const ACCENT_FILL: Rgba = Rgba([0.95, 0.33, 0.26, 0.56]);
const ACCENT_HELD_FILL: Rgba = Rgba([1.0, 0.44, 0.32, 0.84]);
const ACCENT_STROKE: Rgba = Rgba([1.0, 0.86, 0.78, 0.82]);
const ACTIVE_FILL: Rgba = Rgba([0.22, 0.72, 0.62, 0.54]);
const HELD_FILL: Rgba = Rgba([0.95, 0.98, 1.0, 0.46]);
const LABEL_COLOR: Rgba = Rgba([0.98, 0.99, 1.0, 0.94]);

fn disc(center: glam::Vec2, radius: f32) -> Area {
    Area {
        x: center.x - radius,
        y: center.y - radius,
        w: radius * 2.0,
        h: radius * 2.0,
    }
}

fn plain(fill: Option<Rgba>, radius: f32, stroke: Option<Stroke>) -> Plate {
    Plate {
        fill,
        radius,
        stroke,
        angle_radians: 0.0,
        soft: 1.0,
        sweep: 1.0,
    }
}

fn label_weight(draw: &InterfaceDraw) -> Weight {
    let faces = draw.font.as_ref().map_or(0, |font| font.faces.len());
    if faces > 1 { Weight(1) } else { Weight(0) }
}

fn paint_stick(draw: &mut InterfaceDraw, stick: StickVisual) {
    let ring_w = math::max(2.0, stick.radius_px * 0.045);
    let fill = if stick.active {
        BASE_ACTIVE_FILL
    } else {
        BASE_FILL
    };
    let ring = if stick.active {
        RING_ACTIVE_STROKE
    } else {
        RING_STROKE
    };
    draw.plate(
        disc(stick.center_px, stick.radius_px),
        plain(
            Some(fill),
            stick.radius_px,
            Some(Stroke {
                width: ring_w,
                inset: 0.0,
                color: ring,
            }),
        ),
    );
    let knob_fill = if stick.active {
        KNOB_ACTIVE_FILL
    } else {
        KNOB_FILL
    };
    draw.plate(
        disc(stick.knob_px, stick.knob_radius_px),
        plain(
            Some(knob_fill),
            stick.knob_radius_px,
            Some(Stroke {
                width: math::max(1.5, ring_w * 0.8),
                inset: 0.0,
                color: LABEL_COLOR,
            }),
        ),
    );
}

fn paint_button(draw: &mut InterfaceDraw, button: &ButtonVisual, weight: Weight) {
    let area = disc(button.center_px, button.radius_px);
    let fill = if button.accent {
        if button.held {
            ACCENT_HELD_FILL
        } else {
            ACCENT_FILL
        }
    } else if button.held {
        HELD_FILL
    } else if button.active {
        ACTIVE_FILL
    } else {
        BASE_FILL
    };
    let stroke_color = if button.accent {
        ACCENT_STROKE
    } else if button.held || button.active {
        RING_ACTIVE_STROKE
    } else {
        RING_STROKE
    };
    let stroke_w = math::max(2.0, button.radius_px * 0.065);
    draw.plate(
        area,
        plain(
            Some(fill),
            button.radius_px,
            Some(Stroke {
                width: stroke_w,
                inset: 0.0,
                color: stroke_color,
            }),
        ),
    );
    if button.label.is_empty() {
        return;
    }
    let scale_factor = if button.label.len() <= 2 {
        0.72
    } else if button.label.len() <= 4 {
        0.48
    } else {
        0.38
    };
    let px = math::max(draw.min_text_pixels(), button.radius_px * scale_factor);
    let width = draw.measure(&button.label, px, weight);
    let inset = button.radius_px * 0.18;
    let text_area = Area {
        x: area.x + inset,
        y: area.y,
        w: math::max(area.w - inset * 2.0, 1.0),
        h: area.h,
    };
    draw.text(
        (&button.label, width),
        text_area,
        TextStyle {
            px,
            weight,
            align: Align::Center,
            color: LABEL_COLOR,
            grow: 0.0,
            soft: 0.0,
        },
    );
}

pub(super) fn paint_touch(draw: &mut InterfaceDraw, touch: &TouchState) {
    if !touch.visible {
        return;
    }
    if let Some(stick) = touch.stick_visual() {
        paint_stick(draw, stick);
    }
    let weight = label_weight(draw);
    for button in touch.button_visuals() {
        paint_button(draw, button, weight);
    }
}
