use engine::prelude::*;
use engine_touch::{LookPadSpec, TouchAnchor, TouchButtonSpec, TouchLayout, VirtualStickSpec};

use crate::play::flow::{Flow, Stage};
use crate::play::touch as id;
use crate::rules::Phase;

use super::Showing;
use super::hud::HudFacts;

fn button(
    id: u32,
    label: &str,
    anchor: TouchAnchor,
    offset_dp: [f32; 2],
    radius_dp: f32,
) -> TouchButtonSpec {
    TouchButtonSpec {
        id,
        label: label.to_owned(),
        anchor,
        offset_dp,
        radius_dp,
        drag_aim: false,
        accent: false,
        active: false,
        visible: true,
    }
}

fn stick_spec(portrait: bool) -> VirtualStickSpec {
    VirtualStickSpec {
        anchor_dp: if portrait {
            [84.0, 156.0]
        } else {
            [92.0, 114.0]
        },
        radius_dp: 52.0,
        knob_radius_dp: 22.0,
        ..VirtualStickSpec::default()
    }
}

fn combat_buttons(facts: &HudFacts, portrait: bool) -> Vec<TouchButtonSpec> {
    let mut fire = button(
        id::FIRE,
        "FIRE",
        TouchAnchor::BottomRight,
        if portrait { [56.0, 156.0] } else { [62.0, 116.0] },
        if portrait { 35.0 } else { 36.0 },
    );
    fire.drag_aim = true;
    fire.accent = true;

    let mut scope = button(
        id::SCOPE,
        "ADS",
        TouchAnchor::BottomRight,
        if portrait { [126.0, 190.0] } else { [134.0, 148.0] },
        if portrait { 23.0 } else { 24.0 },
    );
    scope.active = facts.scope_ease > 0.0;

    let jump = button(
        id::JUMP,
        "JMP",
        TouchAnchor::BottomRight,
        if portrait { [40.0, 238.0] } else { [42.0, 194.0] },
        if portrait { 23.0 } else { 24.0 },
    );
    let crouch = button(
        id::CROUCH,
        "CRH",
        TouchAnchor::BottomRight,
        if portrait { [126.0, 124.0] } else { [134.0, 84.0] },
        if portrait { 21.0 } else { 22.0 },
    );
    let reload = button(
        id::RELOAD,
        "RLD",
        TouchAnchor::BottomRight,
        if portrait { [98.0, 250.0] } else { [102.0, 202.0] },
        if portrait { 21.0 } else { 22.0 },
    );
    let swap = button(
        id::SWAP,
        "SWP",
        TouchAnchor::BottomRight,
        if portrait { [40.0, 298.0] } else { [190.0, 118.0] },
        21.0,
    );

    let mut interact = button(
        id::USE,
        "USE",
        TouchAnchor::BottomRight,
        if portrait { [98.0, 308.0] } else { [190.0, 180.0] },
        22.0,
    );
    let live_use = matches!(facts.phase, Some(Phase::Live | Phase::Planted));
    interact.visible = facts.pickup.active || facts.action.active || live_use;
    interact.accent = facts.pickup.active || facts.action.active;

    let pause = button(id::PAUSE, "||", TouchAnchor::TopRight, [32.0, 32.0], 20.0);
    let mut buy = button(id::BUY, "BUY", TouchAnchor::TopRight, [82.0, 32.0], 22.0);
    buy.accent = true;
    buy.visible = facts.phase == Some(Phase::Buy);

    let mut buttons = vec![
        fire, scope, jump, crouch, reload, swap, interact, pause, buy,
    ];
    let ability_y = if portrait { 74.0 } else { 76.0 };
    let ability_r = if portrait { 19.0 } else { 20.0 };
    let ability_specs = [
        (id::ABILITY_C, "C", -84.0, 0usize),
        (id::ABILITY_Q, "Q", -28.0, 1usize),
        (id::ABILITY_E, "E", 28.0, 2usize),
        (id::ABILITY_X, "X", 84.0, 3usize),
    ];
    for (button_id, label, x, index) in ability_specs {
        let ability = &facts.abilities[index];
        let mut spec = button(
            button_id,
            label,
            TouchAnchor::BottomCenter,
            [x, ability_y],
            ability_r,
        );
        spec.active = ability.equipped || ability.ready;
        spec.visible = !ability.empty || ability.ready || ability.equipped;
        buttons.push(spec);
    }
    buttons
}

pub(super) fn sync(
    flow: Option<Res<Flow>>,
    showing: Res<Showing>,
    facts: Res<HudFacts>,
    views: Res<Views>,
    mut layout: ResMut<TouchLayout>,
) {
    let playing = flow
        .as_deref()
        .is_none_or(|flow| flow.stage == Stage::Playing && !flow.settings_open);
    let portrait = views.frame().portrait();
    let next = if !playing {
        TouchLayout::default()
    } else if showing.buy_open || showing.board_held {
        let mut close = button(id::CLOSE, "X", TouchAnchor::TopRight, [32.0, 32.0], 22.0);
        close.accent = true;
        TouchLayout {
            enabled: true,
            stick: None,
            look: None,
            buttons: vec![close],
        }
    } else {
        TouchLayout {
            enabled: true,
            stick: Some(stick_spec(portrait)),
            look: Some(LookPadSpec::default()),
            buttons: combat_buttons(&facts, portrait),
        }
    };
    layout.set_if_neq(next);
}
