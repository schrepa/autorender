use serde::Serialize;

use crate::config::MESH_VARIETIES;
use crate::scene::{EFFECTS, LOOKS, SCENE_LOOK, VIEWS};
use crate::{Preset, StressSample, StressScene, presets};

#[derive(Clone, Debug, Serialize)]
pub struct Choice {
    pub value: String,
    pub label: String,
}

fn choice(value: impl Into<String>, label: impl Into<String>) -> Choice {
    Choice {
        value: value.into(),
        label: label.into(),
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Catalog {
    pub presets: Vec<Preset>,
    pub scenes: Vec<Choice>,
    pub looks: Vec<Choice>,
    pub effects: Vec<Choice>,
    pub views: Vec<Choice>,
    pub mesh_varieties: Vec<u32>,
    pub sample: Vec<&'static str>,
}

pub fn catalog() -> Catalog {
    Catalog {
        presets: presets(),
        scenes: StressScene::ALL
            .iter()
            .map(|scene| choice(scene.name(), scene.label()))
            .collect(),
        looks: std::iter::once(choice(SCENE_LOOK.0, SCENE_LOOK.1))
            .chain(LOOKS.iter().map(|&(name, label, ..)| choice(name, label)))
            .collect(),
        effects: [choice("all", "All effects"), choice("none", "None")]
            .into_iter()
            .chain(
                EFFECTS
                    .iter()
                    .map(|&(name, what)| choice(format!("no-{name}"), format!("No {what}"))),
            )
            .collect(),
        views: VIEWS
            .iter()
            .map(|&(name, label)| choice(name, label))
            .collect(),
        mesh_varieties: MESH_VARIETIES.to_vec(),
        sample: StressSample::FIELDS.to_vec(),
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Switch {
    pub name: &'static str,
    pub values: Vec<String>,
    pub default: &'static str,
    pub what: &'static str,
}

pub fn switches() -> Vec<Switch> {
    let catalog = catalog();
    let values = |choices: &[Choice]| choices.iter().map(|choice| choice.value.clone()).collect();
    let on_off = || vec!["on".to_owned(), "off".to_owned()];
    vec![
        Switch {
            name: "look",
            values: values(&catalog.looks),
            default: SCENE_LOOK.0,
            what: "lighting preset over the scene's own",
        },
        Switch {
            name: "effects",
            values: values(&catalog.effects),
            default: "all",
            what: "all, none, or comma-separated no-<effect> exclusions",
        },
        Switch {
            name: "visibility",
            values: on_off(),
            default: "off",
            what: "on: shade virtual geometry through the visibility buffer",
        },
        Switch {
            name: "lod",
            values: on_off(),
            default: "on",
            what: "off: draw every instance as its mesh",
        },
        Switch {
            name: "impostors",
            values: on_off(),
            default: "on",
            what: "off: draw no impostors",
        },
    ]
}
