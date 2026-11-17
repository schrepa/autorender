use std::borrow::Cow;
use std::collections::BTreeMap;

use engine_core::{
    Cut, Expected, ExtraRules, InvalidSetting, TextureHandle, fields, settings, stats,
};

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "CoverClump")]
    pub struct CoverClump {
        pub size: f32 = 1.0 => range(0.1, 8.0),
        pub height: f32 = 0.4 => range(0.0, 1.0),
        pub lean: f32 = 0.35 => range(0.0, 1.0),
        pub fan: f32 = 0.0 => range(0.0, 1.0),
        pub short: [f32; 3] = [0.7, 0.7, 0.7] => range(0.0, 8.0),
        pub tall: [f32; 3] = [1.3, 1.3, 1.3] => range(0.0, 8.0),
    }
}

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "CoverPatch")]
    pub struct CoverPatch {
        pub size: f32 = 20.0 => range(1.0, 200.0),
        pub density: f32 = 0.0 => range(0.0, 2.0),
        pub height: f32 = 0.0 => range(0.0, 1.0),
    }
}

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "CoverTuft")]
    pub struct CoverTuft {
        pub size: f32 = 0.25 => range(0.02, 8.0),
        pub gather: [f32; 2] = [0.0, 0.0] => range(0.0, 1.0),
        pub fan: [f32; 2] = [0.0, 0.0] => range(0.0, 1.0),
    }
}

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "CoverBlade")]
    pub struct CoverBladeForm {
        pub blunt: f32 = 0.0 => range(0.0, 1.0),
        pub bent: f32 = 0.0 => range(0.0, 1.0),
        pub fold: f32 = 0.45 => range(0.0, 2.0),
        pub widths: f32 = 0.0 => range(0.0, 2.0),
    }
}

fields! {
    extra_rules;
    #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "CoverPlants")]
    pub struct CoverPlants {
        pub atlas: String = String::new() => any,
        pub cells: Vec<String> = Vec::new() => any,
        pub bend: f32 = 0.2 => range(0.0, 1.0),
        pub rise: [f32; 2] = [90.0, 90.0] => range(0.0, 90.0),
        pub lift: [f32; 2] = [0.0, 0.0] => range(0.0, 4.0),
        pub count: [u32; 2] = [1, 1] => range(1, 64),
        pub arc: f32 = 360.0 => range(0.0, 360.0),
    }
}

impl ExtraRules for CoverPlants {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.cells.is_empty() {
            return Err(InvalidSetting::new("cells", Expected::NonEmpty));
        }
        if self.rise[0] > self.rise[1] {
            return Err(InvalidSetting::new("rise", Expected::Sorted));
        }
        if self.lift[0] > self.lift[1] {
            return Err(InvalidSetting::new("lift", Expected::Sorted));
        }
        if self.count[0] > self.count[1] {
            return Err(InvalidSetting::new("count", Expected::Sorted));
        }
        Ok(())
    }
}

fields! {
    #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "CoverAccent")]
    pub struct CoverAccent {
        pub look: String = String::new() => any,
        pub share: f32 = 0.0 => range(0.0, 1.0),
    }
}

fields! {
    extra_rules;
    #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "Cover")]
    pub struct CoverLook {
        pub density: f32 = 90.0 => range(0.0, 400.0),
        pub height: [f32; 2] = [0.12, 0.4] => range(0.01, 4.0),
        pub width: f32 = 0.025 => range(0.001, 0.25),
        pub tint: [f32; 3] = [1.0, 1.0, 1.0] => range(0.0, 8.0),
        pub sway: f32 = 0.6 => range(0.0, 1.0),
        pub spread: f32 = 0.3 => range(0.0, 1.0),
        pub smallest: f32 = 1.0 => range(1.0, 64.0),
        pub clump: CoverClump = CoverClump::default() => any,
        pub patch: CoverPatch = CoverPatch::default() => any,
        pub tuft: CoverTuft = CoverTuft::default() => any,
        pub blade: CoverBladeForm = CoverBladeForm::default() => any,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub plants: Option<CoverPlants> = None => any,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        pub accents: Vec<CoverAccent> = Vec::new() => any,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        pub beside: Vec<String> = Vec::new() => any,
    }
}

impl ExtraRules for CoverLook {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.height[0] > self.height[1] {
            return Err(InvalidSetting::new("height", Expected::Sorted));
        }
        if self.tuft.gather[0] > self.tuft.gather[1] {
            return Err(InvalidSetting::new("tuft.gather", Expected::Sorted));
        }
        if self.accents.iter().map(|accent| accent.share).sum::<f32>() > 1.0 {
            return Err(InvalidSetting::new("accents", Expected::Normalized));
        }
        Ok(())
    }
}

fields! {
    #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "CoverCell")]
    pub struct CoverAtlasCell {
        pub name: String = String::new() => any,
        pub x: u32 = 0 => any,
        pub y: u32 = 0 => any,
        pub w: u32 = 1 => at_least(1),
        pub h: u32 = 1 => at_least(1),
        pub root: [u32; 2] = [0, 0] => any,
    }
}

fields! {
    extra_rules;
    #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "CoverAtlas")]
    pub struct CoverAtlas {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub albedo: Option<TextureHandle> = None => any,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub normal: Option<TextureHandle> = None => any,
        pub size: [u32; 2] = [1024, 1024] => range(1, 8192),
        pub cells: Vec<CoverAtlasCell> = Vec::new() => any,
    }
}

impl ExtraRules for CoverAtlas {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.albedo.is_none() {
            return Err(InvalidSetting::new("albedo", Expected::Present));
        }
        for (index, cell) in self.cells.iter().enumerate() {
            let inside = cell.x.saturating_add(cell.w) <= self.size[0]
                && cell.y.saturating_add(cell.h) <= self.size[1];
            let rooted = (cell.x..=cell.x.saturating_add(cell.w)).contains(&cell.root[0])
                && (cell.y..=cell.y.saturating_add(cell.h)).contains(&cell.root[1]);
            if !inside || !rooted {
                return Err(InvalidSetting {
                    field: Cow::Owned(format!("cells[{index}]")),
                    ..InvalidSetting::new("cells", Expected::Matching).got(&cell.name)
                });
            }
        }
        Ok(())
    }
}

fields! {
    extra_rules;
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "CoverShape")]
    pub struct CoverShape {
        pub curl: [f32; 2] = [0.25, 0.45] => range(0.0, 2.0),
        pub taper: f32 = 1.5 => range(0.25, 16.0),
        pub knee: f32 = 0.6 => range(0.0, 1.0),
        pub drop: f32 = 0.45 => range(0.0, 1.0),
        pub reach: f32 = 0.75 => range(0.0, 2.0),
        pub most_bend: f32 = 0.95 => range(0.0, 1.4),
        pub blunt_width: f32 = 0.45 => range(0.0, 1.0),
        pub blunt_round: f32 = 3.0 => range(0.25, 16.0),
        pub bent_more: f32 = 0.45 => range(0.0, 1.4),
        pub bent_most: f32 = 1.25 => range(0.0, 1.4),
        pub bent_knee: f32 = 0.8 => range(0.0, 1.0),
        pub tip_color: f32 = 0.7 => range(0.01, 1.0),
        pub risen: f32 = 0.833 => range(0.01, 1.0),
        pub straight_pixels: f32 = 12.0 => range(0.0, 4096.0),
        pub flat_pixels: f32 = 4.0 => range(0.0, 4096.0),
    }
}

impl ExtraRules for CoverShape {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.curl[0] > self.curl[1] {
            return Err(InvalidSetting::new("curl", Expected::Sorted));
        }
        if self.flat_pixels > self.straight_pixels {
            return Err(InvalidSetting::new("flat_pixels", Expected::Sorted));
        }
        Ok(())
    }
}

fields! {
    extra_rules;
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "CoverLight")]
    pub struct CoverLight {
        pub root_light: f32 = 1.0 => range(0.0, 4.0),
        pub tip_light: f32 = 1.0 => range(0.0, 4.0),
        pub sheen: f32 = 1.0 => range(0.05, 1.0),
        pub stand_fade: [f32; 2] = [8.0, 40.0] => range(0.0, 10000.0),
    }
}

impl ExtraRules for CoverLight {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.stand_fade[0] >= self.stand_fade[1] {
            return Err(InvalidSetting::new("stand_fade", Expected::Sorted));
        }
        Ok(())
    }
}

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "CoverCards")]
    pub struct CoverCards {
        pub cutoff: f32 = 0.5 => range(0.0, 1.0),
        pub turn: f32 = 0.0 => range(0.0, 1.0),
        pub round: f32 = 0.0 => range(0.0, 2.0),
        pub relief: f32 = 1.0 => range(0.0, 4.0),
        pub root_light: f32 = 1.0 => range(0.0, 1.0),
    }
}

settings! {
    #[section("ground_cover")]
    #[check(looks_checked)]
    #[switch(on)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneGroundCover")]
    pub struct GroundCover {
        pub shadows: bool = false => any,
        pub density: f32 = 1.0 => range(0.0, 4.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_blades: Option<u32> = None => budget,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_cards: Option<u32> = None => budget,
        pub up: [f32; 2] = [0.3, 0.6] => range(0.0, 1.0),
        pub roughness: f32 = 0.7 => range(0.0, 1.0),
        pub lush: f32 = 2.0 => range(1.0, 8.0),
        pub lush_tint: [f32; 3] = [0.67, 0.8, 0.89] => range(0.01, 1.0),
        pub full_wind: f32 = 22.5 => range(0.5, 100.0),
        pub layers: f32 = 3.0 => range(1.0, 16.0),
        pub grow_in: f32 = 0.25 => range(0.01, 1.0),
        pub shape: CoverShape = CoverShape::default() => any,
        pub light: CoverLight = CoverLight::default() => any,
        pub cards: CoverCards = CoverCards::default() => any,
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        pub atlases: BTreeMap<String, CoverAtlas> = BTreeMap::new() => any,
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        pub looks: BTreeMap<String, CoverLook> = BTreeMap::new() => any,
    }
}

fn within(parent: String, invalid: InvalidSetting) -> InvalidSetting {
    InvalidSetting {
        field: Cow::Owned(format!("{parent}.{}", invalid.field)),
        detail: invalid.detail,
    }
}

fn missing(field: String, name: &str) -> InvalidSetting {
    InvalidSetting {
        field: Cow::Owned(field),
        ..InvalidSetting::new("look", Expected::Present).got(name)
    }
}

fn plants_checked(cover: &GroundCover, name: &str, look: &CoverLook) -> Result<(), InvalidSetting> {
    let Some(plants) = &look.plants else {
        return Ok(());
    };
    let Some(atlas) = cover.atlases.get(&plants.atlas) else {
        return Err(missing(format!("looks.{name}.plants.atlas"), &plants.atlas));
    };
    plants
        .validate()
        .map_err(|invalid| within(format!("looks.{name}.plants"), invalid))?;
    match plants
        .cells
        .iter()
        .position(|cell| !atlas.cells.iter().any(|known| known.name == *cell))
    {
        Some(index) => Err(missing(
            format!("looks.{name}.plants.cells[{index}]"),
            &plants.cells[index],
        )),
        None => Ok(()),
    }
}

fn looks_checked(cover: &GroundCover) -> Result<(), InvalidSetting> {
    cover
        .shape
        .validate()
        .map_err(|invalid| within("shape".to_owned(), invalid))?;
    cover
        .light
        .validate()
        .map_err(|invalid| within("light".to_owned(), invalid))?;
    cover
        .cards
        .validate()
        .map_err(|invalid| within("cards".to_owned(), invalid))?;
    for (name, atlas) in &cover.atlases {
        atlas
            .validate()
            .map_err(|invalid| within(format!("atlases.{name}"), invalid))?;
    }
    for (name, look) in &cover.looks {
        look.validate()
            .map_err(|invalid| within(format!("looks.{name}"), invalid))?;
        plants_checked(cover, name, look)?;
        if let Some(index) = look
            .accents
            .iter()
            .position(|accent| !cover.looks.contains_key(&accent.look))
        {
            return Err(missing(
                format!("looks.{name}.accents[{index}].look"),
                &look.accents[index].look,
            ));
        }
        if let Some(index) = look
            .beside
            .iter()
            .position(|beside| !cover.looks.contains_key(beside))
        {
            return Err(missing(
                format!("looks.{name}.beside[{index}]"),
                &look.beside[index],
            ));
        }
    }
    Ok(())
}

stats! {
    #[section("ground_cover_stats")]
    pub struct GroundCoverStats {
        pub blade_cut: Cut,
        pub card_cut: Cut,
        pub reach: u32,
        pub tiers: u32,
        pub cells: u32,
        pub strips: u32,
        pub flats: u32,
        pub cards: u32,
        pub tried: u32,
        pub scanned: u32,
        pub ran: u32,
        pub grew: u32,
    }
}
