use std::collections::BTreeMap;

use bevy_ecs::prelude::*;
use engine_core::{
    Asset, AssetId, Assets, Drawn, DrawnTable, Error, ErrorCode, FrameTime, InstanceValues,
    MeshFile, Refusals, Spawner, Stats, ViewSkip,
};
use engine_meshpack::JOINTS;

use super::{FigureForms, FigureView, FiguresBrick, FiguresStats};
use crate::play::figures::{FIGURE, Form, Forms, GROUND};
use crate::play::light::Light;
use crate::play::puppet::{Puppet, Seen};
use crate::play::skinned::{Figure, FigurePalette};
use engine_core::math::{max, min};

const FORMS_PART: &str = "figures_forms";

#[derive(Resource, Default)]
pub struct FigureForming {
    full: BTreeMap<Entity, Asset<MeshFile>>,
    fits: BTreeMap<(Asset<MeshFile>, usize), Result<(), Error>>,
}

impl FigureForming {
    fn keep(&mut self, assets: &Assets) {
        let live = |id: AssetId| assets.revision_of(id).is_some();
        self.fits.retain(|(mesh, _), _| live(mesh.id()));
    }

    fn form_fits(
        &mut self,
        joints: usize,
        (assets, mesh): (&Assets, Asset<MeshFile>),
        path: &str,
    ) -> Result<(), Error> {
        let key = (mesh, joints);
        if let Some(known) = self.fits.get(&key) {
            return known.clone();
        }
        let joints = joints as u32;
        let found = match assets.get(mesh) {
            Err(_) => Err(Error::new(ErrorCode::MissingFile, path.to_owned())),
            Ok(form) => {
                let highest = form
                    .uints4(JOINTS, path)
                    .ok()
                    .map(|lanes| lanes.iter().flatten().copied().max().unwrap_or(0));
                match highest {
                    Some(top) if top < joints => Ok(()),
                    Some(top) => Err(Error::new(ErrorCode::OutOfRange, path.to_owned())
                        .at_most(u64::from(joints.saturating_sub(1)))
                        .got(top)),
                    None => Err(Error::new(ErrorCode::InvalidRenderable, path.to_owned())
                        .got("no joints stream")),
                }
            }
        };
        self.fits.insert(key, found.clone());
        found
    }
}

fn submitted(assets: &Assets, drawn: &Drawn, geometry: Option<Asset<MeshFile>>) -> u64 {
    let count = geometry
        .and_then(|mesh| assets.get(mesh).ok())
        .map_or(0, |mesh| u64::from(mesh.description().vertex_count));
    let outlined = u64::from(drawn.values.emission != [0.0; 3]);
    count * (1 + outlined)
}

fn placed(drawn: &Drawn, base: f32, geometry: Option<Asset<MeshFile>>) -> Option<Drawn> {
    let geometry = geometry.or(drawn.geometry);
    (drawn.values.floats.first() != Some(&base) || drawn.geometry != geometry).then(|| {
        let mut floats = drawn.values.floats.clone();
        match floats.first_mut() {
            Some(first) => *first = base,
            None => floats.push(base),
        }
        Drawn {
            geometry,
            values: InstanceValues {
                floats,
                ..drawn.values.clone()
            },
            ..drawn.clone()
        }
    })
}

type PlaceReads<'w> = (
    Res<'w, FrameTime>,
    Res<'w, FigurePalette>,
    (Option<Res<'w, Seen>>, Res<'w, Assets>),
);

type DrawFigures<'w, 's> =
    Query<'w, 's, (Entity, &'static Drawn, Option<&'static Forms>), With<Figure>>;

fn held_forms(assets: &Assets, figures: &DrawFigures, stats: &mut Stats<'_, FiguresStats>) {
    stats.edit(|stats| stats.forms.clear());
    for (entity, _, forms) in figures {
        let Some(forms) = forms else {
            continue;
        };
        stats.edit(|stats| {
            stats.forms.push(FigureForms {
                entity: entity.to_string(),
                mid: assets.ready(forms.mid),
                far: assets.ready(forms.far),
            });
        });
    }
}

fn viewed(
    (seen, current): (Option<&Seen>, Option<&Seen>),
    drawn: &[Entity],
    (forms, vertices): ([u32; 3], u64),
) -> FigureView {
    let distances = drawn
        .iter()
        .filter_map(|entity| current.and_then(|seen| seen.distance(*entity)));
    let (nearest_m, farthest_m) = distances.fold((0.0, 0.0), |(nearest, farthest), distance| {
        if farthest == 0.0 {
            (distance, distance)
        } else {
            (min(nearest, distance), max(farthest, distance))
        }
    });
    FigureView {
        drawn: drawn.len() as u32,
        out_of_view: seen.map_or(0, |seen| seen.frustum),
        behind_map: seen.map_or(0, |seen| seen.map),
        held: seen.map_or(0, |seen| seen.held),
        full: forms[0],
        mid: forms[1],
        far: forms[2],
        vertices,
        nearest_m,
        farthest_m,
    }
}

fn formed(
    assets: &Assets,
    (entity, forms, drawn, form): (Entity, Option<&Forms>, &Drawn, Form),
    (joints, forming): (usize, &mut FigureForming),
) -> Result<Option<Asset<MeshFile>>, Error> {
    let Some(current) = drawn.geometry else {
        return Ok(None);
    };
    let reduced = forms.is_some_and(|forms| forms.mid == current || forms.far == current);
    let whole = if reduced {
        forming.full.get(&entity).copied().unwrap_or(current)
    } else {
        forming.full.insert(entity, current);
        current
    };
    let (handle, path) = match (form, forms) {
        (Form::Mid, Some(forms)) => (forms.mid, "$.figure_forms.mid"),
        (Form::Far, Some(forms)) => (forms.far, "$.figure_forms.far"),
        (Form::Full | Form::Mid | Form::Far, _) => return Ok(Some(whole)),
    };
    if !assets.ready(handle) {
        return Ok(Some(whole));
    }
    forming.form_fits(joints, (assets, handle), path)?;
    Ok(Some(handle))
}

fn chosen(
    assets: &Assets,
    (seen, entity, forms, drawn): (Option<&Seen>, Entity, Option<&Forms>, &Drawn),
    (joints, forming): (usize, &mut FigureForming),
    refusals: &mut Refusals,
) -> (Option<Asset<MeshFile>>, Form) {
    let form = seen.and_then(|seen| seen.form(entity)).unwrap_or_default();
    let geometry = match formed(assets, (entity, forms, drawn, form), (joints, forming)) {
        Ok(geometry) => {
            refusals.accept(entity, FORMS_PART);
            geometry
        }
        Err(error) => {
            refusals.refuse(entity, FORMS_PART, error);
            forming.full.get(&entity).copied()
        }
    };
    let shown = if geometry == forming.full.get(&entity).copied() {
        Form::Full
    } else {
        form
    };
    (geometry, shown)
}

pub(super) fn place(
    (time, palette, (seen, assets)): PlaceReads,
    (mut forming, skipped): (ResMut<FigureForming>, Query<(), With<ViewSkip>>),
    figures: DrawFigures,
    mut stats: Stats<FiguresStats>,
    (mut refusals, mut spawner): (ResMut<Refusals>, Spawner<FiguresBrick>),
) {
    refusals.keep_part(FORMS_PART, |entity| figures.contains(entity));
    if assets.is_changed() {
        forming.keep(&assets);
    }
    if !skipped.is_empty() {
        stats.edit(|stats| stats.view = FigureView::default());
        return;
    }
    held_forms(&assets, &figures, &mut stats);
    let mut forms_drawn = [0_u32; 3];
    let current = seen.as_deref().filter(|seen| seen.frame == time.frame);
    let mut vertices = 0_u64;
    let mut drawn_now = Vec::new();
    for (entity, (at, joints)) in palette.posed() {
        let Ok((_, drawn, forms)) = figures.get(entity) else {
            continue;
        };
        drawn_now.push(entity);
        let (geometry, shown) = chosen(
            &assets,
            (seen.as_deref(), entity, forms, drawn),
            (joints, &mut forming),
            &mut refusals,
        );
        forms_drawn[shown as usize] += 1;
        vertices += submitted(&assets, drawn, geometry);
        if let Some(moved) = placed(drawn, at as f32, geometry) {
            spawner.insert(entity, moved);
        }
    }
    forming.full.retain(|entity, _| figures.contains(*entity));
    stats.edit(|stats| {
        stats.view = viewed(
            (seen.as_deref(), current),
            &drawn_now,
            (forms_drawn, vertices),
        );
    });
}

pub(super) fn count(
    (table, assets): (Res<DrawnTable>, Res<Assets>),
    (lights, puppet): (Query<(), With<Light>>, Res<Puppet>),
    mut stats: Stats<FiguresStats>,
) {
    let (mut figures, mut indices) = (0_u32, 0_u64);
    for kind in [FIGURE, GROUND] {
        for batch in table.batches(kind) {
            let instances = batch.instances.len() as u64;
            if kind == FIGURE {
                figures += instances as u32;
            }
            let mesh = batch
                .geometry
                .and_then(|geometry| assets.get(geometry).ok());
            if let Some(mesh) = mesh {
                indices += u64::from(mesh.description().index_count) * instances;
            }
        }
    }
    stats.edit(|stats| {
        stats.figures = figures;
        stats.indices = indices;
        stats.triangles = indices / 3;
        stats.lit = !lights.is_empty();
        stats.maps = puppet.table.as_ref().map(|table| table.shade.maps);
    });
}
