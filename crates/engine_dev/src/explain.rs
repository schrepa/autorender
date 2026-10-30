use engine_core::bevy_ecs::prelude::{Entity, World};
use engine_core::{
    AssetLibrary, BrickCost, BrickCosts, BrickRegistry, DrawState, ExplainRecord, ExplainRequest,
    ExplainedEntity, Mesh3d,
};
use engine_scene::{SceneDocument, SceneFiles, SceneInstance};
use engine_settings::{Error, ErrorCode, Expected};
use serde_json::Value;

use crate::DevTools;
use crate::protocol::{Detail, MAX_EXPLAINED};
use crate::results::ExplainResult;

#[derive(Clone, Copy, Debug)]
pub struct EntityFacts {
    pub slot: Option<u32>,
    pub drawn: DrawState,
    pub frustum_culling: bool,
}

impl DevTools {
    pub fn explained_entity(
        &self,
        world: &World,
        scene: Option<&SceneInstance>,
        id: &str,
    ) -> Result<Entity, Error> {
        crate::inspect::validate_name(id, "$.entity")?;
        let scene = scene.ok_or_else(|| {
            Error::new(ErrorCode::NoScene, "$.entity")
                .expected(Expected::Entity)
                .got(id)
        })?;
        let entity = scene.authored_entity(world, id).ok_or_else(|| {
            Error::new(ErrorCode::UnknownEntity, "$.entity")
                .expected(Expected::Entity)
                .got(id)
        })?;
        if world.get_entity(entity).is_err() {
            return Err(Error::new(ErrorCode::EntityRemoved, "$.entity")
                .expected(Expected::Live)
                .got(id));
        }
        Ok(entity)
    }

    pub fn explain_entity(
        &self,
        world: &World,
        (id, entity): (&str, Entity),
        facts: EntityFacts,
        stats: &Value,
    ) -> ExplainResult {
        let mut records = vec![core_entity(world, id, entity, facts)];
        if let Some(registry) = world.get_resource::<BrickRegistry>() {
            let request = ExplainRequest::Entity(ExplainedEntity {
                id: id.to_owned(),
                entity,
                slot: facts.slot,
                drawn: facts.drawn,
            });
            records.extend(registry.explain(world, &request));
            records.extend(cuts(world, registry, stats));
        }
        finished("entity", records)
    }

    pub fn explain_frame(
        &self,
        world: &World,
        stats: &Value,
        top: usize,
        detail: Detail,
    ) -> Result<ExplainResult, Error> {
        let costs = world.get_resource::<BrickCosts>().ok_or_else(|| {
            Error::new(ErrorCode::ProfilingUnavailable, "$")
                .expected(Expected::Present)
                .got("BrickCosts")
        })?;
        let mut records = slow_frame(costs, top, detail);
        if let Some(registry) = world.get_resource::<BrickRegistry>() {
            let request = ExplainRequest::Frame {
                stats: stats.clone(),
            };
            records.extend(registry.explain(world, &request));
            records.extend(cuts(world, registry, stats));
        }
        Ok(finished("frame", records))
    }

    pub fn explain_scene(
        &self,
        world: &World,
        source: &str,
        files: Option<&SceneFiles>,
    ) -> ExplainResult {
        let records = refusal(world, source, files)
            .map(|error| refused(world, &error))
            .into_iter()
            .collect();
        finished("scene", records)
    }
}

fn finished(about: &'static str, mut records: Vec<ExplainRecord>) -> ExplainResult {
    let total = records.len();
    records.truncate(MAX_EXPLAINED);
    ExplainResult {
        about,
        records,
        total,
    }
}

fn core_entity(world: &World, id: &str, entity: Entity, facts: EntityFacts) -> ExplainRecord {
    let mesh3d = world.get::<Mesh3d>(entity);
    let missing = mesh3d.and_then(|mesh3d| {
        world
            .get_resource::<AssetLibrary>()
            .and_then(|assets| crate::inspect::missing_of(assets, mesh3d))
    });
    let decided = match (facts.slot, facts.drawn) {
        (None, _) => "not_extracted",
        (Some(_), DrawState::Culled) => "culled",
        (Some(_), DrawState::Level(_)) => "drawn",
        (Some(_), DrawState::Unread) => "unread",
    };
    ExplainRecord::new(decided)
        .of("core")
        .with("entity", id)
        .with("render_slot", facts.slot)
        .with("mesh", mesh3d.is_some())
        .with("missing", missing)
        .with("frustum_culling", facts.frustum_culling)
}

fn cuts(world: &World, registry: &BrickRegistry, stats: &Value) -> Vec<ExplainRecord> {
    let mut records = Vec::new();
    if let Some(cut) = stats["dropped_instances"].as_u64().filter(|&cut| cut > 0) {
        records.push(
            ExplainRecord::new("cut")
                .of("core")
                .with("stat", "dropped_instances")
                .with("kept", stats["visible_instances"].as_u64())
                .with("cut", cut),
        );
    }
    for (brick, consumer) in registry.consumers() {
        let reported = consumer
            .stat
            .split('.')
            .try_fold(stats, |value, field| value.get(field));
        let Some(reported) = reported else {
            continue;
        };
        let Some(cut) = reported["cut"].as_u64().filter(|&cut| cut > 0) else {
            continue;
        };
        let limit = consumer
            .setting
            .split_once('.')
            .and_then(|(key, field)| {
                registry
                    .section(key)
                    .map(|section| section.current(world)[field].clone())
            })
            .unwrap_or(Value::Null);
        records.push(
            ExplainRecord::new("cut")
                .of(brick)
                .with("consumer", consumer.name)
                .with("setting", consumer.setting)
                .with("limit", limit)
                .with("asked", reported["asked"].as_u64())
                .with("kept", reported["kept"].as_u64())
                .with("cut", cut),
        );
    }
    records
}

fn slow_frame(costs: &BrickCosts, top: usize, detail: Detail) -> Vec<ExplainRecord> {
    let cpu_total: f64 = costs.rows.iter().map(|row| row.cpu_ms).sum();
    let gpu_total: Option<f64> = costs
        .rows
        .iter()
        .any(|row| row.gpu_ms.is_some())
        .then(|| costs.rows.iter().filter_map(|row| row.gpu_ms).sum());
    let cost = |row: &BrickCost| row.cpu_ms + row.gpu_ms.unwrap_or(0.0);
    let mut rows: Vec<&BrickCost> = costs.rows.iter().collect();
    rows.sort_by(|a, b| cost(b).total_cmp(&cost(a)));
    let share = |part: f64, total: f64| if total > 0.0 { part / total } else { 0.0 };
    rows.into_iter()
        .take(top)
        .enumerate()
        .map(|(rank, row)| {
            let record = ExplainRecord::new("cost")
                .of(row.brick)
                .with("frame", costs.frame)
                .with("rank", rank + 1)
                .with("cpu_ms", row.cpu_ms)
                .with("cpu_share", share(row.cpu_ms, cpu_total))
                .with("gpu_ms", row.gpu_ms)
                .with(
                    "gpu_share",
                    row.gpu_ms
                        .zip(gpu_total)
                        .map(|(part, total)| share(part, total)),
                );
            match detail {
                Detail::Short => record,
                Detail::Full => record
                    .with("heap_live_bytes", row.heap_live_bytes)
                    .with("heap_peak_bytes", row.heap_peak_bytes)
                    .with("allocations", row.allocations)
                    .with("gpu_bytes", row.gpu_bytes)
                    .with("created_after_warmup", row.created_after_warmup)
                    .with("duplicate_upload_bytes", row.duplicate_upload_bytes),
            }
        })
        .collect()
}

fn refusal(world: &World, source: &str, files: Option<&SceneFiles>) -> Option<Error> {
    let document = match SceneDocument::from_json(source) {
        Ok(document) => document,
        Err(error) => return Some(error),
    };
    if let Some(files) = files
        && let Err(error) = document.validate_with(files)
    {
        return Some(error);
    }
    document
        .environment
        .as_ref()
        .and_then(|environment| environment.check(world, "$.environment").err())
}

fn refused(world: &World, error: &Error) -> ExplainRecord {
    let brick = error
        .path
        .strip_prefix("$.environment.")
        .and_then(|rest| rest.split(['.', '[']).next())
        .and_then(|key| {
            world
                .get_resource::<BrickRegistry>()?
                .section(key)
                .map(|section| section.brick)
        })
        .unwrap_or("scene");
    ExplainRecord::new("refused")
        .of(brick)
        .with("code", error.code)
        .with("path", &error.path)
        .with("expected", error.detail.expected)
        .with("allowed", &error.detail.allowed)
        .with("nearest", &error.detail.nearest)
        .with("got", &error.detail.got)
}
