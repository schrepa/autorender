use std::any::TypeId;

use bevy_ecs::world::World;
use engine_registry::{BrickRecord, InstallCore};

use super::*;
use crate::registration::PartRecord;

impl Renderer {
    pub fn install(&mut self, world: &mut World, bricks: &[&BrickRecord]) -> Result<(), Vec<Error>> {
        let mut ordered: Vec<&BrickRecord> = bricks.to_vec();
        ordered.sort_by(|first, second| first.package.cmp(second.package));
        let mut kept: Vec<HeldPart> = std::mem::take(&mut self.parts);
        let mut parts: Vec<HeldPart> = Vec::new();
        let mut problems: Vec<Error> = Vec::new();
        let scope = crate::gpu_calls::validation_scope(&self.core.device);
        for brick in ordered {
            let records = brick
                .render
                .iter()
                .filter_map(|entry| entry.downcast_ref::<PartRecord>());
            for record in records {
                if parts.iter().any(|held| held.part == record.part) {
                    continue;
                }
                if let Some(at) = kept.iter().position(|held| held.part == record.part) {
                    let mut held = kept.swap_remove(at);
                    held.held.refresh(world);
                    parts.push(held);
                    continue;
                }
                let _owner = crate::memory::owned(brick.package);
                let mut core = InstallCore {
                    part: parts.len() as u32,
                };
                match (record.install)(world, &mut core) {
                    Ok(held) => parts.push(HeldPart {
                        brick: brick.package,
                        part: record.part,
                        held,
                    }),
                    Err(error) => problems.push(error),
                }
            }
        }
        self.installing = Some(Box::pin(scope.pop()));
        if !problems.is_empty() {
            return Err(problems);
        }
        self.parts = parts;
        self.plans.clear();
        Ok(())
    }

    pub(crate) fn part_of(&self, part: TypeId) -> Option<u32> {
        self.parts
            .iter()
            .position(|held| held.part == part)
            .map(|index| index as u32)
    }
}
