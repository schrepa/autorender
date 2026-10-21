//! Cooking navigation, while a session cooks ([`Cooking`]): for each
//! agent's terrain and gait, every tile's portals of the walk graph, as
//! records the host keeps for its pack, asked for row by row and dropped as
//! they arrive. The tiles' records walks read their grounds from cook with
//! their terrain.

use super::*;

/// A walk graph cooking: its terrain, as it stood when cooking began, its
/// gait, its tiles in order, the next to ask for and the records asked for
/// and not yet back.
pub(super) struct CookGraph {
    terrain: WalkedTerrain,
    gait: Gait,
    tiles: Vec<TileKey>,
    next: usize,
    flight: Vec<JobId>,
}

impl NavAgents {
    /// Cooks the walk graph of every agent's terrain at its gait, asking
    /// for records as `jobs` admits them; says in `cooking` how many are
    /// left: a terrain under an agent still baking counts one, and each
    /// tile's portals not back yet one.
    pub(super) fn cook(
        &mut self,
        terrains: &Terrains,
        agents: &Agents,
        jobs: &mut Jobs,
        cooking: &mut Cooking,
    ) {
        let mut left = 0;
        for (_, agent, _, transform) in agents.iter() {
            let at = Vec2::new(transform.translation.x, transform.translation.z);
            match terrains.walked_terrain(at, at) {
                Walked::Ready(terrain) => {
                    let gait = agent.sanitized().options().gait(&terrain.pyramid);
                    if !self
                        .cook
                        .iter()
                        .any(|graph| graph.terrain.entity == terrain.entity && graph.gait == gait)
                    {
                        self.cook.push(CookGraph {
                            tiles: terrain.pyramid.level_tiles(gait.level).collect(),
                            terrain,
                            gait,
                            next: 0,
                            flight: Vec::new(),
                        });
                    }
                }
                Walked::Baking => left += 1,
                Walked::Off => {}
            }
        }
        let mut slots = JobBudget::of(jobs);
        for graph in &mut self.cook {
            graph.flight.retain(|&id| {
                let Some(output) = slots.jobs.take(id) else {
                    return true;
                };
                let made = job_output(
                    PortalsJob::KIND.name(),
                    output,
                    engine_terrain::jobs::portals_output,
                );
                if let Err(error) = made {
                    self.failures.record(self.tick, error);
                }
                false
            });
            while let Some(&key) = graph.tiles.get(graph.next) {
                let priority = crate::job_priority(crate::IDLE_JOBS, graph.next as u32);
                if !slots.take(priority) {
                    break;
                }
                let (input, record) = graph.terrain.portals_record(key, &graph.gait);
                graph.flight.push(slots.jobs.submit_record(
                    PortalsJob::KIND.name(),
                    input,
                    priority,
                    record,
                ));
                graph.next += 1;
            }
            left += (graph.tiles.len() - graph.next + graph.flight.len()) as u64;
        }
        cooking.left("navigation", left);
    }

    /// The jobs of the records cooking asked for and not yet back.
    pub(super) fn cooking_jobs(&self) -> impl Iterator<Item = JobId> + '_ {
        self.cook
            .iter()
            .flat_map(|graph| graph.flight.iter().copied())
    }
}
