use super::*;

pub(super) struct CookGraph {
    walked: Walked,
    gait: Gait,
    tiles: Vec<TileKey>,
    next: usize,
    pub(super) flight: Vec<Ticket>,
}

impl NavAgents {
    pub(super) fn cook(
        &mut self,
        ground: &Ground,
        agents: &Agents,
        (records, jobs): (&Records, &mut Jobs),
    ) -> u64 {
        let mut left = 0;
        for (_, agent, _, transform) in agents.iter() {
            let at = Vec2::new(transform.translation.x, transform.translation.z);
            match ground_source(ground, at, at) {
                GroundRead::Ready(walked) => {
                    let gait = options(&sanitized(agent)).gait(&walked.source.pyramid);
                    if !self.cook.iter().any(|graph| {
                        graph.walked.source.ground == walked.source.ground && graph.gait == gait
                    }) {
                        self.cook.push(CookGraph {
                            tiles: walked.source.pyramid.level_tiles(gait.level).collect(),
                            walked,
                            gait,
                            next: 0,
                            flight: Vec::new(),
                        });
                    }
                }
                GroundRead::Waiting => left += 1,
                GroundRead::Off => {}
            }
        }
        for graph in &mut self.cook {
            let mut flying = Vec::with_capacity(graph.flight.len());
            for ticket in std::mem::take(&mut graph.flight) {
                let Some(output) = records.get(&ticket) else {
                    flying.push(ticket);
                    continue;
                };
                jobs.cancel(ticket);
                if let Err(error) = made(PORTALS_NAME, output, portals_output) {
                    self.failures.record(self.tick, error);
                }
            }
            graph.flight = flying;
            while let Some(&key) = graph.tiles.get(graph.next) {
                let priority = job_priority(IDLE_JOBS, graph.next as u32);
                let input = portals_input(&graph.walked, key, &graph.gait);
                graph.flight.push(jobs.fetch(&PORTALS, &input, priority));
                graph.next += 1;
            }
            left += (graph.tiles.len() - graph.next + graph.flight.len()) as u64;
        }
        left
    }
}
