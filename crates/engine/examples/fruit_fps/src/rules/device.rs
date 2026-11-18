use super::{
    CreditReason, Defusing, Device, Event, Match, Phase, Planting, PlayerFacts, PlayerId, Side,
    SiteId, TickInput,
};

fn facts(input: &TickInput, player: PlayerId) -> Option<&PlayerFacts> {
    input.players.get(player)
}

impl Match {
    fn can_act(&self, player: PlayerId, side: Side) -> bool {
        self.players.get(player).is_some_and(|p| p.alive)
            && self.side_of_player(player) == Some(side)
    }

    pub(super) fn drop_device_of(
        &mut self,
        victim: PlayerId,
        input: &TickInput,
        events: &mut Vec<Event>,
    ) {
        let Device::Carried { player, planting } = self.device else {
            return;
        };
        if player != victim {
            return;
        }
        if let Some(planting) = planting {
            events.push(Event::PlantCancelled {
                player,
                site: planting.site,
            });
        }
        let at = facts(input, victim).map_or([0.0; 3], |f| f.position);
        self.device = Device::Dropped { at };
        events.push(Event::DeviceDropped { player, at });
    }

    pub(super) fn pick_up_device(&mut self, input: &TickInput, events: &mut Vec<Event>) {
        if !matches!(self.device, Device::Dropped { .. }) {
            return;
        }
        let near = |p: PlayerId| facts(input, p).is_some_and(|f| f.near_device);
        if self.dropped_by.is_some_and(|p| !near(p)) {
            self.dropped_by = None;
        }
        let carrier = (0..self.players.len())
            .find(|&p| self.can_act(p, Side::Attack) && near(p) && self.dropped_by != Some(p));
        if let Some(player) = carrier {
            self.device = Device::Carried {
                player,
                planting: None,
            };
            events.push(Event::DevicePickedUp { player });
        }
    }

    fn held_site(&self, player: PlayerId, input: &TickInput) -> Option<SiteId> {
        if !self.can_act(player, Side::Attack) {
            return None;
        }
        facts(input, player)
            .filter(|f| f.plant)
            .and_then(|f| f.site)
            .filter(|&s| s < self.catalog.sites.len())
    }

    pub(super) fn advance_plant(&mut self, input: &TickInput, events: &mut Vec<Event>) -> bool {
        let Device::Carried { player, planting } = self.device else {
            return false;
        };
        let held = self.held_site(player, input);
        let planting = match (planting, held) {
            (Some(mut p), Some(site)) if site == p.site => {
                p.progress += 1;
                p
            }
            (Some(p), _) => {
                events.push(Event::PlantCancelled {
                    player,
                    site: p.site,
                });
                self.device = Device::Carried {
                    player,
                    planting: None,
                };
                return false;
            }
            (None, Some(site)) => {
                events.push(Event::PlantStarted { player, site });
                Planting {
                    site,
                    progress: 1,
                    needed: self.catalog.timings.plant,
                }
            }
            (None, None) => return false,
        };
        if planting.progress < planting.needed {
            self.device = Device::Carried {
                player,
                planting: Some(planting),
            };
            return false;
        }
        self.complete_plant(player, planting.site, input, events);
        true
    }

    fn complete_plant(
        &mut self,
        player: PlayerId,
        site: SiteId,
        input: &TickInput,
        events: &mut Vec<Event>,
    ) {
        let at = facts(input, player).map_or([0.0; 3], |f| f.position);
        let timings = self.catalog.timings;
        self.device = Device::Planted {
            site,
            at,
            fuse_ticks_left: timings.fuse,
            defusing: Defusing {
                player: None,
                progress: 0,
                checkpoint: 0,
                needed: timings.defuse,
            },
        };
        events.push(Event::PlantCompleted {
            player,
            site,
            at,
            fuse_ticks: timings.fuse,
        });
        let bonus = i64::from(self.catalog.economy.plant_team_bonus);
        for p in 0..self.players.len() {
            if self.side_of_player(p) == Some(Side::Attack) {
                self.credit(p, bonus, CreditReason::PlantBonus, events);
            }
        }
        self.ultimate_point(player, events);
        self.enter_phase(Phase::Planted, timings.fuse, events);
    }

    fn holds_defuse(&self, player: PlayerId, input: &TickInput) -> bool {
        self.can_act(player, Side::Defend)
            && facts(input, player).is_some_and(|f| f.defuse && f.near_device)
    }

    pub(super) fn advance_defuse(&mut self, input: &TickInput, events: &mut Vec<Event>) -> bool {
        let Device::Planted {
            site,
            at,
            fuse_ticks_left,
            mut defusing,
        } = self.device
        else {
            return false;
        };
        let defuser = match defusing.player {
            Some(p) if self.holds_defuse(p, input) => Some(p),
            Some(p) => {
                defusing.progress = defusing.checkpoint;
                defusing.player = None;
                events.push(Event::DefuseCancelled {
                    player: p,
                    kept: defusing.checkpoint,
                });
                None
            }
            None => {
                let next = (0..self.players.len()).find(|&p| self.holds_defuse(p, input));
                if let Some(p) = next {
                    events.push(Event::DefuseStarted {
                        player: p,
                        progress: defusing.progress,
                    });
                }
                next
            }
        };
        if let Some(player) = defuser {
            defusing.player = Some(player);
            defusing.progress += 1;
            if let Some(half) = self.catalog.timings.defuse_checkpoint
                && defusing.checkpoint < half
                && defusing.progress >= half
            {
                defusing.checkpoint = half;
                events.push(Event::DefuseHalf { player });
            }
            if defusing.progress >= defusing.needed {
                self.device = Device::Defused { site, at };
                events.push(Event::DefuseCompleted { player });
                self.ultimate_point(player, events);
                return true;
            }
        }
        self.device = Device::Planted {
            site,
            at,
            fuse_ticks_left,
            defusing,
        };
        false
    }

    pub(super) fn advance_fuse(&mut self, events: &mut Vec<Event>) -> bool {
        let Device::Planted {
            site,
            at,
            fuse_ticks_left,
            defusing,
        } = self.device
        else {
            return false;
        };
        let left = fuse_ticks_left.saturating_sub(1);
        self.phase_left = left;
        if left > 0 {
            self.device = Device::Planted {
                site,
                at,
                fuse_ticks_left: left,
                defusing,
            };
            return false;
        }
        self.device = Device::Exploded { site, at };
        events.push(Event::DeviceExploded { site, at });
        true
    }
}
