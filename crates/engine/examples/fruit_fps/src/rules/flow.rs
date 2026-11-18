use super::{
    Ability, AbilitySlot, AbilityUse, Catalog, CreditReason, Death, Device, Event, Ledger, Match,
    Phase, Player, PlayerId, RoundEndReason, RulesError, RulesNumbers, Side, Slot, Team, TickInput,
    Ultimate,
};

fn fresh_player(catalog: &Catalog, slot: &Slot) -> Result<Player, RulesError> {
    let character = catalog
        .character_id(&slot.character)
        .ok_or_else(|| RulesError::UnknownCharacter(slot.character.clone()))?;
    let entry = &catalog.characters[character];
    let bought = |i: usize| Ability {
        charges: 0,
        max: entry.bought[i].charges,
        cooldown_ticks: 0,
    };
    Ok(Player {
        team: slot.team,
        character,
        control: slot.control,
        alive: false,
        credits: 0,
        primary: None,
        sidearm: None,
        armor: None,
        abilities: [
            bought(0),
            bought(1),
            Ability {
                charges: 0,
                max: entry.signature.charges,
                cooldown_ticks: 0,
            },
        ],
        ultimate: Ultimate {
            points: 0,
            cost: entry.ultimate.cost,
            active_ticks: 0,
        },
        kills: 0,
        deaths: 0,
        assists: 0,
    })
}

fn check_roster(catalog: &Catalog, roster: &[Slot]) -> Result<(), RulesError> {
    let expected = catalog.team_size * 2;
    if roster.len() != expected {
        return Err(RulesError::RosterSize {
            found: roster.len(),
            expected,
        });
    }
    for team in Team::BOTH {
        let found = roster.iter().filter(|s| s.team == team).count();
        if found != catalog.team_size {
            return Err(RulesError::TeamSize {
                team,
                found,
                expected: catalog.team_size,
            });
        }
    }
    Ok(())
}

impl Match {
    pub fn new(
        numbers: &RulesNumbers,
        roster: &[Slot],
        seed: u64,
        ticks_per_second: u32,
    ) -> Result<Match, RulesError> {
        let catalog = Catalog::new(numbers, ticks_per_second)?;
        check_roster(&catalog, roster)?;
        let players = roster
            .iter()
            .map(|slot| fresh_player(&catalog, slot))
            .collect::<Result<Vec<_>, _>>()?;
        let mut game = Match {
            ledgers: vec![Ledger::default(); players.len()],
            players,
            catalog,
            device: Device::Dropped { at: [0.0; 3] },
            phase: Phase::Buy,
            phase_left: 0,
            round: 1,
            score: [0; 2],
            sides: [Side::Attack, Side::Defend],
            loss_streak: [0; 2],
            tick: 0,
            random: seed,
            pending: Vec::new(),
        };
        let mut events = Vec::new();
        game.reset_for_half(CreditReason::Start, &mut events);
        game.begin_round(&mut events);
        game.pending = events;
        Ok(game)
    }

    fn next_random(&mut self) -> u64 {
        self.random = self.random.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.random;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn reset_for_half(&mut self, reason: CreditReason, events: &mut Vec<Event>) {
        let start = self.catalog.economy.start;
        for player in 0..self.players.len() {
            self.set_credits(player, start, reason, events);
            let p = &mut self.players[player];
            if let Some(weapon) = p.primary.take() {
                events.push(Event::TakeWeapon { player, weapon });
            }
            if p.armor.take().is_some() {
                events.push(Event::Armor {
                    player,
                    armor: None,
                });
            }
            for (ability, slot) in
                p.abilities
                    .iter_mut()
                    .zip([AbilitySlot::C, AbilitySlot::Q, AbilitySlot::E])
            {
                if ability.charges != 0 {
                    events.push(Event::Charges {
                        player,
                        slot,
                        charges: 0,
                    });
                }
                ability.charges = 0;
                ability.cooldown_ticks = 0;
            }
            p.ultimate.points = 0;
            p.ultimate.active_ticks = 0;
            self.ledgers[player] = Ledger::default();
        }
        self.loss_streak = [0; 2];
    }

    fn begin_round(&mut self, events: &mut Vec<Event>) {
        let health = self.catalog.health;
        let sidearm = self.catalog.sidearm;
        let mut spawns = [0usize; 2];
        for player in 0..self.players.len() {
            let side = self.side_of_player(player).unwrap_or(Side::Defend);
            let p = &mut self.players[player];
            let team = p.team.index();
            p.alive = true;
            events.push(Event::Respawn {
                player,
                side,
                spawn: spawns[team],
                health,
            });
            spawns[team] += 1;
            if p.sidearm.is_none() {
                p.sidearm = Some(sidearm);
                events.push(Event::GiveWeapon {
                    player,
                    weapon: sidearm,
                });
            }
            let signature = &mut p.abilities[2];
            if signature.charges == 0 && signature.max > 0 {
                signature.charges = 1;
                events.push(Event::Charges {
                    player,
                    slot: AbilitySlot::E,
                    charges: 1,
                });
            }
            if signature.charges >= signature.max {
                signature.cooldown_ticks = 0;
            }
            self.ledgers[player].close_buying();
        }
        self.give_device(events);
        let swap = self.catalog.swap_sides_after_round;
        let first = self.round == 1 || self.round == swap + 1;
        let timings = self.catalog.timings;
        let ticks = if first {
            timings.buy_first_round_of_half
        } else {
            timings.buy
        };
        self.enter_phase(Phase::Buy, ticks, events);
    }

    fn give_device(&mut self, events: &mut Vec<Event>) {
        let attackers: Vec<PlayerId> = (0..self.players.len())
            .filter(|&p| self.side_of_player(p) == Some(Side::Attack))
            .collect();
        let roll = self.next_random();
        let count = attackers.len() as u64;
        if count == 0 {
            return;
        }
        let player = attackers[(roll % count) as usize];
        self.device = Device::Carried {
            player,
            planting: None,
        };
        events.push(Event::DeviceGiven { player });
    }

    pub(super) fn enter_phase(&mut self, phase: Phase, ticks: u32, events: &mut Vec<Event>) {
        self.phase = phase;
        self.phase_left = ticks;
        events.push(Event::Phase {
            phase,
            ticks,
            round: self.round,
        });
    }

    pub(super) fn advance_buy(&mut self, events: &mut Vec<Event>) {
        self.phase_left = self.phase_left.saturating_sub(1);
        if self.phase_left == 0 {
            for ledger in &mut self.ledgers {
                ledger.close_buying();
            }
            let ticks = self.catalog.timings.round;
            self.enter_phase(Phase::Live, ticks, events);
        }
    }

    pub(super) fn advance_live(&mut self, input: &TickInput, events: &mut Vec<Event>) {
        self.pick_up_device(input, events);
        if self.advance_plant(input, events) {
            return;
        }
        if self.alive_on(Side::Attack) == 0 {
            self.end_round(Side::Defend, RoundEndReason::Elimination, events);
            return;
        }
        if self.alive_on(Side::Defend) == 0 {
            self.end_round(Side::Attack, RoundEndReason::Elimination, events);
            return;
        }
        self.phase_left = self.phase_left.saturating_sub(1);
        if self.phase_left == 0 {
            self.end_round(Side::Defend, RoundEndReason::Time, events);
        }
    }

    pub(super) fn advance_planted(&mut self, input: &TickInput, events: &mut Vec<Event>) {
        if self.advance_defuse(input, events) {
            self.end_round(Side::Defend, RoundEndReason::Defused, events);
            return;
        }
        if self.alive_on(Side::Defend) == 0 {
            self.end_round(Side::Attack, RoundEndReason::Elimination, events);
            return;
        }
        if self.advance_fuse(events) {
            self.end_round(Side::Attack, RoundEndReason::Exploded, events);
        }
    }

    pub(super) fn advance_round_end(&mut self, events: &mut Vec<Event>) {
        self.phase_left = self.phase_left.saturating_sub(1);
        if self.phase_left > 0 {
            return;
        }
        let to_win = self.catalog.rounds_to_win;
        if let Some(winner) = Team::BOTH
            .into_iter()
            .find(|t| self.score[t.index()] >= to_win)
        {
            events.push(Event::MatchEnd {
                winner,
                score: self.score,
            });
            self.enter_phase(Phase::Over, 0, events);
            return;
        }
        if self.round == self.catalog.swap_sides_after_round {
            self.sides = [self.sides[0].other(), self.sides[1].other()];
            events.push(Event::SideSwap { sides: self.sides });
            self.reset_for_half(CreditReason::SideSwap, events);
        }
        self.round += 1;
        self.begin_round(events);
    }

    fn end_round(&mut self, winning_side: Side, reason: RoundEndReason, events: &mut Vec<Event>) {
        if let Device::Carried {
            player,
            planting: Some(planting),
        } = self.device
        {
            events.push(Event::PlantCancelled {
                player,
                site: planting.site,
            });
            self.device = Device::Carried {
                player,
                planting: None,
            };
        }
        let winner = self.team_on(winning_side);
        let loser = winner.other();
        self.score[winner.index()] += 1;
        events.push(Event::RoundEnd {
            round: self.round,
            winner,
            reason,
            score: self.score,
        });
        let win = i64::from(self.catalog.economy.round_win);
        let loss = i64::from(self.catalog.loss_reward(self.loss_streak[loser.index()]));
        self.loss_streak[winner.index()] = 0;
        self.loss_streak[loser.index()] += 1;
        for player in 0..self.players.len() {
            if self.players[player].team == winner {
                self.credit(player, win, CreditReason::RoundWin, events);
            } else {
                self.credit(player, loss, CreditReason::RoundLoss, events);
            }
        }
        for player in 0..self.players.len() {
            self.ultimate_point(player, events);
        }
        let ticks = self.catalog.timings.round_end;
        self.enter_phase(Phase::RoundEnd, ticks, events);
    }

    pub(super) fn ultimate_point(&mut self, player: PlayerId, events: &mut Vec<Event>) {
        let Some(p) = self.players.get_mut(player) else {
            return;
        };
        let ultimate = &mut p.ultimate;
        if ultimate.points >= ultimate.cost {
            return;
        }
        ultimate.points += 1;
        events.push(Event::UltimatePoint {
            player,
            points: ultimate.points,
            cost: ultimate.cost,
        });
        if ultimate.points == ultimate.cost {
            events.push(Event::UltimateReady { player });
        }
    }

    pub(super) fn apply_armor_facts(&mut self, input: &TickInput, events: &mut Vec<Event>) {
        for (player, facts) in input.players.iter().enumerate() {
            let Some(left) = facts.armor_left else {
                continue;
            };
            let Some(p) = self.players.get_mut(player) else {
                continue;
            };
            let Some(armor) = p.armor else {
                continue;
            };
            if !p.alive || left >= armor.points {
                continue;
            }
            p.armor = (left > 0).then_some(super::Armor {
                tier: armor.tier,
                points: left,
            });
            events.push(Event::Armor {
                player,
                armor: p.armor,
            });
        }
    }

    pub(super) fn apply_death(
        &mut self,
        death: &Death,
        input: &TickInput,
        events: &mut Vec<Event>,
    ) {
        let victim = death.victim;
        let Some(v) = self.players.get_mut(victim) else {
            return;
        };
        if !v.alive {
            return;
        }
        v.alive = false;
        v.deaths += 1;
        let victim_team = v.team;
        for weapon in [v.primary.take(), v.sidearm.take()].into_iter().flatten() {
            events.push(Event::TakeWeapon {
                player: victim,
                weapon,
            });
        }
        if v.armor.take().is_some() {
            events.push(Event::Armor {
                player: victim,
                armor: None,
            });
        }
        self.drop_device_of(victim, input, events);
        self.ultimate_point(victim, events);
        let enemy =
            |p: PlayerId, players: &[Player]| players.get(p).is_some_and(|q| q.team != victim_team);
        if let Some(killer) = death.killer.filter(|&k| enemy(k, &self.players)) {
            self.players[killer].kills += 1;
            let kill = i64::from(self.catalog.economy.kill);
            self.credit(killer, kill, CreditReason::Kill, events);
            self.ultimate_point(killer, events);
            self.refresh_signature_on_kill(killer, events);
        }
        for (i, &assistant) in death.assistants.iter().enumerate() {
            let repeated = death.assistants[..i].contains(&assistant);
            if !repeated && Some(assistant) != death.killer && enemy(assistant, &self.players) {
                self.players[assistant].assists += 1;
            }
        }
        events.push(Event::KillFeed {
            killer: death.killer,
            victim,
            assistants: death.assistants.clone(),
            headshot: death.headshot,
            cause: death.cause,
        });
    }

    fn refresh_signature_on_kill(&mut self, player: PlayerId, events: &mut Vec<Event>) {
        let entry = &self.catalog.characters[self.players[player].character];
        let refresh_all = entry.ultimate.refresh_signature_on_kill;
        let refresh_every = entry.signature.refresh_on_kills;
        let p = &mut self.players[player];
        let active = p.ultimate.active_ticks > 0;
        let signature = &mut p.abilities[2];
        if signature.charges >= signature.max {
            return;
        }
        let ledger = &mut self.ledgers[player];
        if active && refresh_all {
            signature.charges = signature.max;
        } else if let Some(every) = refresh_every {
            ledger.kills_toward_refresh += 1;
            if ledger.kills_toward_refresh < every {
                return;
            }
            signature.charges += 1;
        } else {
            return;
        }
        ledger.kills_toward_refresh = 0;
        if signature.charges >= signature.max {
            signature.cooldown_ticks = 0;
        }
        events.push(Event::Charges {
            player,
            slot: AbilitySlot::E,
            charges: signature.charges,
        });
    }

    pub(super) fn apply_ability(&mut self, used: AbilityUse, events: &mut Vec<Event>) {
        let player = used.player;
        let refused = Event::AbilityRefused {
            player,
            slot: used.slot,
        };
        let Some(p) = self.players.get_mut(player).filter(|p| p.alive) else {
            events.push(refused);
            return;
        };
        let character = p.character;
        let Some(index) = used.slot.index() else {
            if p.ultimate.points < p.ultimate.cost {
                events.push(refused);
                return;
            }
            p.ultimate.points = 0;
            p.ultimate.active_ticks = self.catalog.characters[character]
                .ultimate
                .active_ticks
                .unwrap_or(0);
            events.push(Event::AbilityUsed {
                player,
                slot: used.slot,
            });
            return;
        };
        let ability = &mut p.abilities[index];
        if ability.charges == 0 {
            events.push(refused);
            return;
        }
        ability.charges -= 1;
        let ledger = &mut self.ledgers[player];
        if index < 2 {
            ledger.charges_bought[index] = ledger.charges_bought[index].min(ability.charges);
        } else if ability.cooldown_ticks == 0 {
            ability.cooldown_ticks = self.catalog.characters[character]
                .signature
                .cooldown_ticks
                .unwrap_or(0);
        }
        events.push(Event::AbilityUsed {
            player,
            slot: used.slot,
        });
        events.push(Event::Charges {
            player,
            slot: used.slot,
            charges: ability.charges,
        });
    }

    pub(super) fn advance_cooldowns(&mut self, events: &mut Vec<Event>) {
        for player in 0..self.players.len() {
            let cooldown = self.catalog.characters[self.players[player].character]
                .signature
                .cooldown_ticks
                .unwrap_or(0);
            let p = &mut self.players[player];
            p.ultimate.active_ticks = p.ultimate.active_ticks.saturating_sub(1);
            let signature = &mut p.abilities[2];
            if signature.cooldown_ticks == 0 {
                continue;
            }
            signature.cooldown_ticks -= 1;
            if signature.cooldown_ticks > 0 {
                continue;
            }
            signature.charges = (signature.charges + 1).min(signature.max);
            if signature.charges < signature.max {
                signature.cooldown_ticks = cooldown;
            }
            events.push(Event::Charges {
                player,
                slot: AbilitySlot::E,
                charges: signature.charges,
            });
        }
    }
}
