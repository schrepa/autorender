mod bots;
mod buying;
mod device;
mod flow;
mod numbers;

pub use bots::suggest_buy;
pub use numbers::{
    ArmorNumbers, BoughtAbilityNumbers, BoughtEntry, Catalog, CharacterEntry, CharacterNumbers,
    EconomyNumbers, MatchNumbers, PlayerNumbers, RulesError, RulesNumbers, SignatureEntry,
    SignatureNumbers, Timings, UltimateEntry, UltimateNumbers, WeaponEntry, WeaponNumbers,
    WeaponSlot, ticks,
};

pub type PlayerId = usize;
pub type SiteId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WeaponId(pub usize);

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Team {
    First,
    Second,
}

impl Team {
    pub const BOTH: [Team; 2] = [Team::First, Team::Second];

    pub fn index(self) -> usize {
        match self {
            Team::First => 0,
            Team::Second => 1,
        }
    }

    #[must_use]
    pub fn other(self) -> Team {
        match self {
            Team::First => Team::Second,
            Team::Second => Team::First,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Attack,
    Defend,
}

impl Side {
    #[must_use]
    pub fn other(self) -> Side {
        match self {
            Side::Attack => Side::Defend,
            Side::Defend => Side::Attack,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Control {
    Human,
    Bot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot {
    pub team: Team,
    pub character: String,
    pub control: Control,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Buy,
    Live,
    Planted,
    RoundEnd,
    Over,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RoundEndReason {
    Elimination,
    Exploded,
    Defused,
    Time,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AbilitySlot {
    C,
    Q,
    E,
    X,
}

impl AbilitySlot {
    fn index(self) -> Option<usize> {
        match self {
            AbilitySlot::C => Some(0),
            AbilitySlot::Q => Some(1),
            AbilitySlot::E => Some(2),
            AbilitySlot::X => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArmorTier {
    Light,
    Heavy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Item {
    Weapon(WeaponId),
    Armor(ArmorTier),
    Ability(AbilitySlot),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuyRequest {
    Buy(Item),
    Sell(Item),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuyRefusal {
    WrongPhase,
    Dead,
    OutsideBuyZone,
    UnknownPlayer,
    NotForSale,
    TooFewCredits,
    AlreadyOwned,
    AtMaximum,
    NothingToSell,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cause {
    Weapon(WeaponId),
    Ability(AbilitySlot),
    World,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CreditReason {
    Start,
    Kill,
    PlantBonus,
    RoundWin,
    RoundLoss,
    Bought,
    Refunded,
    SideSwap,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlayerFacts {
    pub position: [f32; 3],
    pub site: Option<SiteId>,
    pub in_buy_zone: bool,
    pub near_device: bool,
    pub plant: bool,
    pub defuse: bool,
    pub armor_left: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Death {
    pub victim: PlayerId,
    pub killer: Option<PlayerId>,
    pub assistants: Vec<PlayerId>,
    pub headshot: bool,
    pub cause: Cause,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AbilityUse {
    pub player: PlayerId,
    pub slot: AbilitySlot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Holding {
    pub player: PlayerId,
    pub primary: Option<WeaponId>,
    pub sidearm: Option<WeaponId>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TickInput {
    pub players: Vec<PlayerFacts>,
    pub buys: Vec<(PlayerId, BuyRequest)>,
    pub deaths: Vec<Death>,
    pub abilities: Vec<AbilityUse>,
    pub holdings: Vec<Holding>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Armor {
    pub tier: ArmorTier,
    pub points: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Ability {
    pub charges: u32,
    pub max: u32,
    pub cooldown_ticks: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Ultimate {
    pub points: u32,
    pub cost: u32,
    pub active_ticks: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Player {
    pub team: Team,
    pub character: usize,
    pub control: Control,
    pub alive: bool,
    pub credits: u32,
    pub primary: Option<WeaponId>,
    pub sidearm: Option<WeaponId>,
    pub armor: Option<Armor>,
    pub abilities: [Ability; 3],
    pub ultimate: Ultimate,
    pub kills: u32,
    pub deaths: u32,
    pub assists: u32,
}

impl Player {
    pub fn ability(&self, slot: AbilitySlot) -> Option<&Ability> {
        slot.index().and_then(|i| self.abilities.get(i))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Planting {
    pub site: SiteId,
    pub progress: u32,
    pub needed: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Defusing {
    pub player: Option<PlayerId>,
    pub progress: u32,
    pub checkpoint: u32,
    pub needed: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Device {
    Carried {
        player: PlayerId,
        planting: Option<Planting>,
    },
    Dropped {
        at: [f32; 3],
    },
    Planted {
        site: SiteId,
        at: [f32; 3],
        fuse_ticks_left: u32,
        defusing: Defusing,
    },
    Exploded {
        site: SiteId,
        at: [f32; 3],
    },
    Defused {
        site: SiteId,
        at: [f32; 3],
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Phase {
        phase: Phase,
        ticks: u32,
        round: u32,
    },
    Respawn {
        player: PlayerId,
        side: Side,
        spawn: usize,
        health: u32,
    },
    GiveWeapon {
        player: PlayerId,
        weapon: WeaponId,
    },
    TakeWeapon {
        player: PlayerId,
        weapon: WeaponId,
    },
    Armor {
        player: PlayerId,
        armor: Option<Armor>,
    },
    Charges {
        player: PlayerId,
        slot: AbilitySlot,
        charges: u32,
    },
    Credits {
        player: PlayerId,
        credits: u32,
        delta: i64,
        reason: CreditReason,
    },
    BuyRefused {
        player: PlayerId,
        request: BuyRequest,
        reason: BuyRefusal,
    },
    AbilityUsed {
        player: PlayerId,
        slot: AbilitySlot,
    },
    AbilityRefused {
        player: PlayerId,
        slot: AbilitySlot,
    },
    DeviceGiven {
        player: PlayerId,
    },
    DeviceDropped {
        player: PlayerId,
        at: [f32; 3],
    },
    DevicePickedUp {
        player: PlayerId,
    },
    PlantStarted {
        player: PlayerId,
        site: SiteId,
    },
    PlantCancelled {
        player: PlayerId,
        site: SiteId,
    },
    PlantCompleted {
        player: PlayerId,
        site: SiteId,
        at: [f32; 3],
        fuse_ticks: u32,
    },
    DefuseStarted {
        player: PlayerId,
        progress: u32,
    },
    DefuseHalf {
        player: PlayerId,
    },
    DefuseCancelled {
        player: PlayerId,
        kept: u32,
    },
    DefuseCompleted {
        player: PlayerId,
    },
    DeviceExploded {
        site: SiteId,
        at: [f32; 3],
    },
    KillFeed {
        killer: Option<PlayerId>,
        victim: PlayerId,
        assistants: Vec<PlayerId>,
        headshot: bool,
        cause: Cause,
    },
    UltimatePoint {
        player: PlayerId,
        points: u32,
        cost: u32,
    },
    UltimateReady {
        player: PlayerId,
    },
    RoundEnd {
        round: u32,
        winner: Team,
        reason: RoundEndReason,
        score: [u32; 2],
    },
    SideSwap {
        sides: [Side; 2],
    },
    MatchEnd {
        winner: Team,
        score: [u32; 2],
    },
}

#[derive(Clone, Copy, Debug)]
pub struct View<'a> {
    pub phase: Phase,
    pub ticks_left: u32,
    pub tick: u64,
    pub round: u32,
    pub first_round_of_half: bool,
    pub last_round_of_half: bool,
    pub score: [u32; 2],
    pub sides: [Side; 2],
    pub loss_streak: [u32; 2],
    pub players: &'a [Player],
    pub device: &'a Device,
    pub catalog: &'a Catalog,
}

impl View<'_> {
    pub fn side_of(&self, team: Team) -> Side {
        self.sides[team.index()]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Ledger {
    primary_paid: Option<u32>,
    replaced_primary: Option<WeaponId>,
    armor_paid: Option<u32>,
    replaced_armor: Option<Armor>,
    charges_bought: [u32; 2],
    kills_toward_refresh: u32,
}

impl Ledger {
    fn close_buying(&mut self) {
        *self = Ledger {
            kills_toward_refresh: self.kills_toward_refresh,
            ..Ledger::default()
        };
    }
}

#[derive(Clone, Debug)]
pub struct Match {
    catalog: Catalog,
    players: Vec<Player>,
    ledgers: Vec<Ledger>,
    device: Device,
    phase: Phase,
    phase_left: u32,
    round: u32,
    score: [u32; 2],
    sides: [Side; 2],
    loss_streak: [u32; 2],
    tick: u64,
    random: u64,
    pending: Vec<Event>,
}

impl Match {
    pub fn tick(&mut self, input: &TickInput) -> Vec<Event> {
        let mut events = std::mem::take(&mut self.pending);
        if self.phase == Phase::Over {
            return events;
        }
        self.tick += 1;
        self.apply_armor_facts(input, &mut events);
        for holding in &input.holdings {
            self.apply_holding(*holding);
        }
        for death in &input.deaths {
            self.apply_death(death, input, &mut events);
        }
        for (player, request) in &input.buys {
            self.apply_buy(*player, *request, input, &mut events);
        }
        for used in &input.abilities {
            self.apply_ability(*used, &mut events);
        }
        self.advance_cooldowns(&mut events);
        match self.phase {
            Phase::Buy => self.advance_buy(&mut events),
            Phase::Live => self.advance_live(input, &mut events),
            Phase::Planted => self.advance_planted(input, &mut events),
            Phase::RoundEnd => self.advance_round_end(&mut events),
            Phase::Over => {}
        }
        events
    }

    fn apply_holding(&mut self, holding: Holding) {
        let Some(p) = self.players.get_mut(holding.player).filter(|p| p.alive) else {
            return;
        };
        if p.primary != holding.primary {
            let ledger = &mut self.ledgers[holding.player];
            ledger.primary_paid = None;
            ledger.replaced_primary = None;
        }
        p.primary = holding.primary;
        p.sidearm = holding.sidearm;
    }

    pub fn view(&self) -> View<'_> {
        let swap = self.catalog.swap_sides_after_round;
        View {
            phase: self.phase,
            ticks_left: self.phase_left,
            tick: self.tick,
            round: self.round,
            first_round_of_half: self.round == 1 || self.round == swap + 1,
            last_round_of_half: self.round == swap,
            score: self.score,
            sides: self.sides,
            loss_streak: self.loss_streak,
            players: &self.players,
            device: &self.device,
            catalog: &self.catalog,
        }
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    pub fn ticks(&self) -> u64 {
        self.tick
    }

    pub fn side_of(&self, team: Team) -> Side {
        self.sides[team.index()]
    }

    fn side_of_player(&self, player: PlayerId) -> Option<Side> {
        self.players.get(player).map(|p| self.side_of(p.team))
    }

    fn alive_on(&self, side: Side) -> usize {
        self.players
            .iter()
            .filter(|p| p.alive && self.side_of(p.team) == side)
            .count()
    }

    fn team_on(&self, side: Side) -> Team {
        if self.sides[0] == side {
            Team::First
        } else {
            Team::Second
        }
    }

    fn credit(
        &mut self,
        player: PlayerId,
        delta: i64,
        reason: CreditReason,
        events: &mut Vec<Event>,
    ) {
        let cap = i64::from(self.catalog.economy.cap);
        let Some(p) = self.players.get_mut(player) else {
            return;
        };
        let old = i64::from(p.credits);
        let new = (old + delta).clamp(0, cap.max(old));
        p.credits = u32::try_from(new).unwrap_or(p.credits);
        events.push(Event::Credits {
            player,
            credits: p.credits,
            delta: new - old,
            reason,
        });
    }

    fn set_credits(
        &mut self,
        player: PlayerId,
        credits: u32,
        reason: CreditReason,
        events: &mut Vec<Event>,
    ) {
        let Some(p) = self.players.get_mut(player) else {
            return;
        };
        let old = i64::from(p.credits);
        p.credits = credits;
        events.push(Event::Credits {
            player,
            credits,
            delta: i64::from(credits) - old,
            reason,
        });
    }
}
