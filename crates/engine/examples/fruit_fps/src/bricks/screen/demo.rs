use crate::play::screen::{
    AbilityFacts, ActionFacts, BoardRow, BuyFacts, Card, CastFacts, FeedLine, HudFacts, MapItem,
    Mate, MenuFacts, MinimapFacts, PickupFacts, PlaceFacts, PlateFacts, PlayerFacts, ResultFacts,
    RoundFacts, SpectateFacts, TeamFacts, Tone, WeaponFacts, WeaponStats,
};
use crate::rules::{AbilitySlot, ArmorTier, BuyRequest, Item, Phase, WeaponId};

use super::Words;
use super::live::credits_into;

const FRUIT: [&str; 5] = ["banana", "strawberry", "orange", "watermelon", "eggplant"];
const LEFT: [&str; 5] = [
    "slipnslide",
    "berrybold",
    "Pulp_Fiction",
    "melonchoNY",
    "nightshade_",
];
const RIGHT: [&str; 5] = [
    "Peelgood",
    "jam_session",
    "ZestInPeace",
    "rindsight",
    "auberGENE",
];

fn plates(alive: [bool; 5], health: [f32; 5], ready: [bool; 5]) -> Vec<PlateFacts> {
    (0..5)
        .map(|at| PlateFacts {
            icon: format!("char_{}", FRUIT[at]),
            alive: alive[at],
            health: health[at],
            ult_ready: ready[at],
        })
        .collect()
}

fn line(
    killer: (&str, Tone),
    weapon: &str,
    headshot: bool,
    victim: (&str, Tone),
    mine: bool,
) -> FeedLine {
    FeedLine {
        killer: killer.0.to_owned(),
        killer_color: killer.1,
        weapon: if weapon.starts_with("weapon_") {
            weapon.to_owned()
        } else {
            String::new()
        },
        headshot,
        victim: victim.0.to_owned(),
        victim_color: victim.1,
        mine,
        ability: if weapon.starts_with("ability_") {
            weapon.to_owned()
        } else {
            String::new()
        },
        world: weapon == "world_fall",
    }
}

fn ability(icon: &str, max: u32, charges: u32) -> AbilityFacts {
    AbilityFacts {
        icon: icon.to_owned(),
        max,
        charges,
        empty: charges == 0,
        ..AbilityFacts::default()
    }
}

fn card(
    kind: &str,
    name: &str,
    icon: &str,
    price: &str,
    (owned, afford): (bool, bool),
    item: Item,
) -> Card {
    Card {
        kind: kind.to_owned(),
        name: name.to_owned(),
        icon: icon.to_owned(),
        price: price.to_owned(),
        owned,
        afford,
        buy: Some(BuyRequest::Buy(item)),
        sell: Some(BuyRequest::Sell(item)),
        ..Card::default()
    }
}

fn buy(words: &Words) -> BuyFacts {
    let weapons = [
        ("sidearm", "Pip", words.free.as_str(), true, true),
        ("smg", "Zest", "1,600", false, true),
        ("rifle", "Rind", "2,900", false, true),
        ("sniper", "Stem", "4,700", false, false),
    ];
    let stats = [
        (0.27, 0.6, 0.55),
        (0.18, 1.0, 0.4),
        (0.27, 0.73, 0.7),
        (1.0, 0.07, 1.0),
    ];
    let mut abilities = vec![
        card(
            &words.per_charge,
            "Peel",
            "ability_peel",
            "200",
            (false, true),
            Item::Ability(AbilitySlot::C),
        ),
        card(
            &words.per_charge,
            "Ripe Flash",
            "ability_ripe_flash",
            "250",
            (false, true),
            Item::Ability(AbilitySlot::Q),
        ),
    ];
    for (shown, charges) in abilities.iter_mut().zip([1, 0]) {
        shown.max = 2;
        shown.charges = charges;
    }
    let armor = [
        (
            ArmorTier::Light,
            &words.light,
            "armor_light",
            "400",
            25,
            false,
        ),
        (
            ArmorTier::Heavy,
            &words.heavy,
            "armor_heavy",
            "1,000",
            50,
            true,
        ),
    ];
    BuyFacts {
        time: 14.0,
        time_low: false,
        weapons: weapons
            .iter()
            .enumerate()
            .zip(stats)
            .map(
                |((at, (kind, name, price, owned, afford)), (damage, rate, range))| {
                    let icon = format!("weapon_{}", name.to_lowercase());
                    Card {
                        stats: WeaponStats {
                            damage,
                            rate,
                            range,
                        },
                        ..card(
                            kind,
                            name,
                            &icon,
                            price,
                            (*owned, *afford),
                            Item::Weapon(WeaponId(at)),
                        )
                    }
                },
            )
            .collect(),
        armor: armor
            .into_iter()
            .map(|(tier, name, icon, price, points, owned)| Card {
                points,
                ..card(
                    &words.armor_kind,
                    name,
                    icon,
                    price,
                    (owned, true),
                    Item::Armor(tier),
                )
            })
            .collect(),
        abilities,
        reward_win: 3000,
        reward_loss: 1900,
        team: LEFT[1..]
            .iter()
            .zip(&FRUIT[1..])
            .zip([3150, 800, 4200, 2900])
            .map(|((name, fruit), credits)| Mate {
                name: (*name).to_owned(),
                icon: format!("char_{fruit}"),
                credits,
                can_rifle: credits >= 2900,
            })
            .collect(),
    }
}

fn minimap() -> MinimapFacts {
    let item = |pos: [f32; 2], yaw_degrees: f32, color: Option<Tone>| MapItem {
        pos,
        yaw_degrees,
        color,
    };
    let attack = Some(Tone::Attack);
    MinimapFacts {
        me: vec![item([-26.5, 9.0], 25.0, None)],
        players: vec![
            item([-20.0, 16.5], 300.0, attack),
            item([-31.0, -4.0], 0.0, attack),
            item([-12.0, 11.7], 270.0, attack),
            item([-23.0, 27.0], 180.0, Some(Tone::Defend)),
        ],
        device: vec![item([-24.0, 12.0], 0.0, None)],
    }
}
fn board(words: &Words) -> [Vec<BoardRow>; 2] {
    let rows = |names: [&str; 5], numbers: [(u32, u32, u32, u32, bool); 5], mine: bool| {
        names
            .iter()
            .zip(numbers)
            .enumerate()
            .map(|(at, (name, (k, d, a, ult, alive)))| BoardRow {
                name: (*name).to_owned(),
                icon: format!("char_{}", FRUIT[at]),
                k,
                d,
                a,
                adr: 60 + k * 8,
                loadout: {
                    let mut out = String::new();
                    credits_into(&mut out, 3900 + at as u32 * 100);
                    out
                },
                credits: {
                    let mut out = String::new();
                    if mine {
                        credits_into(&mut out, 900 + at as u32 * 700);
                    } else {
                        out.push_str(&words.hidden);
                    }
                    out
                },
                ult,
                ult_need: 7,
                alive,
                me: mine && at == 0,
            })
            .collect()
    };
    [
        rows(
            LEFT,
            [
                (14, 7, 3, 5, true),
                (9, 8, 6, 7, true),
                (6, 9, 4, 2, false),
                (11, 6, 2, 6, true),
                (4, 10, 8, 3, true),
            ],
            true,
        ),
        rows(
            RIGHT,
            [
                (12, 8, 2, 4, true),
                (7, 9, 9, 6, true),
                (8, 10, 1, 1, false),
                (10, 8, 3, 8, true),
                (3, 9, 5, 2, false),
            ],
            false,
        ),
    ]
}

pub(super) fn demo(words: &Words) -> HudFacts {
    let attack = Tone::Attack;
    let defend = Tone::Defend;
    HudFacts {
        round: RoundFacts {
            time: 72.0,
            number: 9,
            ..RoundFacts::default()
        },
        score: [5, 3],
        teams: [
            TeamFacts {
                color: attack,
                name: words.attackers.clone(),
            },
            TeamFacts {
                color: defend,
                name: words.defenders.clone(),
            },
        ],
        players: [
            plates(
                [true, true, false, true, true],
                [76.0, 100.0, 0.0, 41.0, 88.0],
                [false, true, false, false, false],
            ),
            plates(
                [true, true, false, true, false],
                [100.0; 5],
                [false, false, false, true, false],
            ),
        ],
        feed: vec![
            line(
                (LEFT[0], attack),
                "weapon_rind",
                true,
                (RIGHT[2], defend),
                true,
            ),
            line(
                (RIGHT[3], defend),
                "weapon_stem",
                false,
                (LEFT[2], attack),
                false,
            ),
            line(
                (LEFT[3], attack),
                "ability_seed_grenade",
                false,
                (RIGHT[4], defend),
                false,
            ),
        ],
        player: PlayerFacts {
            health: 76.0,
            health_low: false,
            armor: 50.0,
            armor_icon: "armor_heavy".to_owned(),
            credits: 2450,
            char_name: "Bananito".to_owned(),
            role: "entry".to_owned(),
        },
        abilities: [
            ability("ability_peel", 2, 2),
            ability("ability_ripe_flash", 2, 1),
            ability("ability_split", 1, 1),
            AbilityFacts {
                icon: "ability_bunch".to_owned(),
                points: 5,
                points_needed: 7,
                ..AbilityFacts::default()
            },
        ],
        weapon: WeaponFacts {
            name: "Rind".to_owned(),
            mag: Some(18),
            mag_size: Some(25),
            reserve: Some(75),
            low: false,
            reload: None,
        },
        spread_px: 0.0,
        action: ActionFacts {
            active: true,
            label: "Planting Juicer".to_owned(),
            left: 1.5,
            progress: 0.62,
            color: attack,
            checkpoint: false,
        },
        buy: buy(words),
        board: board(words),
        result: ResultFacts {
            title: words.round_won.clone(),
            reason: "Juicer detonated on site B".to_owned(),
            color: attack,
            mvp: "slipnslide   3 kills".to_owned(),
            mvp_icon: "char_banana".to_owned(),
            mvp_color: attack,
            mvp_portrait: "char_banana".to_owned(),
            mvp_name: LEFT[0].to_owned(),
            mvp_stats: "24 kills   12 deaths   5 assists   ADR 168".to_owned(),
            mvp_char: "Bananito  entry".to_owned(),
            next_seconds: 9.0,
        },
        menu: MenuFacts::default(),
        spectate: SpectateFacts::default(),
        place: PlaceFacts::default(),
        scope: 0,
        pickup: PickupFacts::default(),
        cast: CastFacts::default(),
        minimap: minimap(),
        hit: None,
        damage: Vec::new(),
        phase: Some(Phase::Buy),
        live: false,
    }
}
