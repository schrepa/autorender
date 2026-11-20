use bevy_ecs::world::World;

use crate::{Assets, Board, Readiness, Refusals};

#[derive(Clone, Copy)]
pub(crate) struct Reported {
    assets: Option<(u64, u32)>,
    board: u64,
    from_board: usize,
}

fn same_refusals(readiness: &Readiness, from_board: usize, refusals: Option<&Refusals>) -> bool {
    let Some(listed) = readiness.failed.get(from_board..) else {
        return false;
    };
    let held = refusals.into_iter().flat_map(Refusals::iter);
    listed
        .iter()
        .map(|(by, error)| (by.as_str(), error))
        .eq(held.map(|(_, refused)| (refused.part, &refused.reason)))
}

pub(crate) fn settle_readiness(world: &mut World, board: &Board, reported: &mut Option<Reported>) {
    let assets = world.get_resource::<Assets>().map(Assets::revision).zip(
        world
            .get_resource_change_ticks::<Assets>()
            .map(|ticks| ticks.changed.get()),
    );
    let refusals = world.get_resource::<Refusals>();
    if let (Some(known), Some(readiness)) = (*reported, world.get_resource::<Readiness>())
        && (known.assets, known.board) == (assets, board.changes())
        && same_refusals(readiness, known.from_board, refusals)
    {
        return;
    }
    let compiling = world
        .get_resource::<Readiness>()
        .map_or(0, |readiness| readiness.compiling);
    let mut readiness = Readiness {
        compiling,
        groups: world
            .get_resource::<Assets>()
            .map(Assets::groups)
            .unwrap_or_default(),
        ..Readiness::default()
    };
    let changes = board.changes();
    board.add_to(&mut readiness);
    let from_board = readiness.failed.len();
    for (_, refused) in refusals.into_iter().flat_map(Refusals::iter) {
        readiness.fail(refused.part, refused.reason.clone());
    }
    *reported = Some(Reported {
        assets,
        board: changes,
        from_board,
    });
    if world.get_resource::<Readiness>() != Some(&readiness) {
        world.insert_resource(readiness);
    }
}
