import package::place::placed::{Pose, Record};

const DEAD: u32 = 0xffffffffu;
const HIDDEN: u32 = 0x80000000u;

fn placed_shown(record: Record) -> bool {
    return record.group != DEAD && (record.terrain & HIDDEN) == 0u;
}
