const ORDER_LEVELS: u32 = 16u;

fn hash(value: u32) -> u32 {
    var h = value * 0x9e3779b1u;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}

fn order_turn(block: vec2<u32>, level: u32, slot: u32) -> u32 {
    return hash((block.x * 0x8da6b343u) ^ (block.y * 0xd8163841u)
        ^ ((level + ORDER_LEVELS * slot) * 0x2c1b3c6du)) >> 30u;
}

fn ordered_cell(start: vec2<u32>, top: u32, count: u32, digits: u32, slot: u32) -> vec2<u32> {
    var block = start;
    var rest = digits;
    for (var taken = 0u; taken < count; taken++) {
        let child = (rest & 3u) ^ order_turn(block, top - taken, slot);
        rest = rest >> 2u;
        block = block * 2u + vec2<u32>(child & 1u, child >> 1u);
    }
    return block;
}

fn quad_order(quad: vec2<u32>, depth: u32, slot: u32, limit: u32) -> u32 {
    var order = 0u;
    for (var level = 1u; level <= depth && order < limit; level++) {
        let child = ((quad.x >> (level - 1u)) & 1u) | (((quad.y >> (level - 1u)) & 1u) << 1u);
        order |= (child ^ order_turn(quad >> vec2<u32>(level), level, slot))
            << (2u * (depth - level));
    }
    return order;
}

fn order_tail(cell: vec2<u32>, depth: u32, slot: u32) -> f32 {
    var block = cell;
    var tail = 0.0;
    var weight = 0.25;
    for (var level = depth + 1u; level <= ORDER_LEVELS; level++) {
        let child = (block.x & 1u) | ((block.y & 1u) << 1u);
        block = block >> vec2<u32>(1u);
        tail += f32(child ^ order_turn(block, level, slot)) * weight;
        weight *= 0.25;
    }
    return tail;
}
