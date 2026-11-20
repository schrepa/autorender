mod gpu;

pub use gpu::Gpu;

pub const NONE: u32 = u32::MAX;
const TRIANGLE_WORDS: usize = 10;

#[derive(Clone, Copy)]
struct Bounds {
    low: [f32; 3],
    high: [f32; 3],
}

impl Bounds {
    fn empty() -> Self {
        Self {
            low: [f32::INFINITY; 3],
            high: [f32::NEG_INFINITY; 3],
        }
    }

    fn grow(&mut self, point: [f32; 3]) {
        for ((low, high), value) in self.low.iter_mut().zip(&mut self.high).zip(point) {
            if value < *low {
                *low = value;
            }
            if value > *high {
                *high = value;
            }
        }
    }

    fn join(&mut self, other: &Self) {
        self.grow(other.low);
        self.grow(other.high);
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[derive(Clone, Copy)]
struct Triangle {
    corner: [f32; 3],
    first: [f32; 3],
    second: [f32; 3],
    two_sided: bool,
}

struct Node {
    bounds: Bounds,
    start: u32,
    count: u32,
}

struct Item {
    bounds: Bounds,
    centre: [f32; 3],
    triangle: Triangle,
}

pub struct Occluders {
    nodes: Vec<u32>,
    triangles: Vec<u32>,
}

fn build_node(items: &mut [Item], first: usize, nodes: &mut Vec<Node>, at: usize) {
    let mut bounds = Bounds::empty();
    let mut centres = Bounds::empty();
    for item in items.iter() {
        bounds.join(&item.bounds);
        centres.grow(item.centre);
    }
    let count = u32::try_from(items.len()).expect("tree_items_fit_u32");
    let start = u32::try_from(first).expect("tree_items_fit_u32");
    if items.len() <= 4 {
        nodes[at] = Node {
            bounds,
            start,
            count,
        };
        return;
    }
    let size = sub(centres.high, centres.low);
    let axis = if size[0] >= size[1] && size[0] >= size[2] {
        0
    } else if size[1] >= size[2] {
        1
    } else {
        2
    };
    let middle = items.len() / 2;
    items.select_nth_unstable_by(middle, |a, b| a.centre[axis].total_cmp(&b.centre[axis]));
    let left = nodes.len();
    for _ in 0..2 {
        nodes.push(Node {
            bounds,
            start: 0,
            count: 0,
        });
    }
    nodes[at] = Node {
        bounds,
        start: u32::try_from(left).expect("tree_nodes_fit_u32"),
        count: 0,
    };
    let (low, high) = items.split_at_mut(middle);
    build_node(low, first, nodes, left);
    build_node(high, first + middle, nodes, left + 1);
}

impl Occluders {
    pub fn new(triangles: impl IntoIterator<Item = ([[f32; 3]; 3], bool)>, pad: f32) -> Self {
        let mut items: Vec<Item> = Vec::new();
        for ([a, b, c], two_sided) in triangles {
            let mut bounds = Bounds::empty();
            for point in [a, b, c] {
                bounds.grow(point);
            }
            for axis in 0..3 {
                bounds.low[axis] -= pad;
                bounds.high[axis] += pad;
            }
            items.push(Item {
                bounds,
                centre: [0, 1, 2].map(|axis| (a[axis] + b[axis] + c[axis]) / 3.0),
                triangle: Triangle {
                    corner: a,
                    first: sub(b, a),
                    second: sub(c, a),
                    two_sided,
                },
            });
        }
        let mut nodes = vec![Node {
            bounds: Bounds::empty(),
            start: 0,
            count: 0,
        }];
        if !items.is_empty() {
            build_node(&mut items, 0, &mut nodes, 0);
        }
        let nodes = nodes.iter().flat_map(|node| {
            (node.bounds.low.into_iter().chain(node.bounds.high))
                .map(f32::to_bits)
                .chain([node.start, node.count])
        });
        let triangles = items.iter().flat_map(|item| {
            let held = item.triangle;
            (held.corner.into_iter().chain(held.first).chain(held.second))
                .map(f32::to_bits)
                .chain([u32::from(held.two_sided)])
        });
        Self {
            nodes: nodes.collect(),
            triangles: triangles.collect(),
        }
    }

    pub fn count(&self) -> usize {
        self.triangles.len() / TRIANGLE_WORDS
    }
}
