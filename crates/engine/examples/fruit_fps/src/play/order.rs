#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Step {
    Input,
    Think,
    Move,
    Act,
    Rules,
    Present,
}

impl Step {
    pub const ALL: [Step; 6] = [
        Step::Input,
        Step::Think,
        Step::Move,
        Step::Act,
        Step::Rules,
        Step::Present,
    ];
}
