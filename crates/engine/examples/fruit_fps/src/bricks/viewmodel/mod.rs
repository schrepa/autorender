engine::brick!(viewmodel);

mod place;

use engine::prelude::*;

use crate::play::puppet::ViewModel;

stats! {
    #[section("viewmodel")]
    pub struct ViewModelStats {
        pub weapon: Option<String>,
        pub clip: String,
        pub offset: [f32; 3],
        pub scale: f32,
        pub raise: f32,
        pub scope: f32,
        pub lower: f32,
        pub kick: f32,
        pub muzzle: Option<[f32; 3]>,
        pub refusal: Option<String>,
    }
}

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct FirstPersonWeapon;

pub struct ViewModelBrick;

impl Brick for ViewModelBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .resource::<ViewModel>()
            .resource::<place::Held>()
            .stats::<ViewModelStats>()
            .moves::<FirstPersonWeapon>(Pace::Tick)
            .systems(Schedule::FixedSimulation, place::place)
            .systems(Schedule::Presentation, place::veil);
    }
}
