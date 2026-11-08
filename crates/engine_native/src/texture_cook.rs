use engine::bevy_ecs::world::World;
use engine_core::{
    Brick, Bricks, Cook, CookAsks, CookList, Error, Job, JobCx, JobKind, RecordExchange, Source,
    TEXTURE_COOK_NAME, TEXTURE_COOK_SOURCE, TextureFamily,
};

pub static TEXTURE_COOK: JobKind = JobKind::new(TEXTURE_COOK_NAME, &TEXTURE_COOK_SOURCE, start);

struct Cooking(Vec<u8>);

impl Job for Cooking {
    fn step(&mut self, _: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        engine_texture_cook::cook_job(&self.0).map(Some)
    }
}

fn start(input: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    Ok(Box::new(Cooking(input.to_vec())))
}

pub struct TextureCook;

impl Brick for TextureCook {
    const SOURCE: &'static Source = &TEXTURE_COOK_SOURCE;
    const JOBS: &'static [&'static JobKind] = &[&TEXTURE_COOK];

    fn register(bricks: &mut Bricks<'_>) {
        bricks.join::<CookList>(Cook::new(cook), None);
    }
}

fn cook(world: &World, asks: &mut CookAsks<'_>) {
    let Some(records) = world.get_resource::<RecordExchange>() else {
        return;
    };
    for set in records.texture_sets() {
        let keys: Vec<_> = TextureFamily::ALL
            .into_iter()
            .flat_map(|family| (0..set.textures.len()).map(move |index| (index, family)))
            .map(|(index, family)| (index, family, set.key(index, family)))
            .filter(|(_, _, key)| !asks.asked(key))
            .collect();
        if keys.is_empty() {
            continue;
        }
        let Ok(sources) = set.sources() else {
            continue;
        };
        for (index, family, key) in keys {
            asks.ask_as(
                key,
                &TEXTURE_COOK,
                engine_textures::cook_input(&sources[index], family),
            );
        }
    }
}
