use crate::app_host::{Advance, AppHost};
use crate::game::{Game, LoadedScene, Start};
use crate::host::{DeviceGatherer, FrameLoop, FramePhase, InputSource};
use engine_core::{
    Assets, BrickList, CursorMode, Devices, Engine, Error, ErrorCode, Expected, IoOrder, IoReply,
    KeyCode, MouseButton, TouchPhase,
};
use engine_render::{Picture, RenderTarget};
use engine_scene::SceneDocument;

pub struct GameHost<G: Game> {
    pub(crate) app: AppHost,
    game: std::marker::PhantomData<fn() -> G>,
    pub(crate) scene: LoadedScene,
    variant: Option<&'static str>,
    cooking: Option<(SceneDocument, String)>,
    input: DeviceGatherer,
    frames: FrameLoop,
    away: bool,
    empty: bool,
    #[cfg(feature = "dev")]
    pub(crate) development: crate::dev_host::DevHost,
    #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
    source: Option<std::path::PathBuf>,
    #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
    watched: Vec<(std::path::PathBuf, Option<std::time::SystemTime>)>,
    #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
    checked: Option<std::time::Instant>,
}

impl<G: Game> GameHost<G> {
    pub async fn launch(
        target: RenderTarget,
        start: Start,
        bricks: BrickList,
        linked: Option<&'static str>,
    ) -> crate::Result<Self> {
        engine_panic::install();
        let pick = start.variant;
        #[cfg_attr(
            not(feature = "dev"),
            expect(unused_mut, reason = "dev installs names")
        )]
        let mut app = AppHost::launch(target, start, bricks, linked, G::default()).await?;
        #[cfg(feature = "dev")]
        crate::dev_host::install(app.engine.app.world_mut());
        let variant = pick.map(|pick| pick(app.renderer.features()));
        Ok(Self {
            app,
            game: std::marker::PhantomData,
            scene: LoadedScene::default(),
            variant,
            cooking: None,
            input: DeviceGatherer::default(),
            frames: FrameLoop::default(),
            away: false,
            empty: false,
            #[cfg(feature = "dev")]
            development: crate::dev_host::DevHost::default(),
            #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
            source: None,
            #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
            watched: Vec::new(),
            #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
            checked: None,
        })
    }

    pub fn engine(&self) -> &Engine {
        self.app.engine()
    }

    pub fn engine_mut(&mut self) -> &mut Engine {
        self.app.engine_mut()
    }

    pub(crate) fn framed(
        &mut self,
        advance: Advance,
        size: (u32, u32),
        picture: Picture,
    ) -> crate::Result<bool> {
        if self.scene.land(self.app.engine_mut().app.world_mut())? {
            self.app.replay_settings()?;
            self.suspend();
        }
        let input = self.frame_input(self.input_source());
        self.app.accept(input);
        self.app.frame(advance, size, picture)
    }

    pub(crate) fn play(&mut self, timestamp: f64, size: (u32, u32)) -> crate::Result<FramePhase> {
        #[cfg(feature = "dev")]
        return self.dev_play(timestamp, size);
        #[cfg(not(feature = "dev"))]
        {
            self.framed(Advance::Clock(timestamp), size, Picture::Shown)?;
            Ok(FramePhase::Playing)
        }
    }

    pub fn input_source(&self) -> InputSource {
        #[cfg(feature = "dev")]
        if self.development.stepping() {
            return InputSource::Agent;
        }
        InputSource::Player
    }

    fn frame_input(&mut self, source: InputSource) -> Devices {
        match source {
            InputSource::Agent => {
                self.input.drop_edges();
                self.agent_input()
            }
            InputSource::Player => self.input.take(),
        }
    }

    #[cfg(feature = "dev")]
    fn agent_input(&mut self) -> Devices {
        self.development.take_input()
    }

    #[cfg(not(feature = "dev"))]
    fn agent_input(&mut self) -> Devices {
        Devices::default()
    }

    pub fn focused(&mut self, focused: bool) {
        if !focused {
            self.input.clear();
        }
        self.app.report_host(|host| host.focused = focused);
    }

    pub fn hidden(&mut self, hidden: bool) {
        self.away = hidden;
        self.report_hidden();
    }

    pub(crate) fn sized(&mut self, empty: bool) {
        self.empty = empty;
        self.report_hidden();
    }

    fn report_hidden(&mut self) {
        let hidden = self.away || self.empty;
        if hidden {
            self.input.clear();
        }
        self.app.report_host(|host| host.hidden = hidden);
    }

    pub fn refresh_interval(&self) -> f64 {
        self.app.refresh_interval()
    }

    pub fn animation_frame(
        &mut self,
        timestamp: f64,
        width: u32,
        height: u32,
    ) -> crate::Result<FramePhase> {
        let mut frames = std::mem::take(&mut self.frames);
        let phase = frames.frame(self, timestamp, (width, height));
        self.frames = frames;
        phase
    }

    pub(crate) fn show(&mut self, showing: bool) {
        self.app.report_host(|host| host.showing = showing);
    }

    pub(crate) fn set_preparation_ms(&mut self, milliseconds: f64) {
        self.app.set_preparation_ms(milliseconds);
    }

    pub fn key(&mut self, code: &str, down: bool) {
        if let Some(key) = KeyCode::named(code) {
            self.input.key(key, down);
            return;
        }
        let world = self.app.engine_mut().app.world_mut();
        let notice = Error::new(ErrorCode::UnknownVariant, "$.key").got(code);
        if let Some(mut registry) = world.get_resource_mut::<engine_registry::Registry>()
            && !registry.notices().contains(&notice)
        {
            crate::app_host::log_notice(&notice);
            registry.report_notice(notice);
        }
    }

    pub fn mouse_button(&mut self, button: MouseButton, down: bool) {
        self.input.mouse_button(button, down);
    }

    pub fn motion(&mut self, dx: f32, dy: f32) {
        self.input.motion(dx, dy);
    }

    pub fn pointer(&mut self, x: f32, y: f32) {
        self.input.pointer(x, y);
    }

    pub fn pointer_left(&mut self) {
        self.input.pointer_left();
    }

    pub fn wheel(&mut self, right_notches: f32, up_notches: f32) {
        self.input.wheel(right_notches, up_notches);
    }

    pub fn pad(&mut self, index: u32, buttons: u32, axes: [f32; 6]) {
        self.input.pad(index, buttons, axes);
    }

    pub fn touch(&mut self, id: u32, x: f32, y: f32, phase: TouchPhase) {
        self.input.touch(id, x, y, phase);
    }

    pub fn touch_motion(&mut self, id: u32, dx: f32, dy: f32) {
        self.input.touch_motion(id, dx, dy);
    }

    pub fn cursor_mode(&self) -> CursorMode {
        *self.app.engine().app.world().resource::<CursorMode>()
    }

    pub fn take_sound(&mut self) -> Vec<engine_core::SoundOrder> {
        let world = self.app.engine_mut().app.world_mut();
        world
            .try_resource_scope(
                |world, mut sounds: bevy_ecs::world::Mut<engine_core::Sounds>| {
                    world
                        .get_resource_mut::<engine_core::Files>()
                        .map(|mut files| sounds.take_orders(&mut files))
                },
            )
            .flatten()
            .unwrap_or_default()
    }

    pub fn hear_sound(&mut self, reply: engine_core::SoundReply) {
        let world = self.app.engine_mut().app.world_mut();
        if let Some(mut sounds) = world.get_resource_mut::<engine_core::Sounds>() {
            sounds.hear(reply);
        }
    }

    pub fn take_io(&mut self) -> Vec<IoOrder> {
        self.app.engine_mut().take_io()
    }

    pub fn land_io(&mut self, reply: IoReply) {
        self.app.engine_mut().land_io(reply);
    }

    pub fn suspend(&mut self) {
        self.app.suspend();
        self.input.drop_edges();
        #[cfg(feature = "dev")]
        self.development.drop_edges();
    }

    pub fn reset_camera(&mut self) {
        G::reset_camera(self.app.engine_mut(), &self.scene);
        self.suspend();
    }

    pub fn offer_close(&mut self) {
        let world = self.app.engine_mut().app.world_mut();
        if let Some(mut closing) = world.get_resource_mut::<engine_core::Closing>() {
            closing.offer();
        }
    }

    pub fn close_asked(&self) -> bool {
        let world = self.app.engine().app.world();
        world
            .get_resource::<engine_core::Closing>()
            .is_some_and(engine_core::Closing::asked)
            && !world
                .get_resource::<engine_core::Kept>()
                .is_some_and(engine_core::Kept::in_flight)
    }

    pub fn keep_in(&mut self, place: &str) {
        let world = self.app.engine_mut().app.world_mut();
        if let Some(mut kept) = world.get_resource_mut::<engine_core::Kept>() {
            kept.keep_in(place);
        }
    }

    pub fn set_file_root(&mut self, root: &str) {
        let world = self.app.engine_mut().app.world_mut();
        if let Some(mut files) = world.get_resource_mut::<engine_core::Files>() {
            files.set_root(root);
        }
    }

    pub fn open_scene(&mut self, source: &str) -> crate::Result<()> {
        self.arrive(SceneDocument::from_json(source)?)
    }

    pub(crate) fn arrive(&mut self, document: SceneDocument) -> crate::Result<()> {
        match document.cooked().map(str::to_owned) {
            Some(uri) => {
                self.cooking = Some((document, uri));
                Ok(())
            }
            None => {
                self.cooking = None;
                self.instantiate(&document)
            }
        }
    }

    fn instantiate(&mut self, document: &SceneDocument) -> crate::Result<()> {
        let world = self.app.engine_mut().app.world_mut();
        let instance = document.instantiate(world, self.variant)?;
        self.scene.replace(world, instance);
        Ok(())
    }

    pub fn files(&self) -> Vec<(String, String)> {
        let assets = self.app.engine().app.world().resource::<Assets>();
        if let Some((_, cooked)) = &self.cooking {
            let group = assets.groups().into_iter().next().map(|group| group.name);
            return vec![(cooked.clone(), group.unwrap_or_default())];
        }
        assets
            .files()
            .into_iter()
            .map(|(uri, group)| (uri.to_owned(), group.to_owned()))
            .collect()
    }

    pub fn file(&mut self, uri: &str, read: crate::Result<Vec<u8>>) -> crate::Result<()> {
        let Some((mut document, _)) = self.cooking.take_if(|(_, cooked)| cooked == uri) else {
            let world = self.app.engine_mut().app.world_mut();
            let mut assets = world.resource_mut::<Assets>();
            assets.file(uri, read.map(Into::into))?;
            if let Some(refused) = assets
                .ids()
                .filter(|&id| assets.uri(id) == Some(uri))
                .find_map(|id| assets.refusal(id))
            {
                crate::app_host::log_notice(refused);
            }
            return Ok(());
        };
        let bytes = read?;
        let text = std::str::from_utf8(&bytes).map_err(|_| {
            Error::new(ErrorCode::InvalidUtf8, "$.cooked")
                .expected(Expected::Utf8)
                .got(uri)
        })?;
        document.append_cooked(text)?;
        self.instantiate(&document)
    }

    pub fn set_settings(&mut self, entries: &[(String, serde_json::Value)]) -> crate::Result<()> {
        self.app.set_settings(entries)
    }

    pub fn set_pixel_ratio(&mut self, ratio: f64) {
        self.app.set_pixel_ratio(ratio);
    }

    pub fn set_refresh_interval(&mut self, milliseconds: Option<f64>) {
        self.app.set_refresh_interval(milliseconds);
    }

    pub fn adapter(&self) -> &crate::AdapterInfo {
        self.app.renderer.adapter()
    }

    pub fn time_frame(&self) {
        self.app.renderer.time_frame();
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_for_gpu(&self) -> crate::Result<()> {
        self.app.renderer.wait_for_gpu()
    }

    pub fn progress(&self) -> engine_core::Readiness {
        self.app.progress()
    }

    pub fn frame_times(&self) -> &crate::frame_stats::FrameStats {
        self.app.frame_times()
    }
}

#[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
impl<G: Game> GameHost<G> {
    pub fn watch_source(&mut self, path: &std::path::Path) {
        self.source = Some(path.to_owned());
        self.watch();
    }

    pub fn source(&self) -> Option<&std::path::Path> {
        self.source.as_deref()
    }

    pub fn follow_source(&mut self) -> bool {
        let now = std::time::Instant::now();
        if self
            .checked
            .is_some_and(|checked| now.duration_since(checked) < CHECK_EVERY)
        {
            return false;
        }
        self.checked = Some(now);
        self.app.reload_shaders();
        self.watched
            .iter()
            .any(|(file, stamp)| modified(file) != *stamp)
    }

    fn watch(&mut self) {
        let Some(path) = self.source.clone() else {
            self.watched.clear();
            return;
        };
        let cooked = std::fs::read_to_string(&path)
            .ok()
            .and_then(|source| SceneDocument::from_json(&source).ok())
            .and_then(|document| document.cooked().map(str::to_owned));
        let assets = self.app.engine().app.world().resource::<Assets>();
        let uris: std::collections::BTreeSet<String> = cooked
            .into_iter()
            .chain(assets.ids().filter_map(|id| assets.uri(id).map(str::to_owned)))
            .collect();
        let folder = path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_owned();
        let stamp = modified(&path);
        let mut watched = vec![(path, stamp)];
        for uri in uris {
            let file = folder.join(uri);
            let stamp = modified(&file);
            watched.push((file, stamp));
        }
        self.watched = watched;
    }
}

#[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
const CHECK_EVERY: std::time::Duration = std::time::Duration::from_millis(250);

#[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
fn modified(path: &std::path::Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
}

#[cfg(feature = "dev")]
pub(crate) fn reload_checked(
    scene: Option<&engine_scene::SceneInstance>,
    source: &str,
    (expected_revision, revision): (&str, &str),
) -> crate::Result<SceneDocument> {
    let Some(scene) = scene else {
        return Err(Error::new(ErrorCode::UnsupportedCommand, "$.name").got("reload"));
    };
    let (Ok(expected), Ok(target)) = (expected_revision.parse::<u64>(), revision.parse::<u64>())
    else {
        return Err(Error::new(ErrorCode::InvalidRevision, "$.args.revision")
            .expected(Expected::Integer)
            .got(revision));
    };
    if scene.revision() != expected {
        return Err(
            Error::new(ErrorCode::RevisionConflict, "$.args.expected_revision")
                .got(expected_revision)
                .nearest(scene.revision().to_string()),
        );
    }
    let document = SceneDocument::from_json(source)?;
    if document.id != scene.id() || document.revision != target {
        return Err(Error::new(ErrorCode::RevisionConflict, "$.args.revision")
            .expected(Expected::Matching)
            .got(serde_json::json!({
                "id": document.id,
                "revision": document.revision.to_string(),
            })));
    }
    Ok(document)
}
