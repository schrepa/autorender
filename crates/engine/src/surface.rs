#[doc(hidden)]
#[macro_export]
macro_rules! __game_surface {
    ($callback:ident { $($context:tt)* }) => {
        $crate::$callback! {
            { $($context)* }
            always_mut {
                fn frame(timestamp: f64, width: u32, height: u32) -> Result<(), $crate::__private::JsValue>;
                fn key(code: &str, down: bool) -> bool;
                fn mouse_button(button: u32, down: bool) -> bool;
                fn motion(dx: f32, dy: f32);
                fn pointer(x: f32, y: f32);
                fn pointer_left();
                fn wheel(right_notches: f32, up_notches: f32);
                fn pad(index: u32, buttons: u32, axes: Vec<f32>);
                fn touch(id: u32, x: f32, y: f32, phase: u32);
                fn touch_motion(id: u32, dx: f32, dy: f32);
                fn animation_frame(timestamp: f64, width: u32, height: u32) -> Result<u32, $crate::__private::JsValue>;
                fn set_preparation(enabled: bool);
                fn set_job_workers(workers: u32) -> bool;
                fn set_job_cache(caching: bool, verify: bool);
                fn take_jobs() -> $crate::__private::js_sys::Array;
                fn job_kept(id: f64, key: String, output: Option<$crate::__private::js_sys::Uint8Array>);
                fn lose_job_worker(worker: u32, error: &str, replaced: bool);
                fn job_task() -> Option<u32>;
                fn job_task_done(worker: u32);
                fn complete_job(id: f64, output: $crate::__private::js_sys::Uint8Array);
                fn fail_job(id: f64, error: String);
                fn report_job_cache(hits: f64, misses: f64, bytes: f64, mismatches: f64);
                fn suspend();
                fn pause(reason: u32);
                fn resume(reason: u32);
                fn reset_camera();
                fn reload_scene(source: &str, files: SceneFileSet) -> Result<(), $crate::__private::JsValue>;
                fn set_file_root(root: &str);
                fn set_pixel_ratio(ratio: f64);
                fn apply_host_settings(query: &str) -> Result<(), $crate::__private::JsValue>;
            }
            always_ref {
                fn stats() -> String as Json<FrameStats>;
                fn cursor_mode() -> u32;
                fn job_hash() -> Vec<u8>;
            }
            dev_mut {
                fn dev_request(line: &str, context: &str) -> String as Json<Step>;
                fn dev_resume(outcome: &str, context: &str) -> String as Json<Step>;
                fn dev_reloaded(source: &str, context: &str, files: Option<SceneFileSet>) -> String as Json<Step>;
                fn dev_refusal(error: &str) -> String as Json<Reply>;
            }
        }
    };
}

#[macro_export]
macro_rules! game {
    (
        $vis:vis struct $name:ident($game:ty), $title:literal, $bricks:path
        $(, build = $build:path)? $(, scene = $scene:literal)? $(,)?
    ) => {
        include!(concat!(env!("OUT_DIR"), "/brick.rs"));

        pub struct GameBrick;

        impl $crate::Brick for GameBrick {
            const SOURCE: &'static $crate::Source = &SOURCE;

            fn register(bricks: &mut $crate::Bricks<'_>) {
                <$game as $crate::Game>::register(bricks);
            }
        }

        #[doc(hidden)]
        pub fn __game_bricks(list: $crate::BrickList) -> $crate::BrickList {
            <$crate::BrickList as $crate::Listing>::with::<GameBrick>($bricks(list))
        }

        pub const GAME: $crate::GameEntry<$game> = $crate::GameEntry {
            title: $title,
            folder: env!("CARGO_MANIFEST_DIR"),
            bricks: __game_bricks,
            build: $crate::__game_build!($game $(, $build)?),
            scene: $crate::__game_scene!($($scene)?),
        };

        $crate::__worker::export_worker!(__game_bricks);

        $crate::__game_surface!(__export_game { $vis $name $game });

        #[cfg(not(target_arch = "wasm32"))]
        pub fn native(arguments: impl Iterator<Item = String>) -> ::std::process::ExitCode {
            ::engine_native::main(GAME, arguments)
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __game_build {
    ($game:ty) => {
        $crate::game::default_build::<$game>
    };
    ($game:ty, $build:path) => {
        $build
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __game_scene {
    () => {
        ::core::option::Option::None
    };
    ($scene:literal) => {
        ::core::option::Option::Some($crate::DefaultScene {
            name: $scene,
            path: concat!(env!("CARGO_MANIFEST_DIR"), "/", $scene),
            text: include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", $scene)),
        })
    };
}

#[macro_export]
macro_rules! brick {
    ($name:ident) => {
        include!(concat!(
            env!("OUT_DIR"),
            "/bricks/",
            stringify!($name),
            ".rs"
        ));
    };
}

#[macro_export]
macro_rules! game_main {
    ($run:path) => {
        #[cfg(not(target_arch = "wasm32"))]
        fn main() -> ::std::process::ExitCode {
            $run(::std::env::args().skip(1))
        }

        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        #[global_allocator]
        static ALLOCATOR: $crate::profiling::TrackingAllocator<::std::alloc::System> =
            $crate::profiling::TrackingAllocator::new(::std::alloc::System);

        #[cfg(target_arch = "wasm32")]
        fn main() {}
    };
}

#[cfg(target_arch = "wasm32")]
#[doc(hidden)]
#[macro_export]
macro_rules! __export_game {
    (
        { $vis:vis $name:ident $game:ty }
        always_mut { $( fn $am:ident($($am_arg:ident: $am_ty:ty),*) $(-> $am_ret:ty $(as $am_json:ty)?)?; )* }
        always_ref { $( fn $ar:ident($($ar_arg:ident: $ar_ty:ty),*) $(-> $ar_ret:ty $(as $ar_json:ty)?)?; )* }
        dev_mut { $( fn $dm:ident($($dm_arg:ident: $dm_ty:ty),*) $(-> $dm_ret:ty $(as $dm_json:ty)?)?; )* }
    ) => {
        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        #[derive(Default)]
        pub struct SceneFileSet {
            files: $crate::scene::SceneFiles,
        }

        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        impl SceneFileSet {
            #[wasm_bindgen(constructor)]
            pub fn new() -> Self {
                Self::default()
            }

            pub fn add_image(&mut self, uri: &str, width: u32, height: u32, pixels: Vec<u8>) {
                self.files.add_image(uri, width, height, pixels);
            }

            pub fn add_buffer(&mut self, uri: &str, bytes: Vec<u8>) {
                self.files.add_buffer(uri, bytes);
            }
        }

        impl SceneFileSet {
            pub fn files(&self) -> &$crate::scene::SceneFiles {
                &self.files
            }
        }

        impl ::core::convert::From<SceneFileSet> for $crate::scene::SceneFiles {
            fn from(set: SceneFileSet) -> Self {
                set.files
            }
        }

        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen, js_name = SceneLoad)]
        pub struct PageSceneLoad {
            load: $crate::browser::SceneLoading,
        }

        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen, js_class = SceneLoad)]
        impl PageSceneLoad {
            #[wasm_bindgen(constructor)]
            pub fn new(source: &str) -> Result<PageSceneLoad, $crate::__private::JsValue> {
                Ok(Self {
                    load: $crate::browser::SceneLoading::new(source, __game_bricks)?,
                })
            }

            pub fn take_reads(
                &mut self,
            ) -> Result<$crate::__private::js_sys::Array, $crate::__private::JsValue> {
                self.load.take_reads()
            }

            pub fn read(&mut self, uri: &str, bytes: Vec<u8>) -> Result<bool, $crate::__private::JsValue> {
                self.load.read(uri, bytes)
            }

            pub fn image(
                &mut self,
                uri: &str,
                width: u32,
                height: u32,
                pixels: Vec<u8>,
            ) -> Result<(), $crate::__private::JsValue> {
                self.load.image(uri, width, height, pixels)
            }

            pub fn decode(&mut self, uri: &str, bytes: &[u8]) -> Result<(), $crate::__private::JsValue> {
                self.load.decode(uri, bytes)
            }

            pub fn finish(&mut self) -> Result<SceneFileSet, $crate::__private::JsValue> {
                Ok(SceneFileSet {
                    files: self.load.finish()?,
                })
            }
        }

        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        $vis struct $name {
            host: $crate::browser::PageHost<$game>,
        }

        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        impl $name {
            $( pub fn $am(&mut self, $($am_arg: $am_ty),*) $(-> $am_ret)? { self.host.$am($($am_arg),*) } )*
            $( pub fn $ar(&self, $($ar_arg: $ar_ty),*) $(-> $ar_ret)? { self.host.$ar($($ar_arg),*) } )*
        }

        $crate::__if_dev! {
            #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
            impl $name {
                $( pub fn $dm(&mut self, $($dm_arg: $dm_ty),*) $(-> $dm_ret)? { self.host.$dm($($dm_arg),*) } )*
            }
        }

        #[global_allocator]
        static ALLOCATOR: $crate::__private::PageAllocator = $crate::__private::page_allocator();

        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        impl $name {
            pub async fn create(
                canvas: $crate::__private::web_sys::HtmlCanvasElement,
                scene: Option<String>,
                files: Option<SceneFileSet>,
            ) -> Result<$name, $crate::__private::JsValue> {
                Ok(Self {
                    host: $crate::browser::PageHost::start(
                        canvas,
                        GAME,
                        scene.map(|source| (source, files.map(|set| set.files).unwrap_or_default())),
                    )
                    .await?,
                })
            }
        }

        impl $name {
            pub fn host(&self) -> &$crate::GameHost<$game> {
                &self.host
            }

            pub fn host_mut(&mut self) -> &mut $crate::GameHost<$game> {
                &mut self.host
            }
        }
    };
}

#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __export_game {
    ($($surface:tt)*) => {};
}

#[cfg(not(target_arch = "wasm32"))]
pub fn runtime_typescript() -> String {
    crate::__game_surface!(__surface_typescript {})
}

#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __surface_typescript {
    (
        {}
        $( $group:ident { $( fn $method:ident($($arg:ident: $ty:ty),*) $(-> $ret:ty $(as $json:ty)?)?; )* } )*
    ) => {{
        let methods = [$( $(
            $crate::surface::SurfaceMethod {
                group: stringify!($group),
                name: stringify!($method),
                arguments: &[$((stringify!($arg), stringify!($ty))),*],
                returns: [$(stringify!($ret))?].first().copied(),
                json: [$($(stringify!($json))?)?].first().copied(),
            },
        )* )*];
        $crate::surface::typescript(&methods)
    }};
}

#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub struct SurfaceMethod {
    pub group: &'static str,
    pub name: &'static str,
    pub arguments: &'static [(&'static str, &'static str)],
    pub returns: Option<&'static str>,
    pub json: Option<&'static str>,
}

#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub fn typescript(methods: &[SurfaceMethod]) -> String {
    use std::fmt::Write;
    let mut output = String::from(
        "import type { JobOrder } from \"../engine/jobs\";\n\
         import type { Reply, Step } from \"./reply\";\n\
         import type { FrameStats } from \"./stats\";\n\n\
         export type Json<T> = string & { readonly json?: T };\n\n\
         export function parseJson<T>(json: Json<T>): T {\n  \
         return JSON.parse(json) as T;\n\
         }\n\n\
         export interface Runtime {\n",
    );
    for method in methods {
        let optional = if method.group.starts_with("always") {
            ""
        } else {
            "?"
        };
        let arguments: Vec<String> = method
            .arguments
            .iter()
            .map(|(name, ty)| {
                let ty = ty.replace(' ', "");
                match ty
                    .strip_prefix("Option<")
                    .and_then(|ty| ty.strip_suffix('>'))
                {
                    Some(inner) => format!("{name}?: {}", typescript_type(inner)),
                    None => format!("{name}: {}", typescript_type(&ty)),
                }
            })
            .collect();
        let returns = match method.json {
            Some(json) => json_type(&json.replace(' ', "")),
            None => method.returns.map_or("void".to_owned(), |ty| {
                let ty = ty.replace(' ', "");
                if let Some((ok, _)) = ty.strip_prefix("Result<").and_then(|ty| ty.split_once(','))
                {
                    typescript_type(ok)
                } else if let Some(inner) = ty
                    .strip_prefix("Option<")
                    .and_then(|ty| ty.strip_suffix('>'))
                {
                    format!("{} | undefined", typescript_type(inner))
                } else {
                    typescript_type(&ty)
                }
            }),
        };
        writeln!(
            output,
            "  {}{optional}({}): {returns};",
            method.name,
            arguments.join(", ")
        )
        .expect("string_writes");
    }
    output.push_str("  free(): void;\n}\n");
    output
}

#[cfg(not(target_arch = "wasm32"))]
fn json_type(json: &str) -> String {
    let payload = json
        .strip_prefix("Json<")
        .and_then(|json| json.strip_suffix('>'))
        .unwrap_or_else(|| panic!("json_method_type {json}"));
    format!("Json<{payload}>")
}

#[cfg(not(target_arch = "wasm32"))]
fn typescript_type(ty: &str) -> String {
    let leaf = ty.rsplit("::").next().unwrap_or(ty);
    match leaf {
        "()" => "void",
        "f32" | "f64" | "u32" | "i32" => "number",
        "bool" => "boolean",
        "String" | "&str" => "string",
        "Vec<u8>" | "Uint8Array" => "Uint8Array",
        "Vec<f32>" => "Float32Array",
        "Vec<u32>" => "Uint32Array",
        "Array" => "JobOrder[]",
        "SceneFileSet" => "unknown",
        _ => panic!("surface_typescript_type {ty}"),
    }
    .to_owned()
}
