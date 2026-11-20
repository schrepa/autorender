#[doc(hidden)]
#[macro_export]
macro_rules! __game_surface {
    ($callback:ident { $($context:tt)* }) => {
        $crate::$callback! {
            { $($context)* }
            always_mut {
                fn key(code: &str, down: bool);
                fn mouse_button(button: u32, down: bool);
                fn motion(dx: f32, dy: f32);
                fn pointer(x: f32, y: f32);
                fn pointer_left();
                fn wheel(right_notches: f32, up_notches: f32);
                fn pad(index: u32, buttons: u32, axes: Vec<f32>);
                fn touch(id: u32, x: f32, y: f32, phase: u32);
                fn touch_motion(id: u32, dx: f32, dy: f32);
                fn animation_frame(timestamp: f64, width: u32, height: u32) -> Result<u32, $crate::__private::JsValue>;
                fn suspend();
                fn focused(focused: bool);
                fn hidden(hidden: bool);
                fn reset_camera();
                fn reload_scene(source: &str) -> Result<(), $crate::__private::JsValue>;
                fn file(uri: &str, bytes: Vec<u8>) -> Result<(), $crate::__private::JsValue>;
                fn file_failed(uri: &str, error: &str) -> Result<(), $crate::__private::JsValue>;
                fn set_file_root(root: &str);
                fn set_pixel_ratio(ratio: f64);
                fn apply_host_settings(query: &str) -> Result<(), $crate::__private::JsValue>;
                fn take_sound() -> Vec<f64>;
                fn take_sound_bytes() -> Vec<u8>;
                fn sound_opened(running: bool, rate: u32, channels: u32, latency_seconds: f32);
                fn sound_open_failed(error: String);
                fn sound_decoded(sound: u32, frames: f64, rate: u32, channels: u32);
                fn sound_failed(sound: u32, error: String);
                fn voice_ended(voice: u32);
                fn voice_dropped(voice: u32, error: String);
            }
            always_ref {
                fn stats() -> String as Json<FrameStats>;
                fn cursor_mode() -> u32;
                fn files() -> String as Json<SceneFileList>;
            }
            dev_mut {
                fn dev_push(line: &str) -> u32;
                fn dev_take() -> String as Json<DevTake>;
                fn dev_pixels() -> Vec<u8>;
                fn dev_reloaded(source: &str);
                fn dev_reload_ended(error: Option<String>);
            }
        }
    };
}

#[macro_export]
macro_rules! game {
    (
        $vis:vis struct $name:ident($game:ty), $title:literal, $bricks:path,
        scene = $scene:literal $(,)?
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

        pub const GAME: $crate::GameEntry = $crate::__game_entry! {
            name: env!("CARGO_PKG_NAME"),
            title: $title,
            bricks: __game_bricks,
            scene: $scene,
            linked: LINKED,
        };

        $crate::__game_surface!(__export_game { $vis $name $game });

        #[cfg(not(target_arch = "wasm32"))]
        pub fn native(arguments: impl Iterator<Item = String>) -> ::std::process::ExitCode {
            ::engine_native::main::<$game>(GAME, arguments)
        }
    };
}

#[cfg(feature = "dev")]
#[doc(hidden)]
#[macro_export]
macro_rules! __game_entry {
    ($($field:tt)*) => {
        $crate::GameEntry { folder: env!("CARGO_MANIFEST_DIR"), $($field)* }
    };
}

#[cfg(not(feature = "dev"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __game_entry {
    ($($field:tt)*) => {
        $crate::GameEntry { $($field)* }
    };
}

#[macro_export]
macro_rules! brick {
    () => {
        include!(concat!(env!("OUT_DIR"), "/brick.rs"));
    };
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
        static ALLOCATOR: $crate::__private::TrackingAllocator<::std::alloc::System> =
            $crate::__private::TrackingAllocator::new(::std::alloc::System);

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
                scene: String,
            ) -> Result<$name, $crate::__private::JsValue> {
                Ok(Self {
                    host: $crate::browser::PageHost::<$game>::start(canvas, GAME, &scene).await?,
                })
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
        "import type { FrameStats } from \"./stats\";\n\n\
         export interface DevTake {\n  replies: [number, string][];\n  pictures: { file: string; width: number; height: number }[];\n  reload: boolean;\n}\n\n\
         export type SceneFileList = { uri: string; group: string }[];\n\n\
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
        "f32" | "f64" | "u32" => "number",
        "bool" => "boolean",
        "String" | "&str" => "string",
        "Vec<u8>" => "Uint8Array",
        "Vec<f32>" => "Float32Array",
        "Vec<f64>" => "Float64Array",
        _ => panic!("surface_typescript_type {ty}"),
    }
    .to_owned()
}
