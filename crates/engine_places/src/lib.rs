#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

extern crate engine_ecs as bevy_ecs;

#[macro_export]
macro_rules! contract {
    ($file:literal $(, $name:ident)* $(,)? $(; $($float:ident),+ $(,)?)?) => {{
        const CONSTANTS: &[(&str, $crate::contract::Value)] = &[
            $((::core::stringify!($name), $crate::contract::Value::U32($name)),)*
            $($((::core::stringify!($float), $crate::contract::Value::F32($float)),)+)?
        ];
        const TEXT: &str = ::core::include_str!($file);
        const BYTES: [u8; $crate::contract::length(CONSTANTS, TEXT)] =
            $crate::contract::written(CONSTANTS, TEXT);
        $crate::contract::text(&BYTES)
    }};
}

pub mod meshes;

use engine_place::PlaceDecl;

#[doc(hidden)]
pub mod contract {
    use engine_place::__private::wgsl::{Piece, Written, write};

    const LEAD: &[u8] = b"const ";
    const TYPED: &[u8] = b": u32 = ";
    const END: &[u8] = b"u;\n";
    const FLOAT_TYPED: &[u8] = b": f32 = ";
    const FLOAT_END: &[u8] = b";\n";
    const IMPORT: &[u8] = b"import ";
    const FLOAT_ROOM: usize = 64;

    #[derive(Clone, Copy)]
    pub enum Value {
        U32(u32),
        F32(f32),
    }

    const fn float(value: f32) -> Written<FLOAT_ROOM> {
        let written = write::<FLOAT_ROOM>(&[Piece::F32(value)]);
        assert!(written.len <= FLOAT_ROOM, "contract_float_fits");
        written
    }

    const fn width(value: Value) -> usize {
        match value {
            Value::U32(value) => TYPED.len() + digits(value) + END.len(),
            Value::F32(value) => FLOAT_TYPED.len() + float(value).len + FLOAT_END.len(),
        }
    }

    const fn digits(value: u32) -> usize {
        let mut count = 1;
        let mut rest = value / 10;
        while rest > 0 {
            count += 1;
            rest /= 10;
        }
        count
    }

    pub const fn length(constants: &[(&str, Value)], text: &str) -> usize {
        let mut total = text.len();
        let mut at = 0;
        while at < constants.len() {
            let (name, value) = constants[at];
            total += LEAD.len() + name.len() + width(value);
            at += 1;
        }
        total
    }

    const fn opens(text: &[u8], at: usize) -> bool {
        let mut index = 0;
        while index < IMPORT.len() {
            if at + index >= text.len() || text[at + index] != IMPORT[index] {
                return false;
            }
            index += 1;
        }
        true
    }

    const fn imports_end(text: &[u8]) -> usize {
        let mut at = 0;
        loop {
            let mut next = at;
            while next < text.len() && text[next].is_ascii_whitespace() {
                next += 1;
            }
            if !opens(text, next) {
                return at;
            }
            at = next;
            while at < text.len() && text[at] != b';' {
                at += 1;
            }
            while at < text.len() && text[at] != b'\n' {
                at += 1;
            }
            if at < text.len() {
                at += 1;
            }
        }
    }

    const fn put<const N: usize>(out: &mut [u8; N], mut end: usize, bytes: &[u8]) -> usize {
        let mut at = 0;
        while at < bytes.len() {
            out[end] = bytes[at];
            end += 1;
            at += 1;
        }
        end
    }

    pub const fn written<const N: usize>(
        constants: &[(&str, Value)],
        text: &str,
    ) -> [u8; N] {
        let (imports, rest) = text.as_bytes().split_at(imports_end(text.as_bytes()));
        let mut out = [0; N];
        let mut end = put(&mut out, 0, imports);
        let mut at = 0;
        while at < constants.len() {
            let (name, value) = constants[at];
            end = put(&mut out, end, LEAD);
            end = put(&mut out, end, name.as_bytes());
            match value {
                Value::U32(value) => {
                    end = put(&mut out, end, TYPED);
                    let mut place = digits(value);
                    let mut rest = value;
                    while place > 0 {
                        place -= 1;
                        out[end + place] = b'0' + (rest % 10) as u8;
                        rest /= 10;
                    }
                    end += digits(value);
                    end = put(&mut out, end, END);
                }
                Value::F32(value) => {
                    end = put(&mut out, end, FLOAT_TYPED);
                    let written = float(value);
                    let (shown, _) = written.bytes.split_at(written.len);
                    end = put(&mut out, end, shown);
                    end = put(&mut out, end, FLOAT_END);
                }
            }
            at += 1;
        }
        put(&mut out, end, rest);
        out
    }

    pub const fn text(bytes: &'static [u8]) -> &'static str {
        match core::str::from_utf8(bytes) {
            Ok(text) => text,
            Err(_) => panic!("contract_text_is_utf8"),
        }
    }
}

pub const PLACES: &[PlaceDecl] = &[
    PlaceDecl::slot::<meshes::Instances>(),
];
