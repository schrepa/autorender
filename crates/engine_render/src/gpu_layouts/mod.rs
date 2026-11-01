#[derive(Debug)]
pub struct Mirror {
    pub name: &'static str,
    pub module: Option<&'static str>,
}

#[derive(Debug)]
pub struct Field {
    pub name: &'static str,
    pub offset: usize,
    pub size: usize,
}

#[derive(Debug)]
pub struct Layout {
    pub rust: &'static str,
    pub mirrors: &'static [Mirror],
    pub size: usize,
    pub fields: &'static [Field],
}

impl Layout {
    pub const fn field(&self, name: &str) -> &Field {
        let mut index = 0;
        while index < self.fields.len() {
            if same(self.fields[index].name, name) {
                return &self.fields[index];
            }
            index += 1;
        }
        panic!("field_unknown")
    }
}

const fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut index = 0;
    while index < a.len() {
        if a[index] != b[index] {
            return false;
        }
        index += 1;
    }
    true
}

macro_rules! gpu_structs {
    (@module) => { None };
    (@module $module:literal) => { Some($module) };
    ($(
        $(#[$meta:meta])*
        $vis:vis struct $name:ident $(as [$($mirror:literal $(in $module:literal)?),* $(,)?])? {
            $($(#[$field_meta:meta])* $field_vis:vis $field:ident: $ty:ty),* $(,)?
        }
    )*) => {
        $(
            $(#[$meta])*
            #[repr(C)]
            $vis struct $name {
                $($(#[$field_meta])* pub $field: $ty,)*
            }

            impl $name {
                pub const LAYOUT: $crate::gpu_layouts::Layout = $crate::gpu_layouts::Layout {
                    rust: stringify!($name),
                    mirrors: &[$($($crate::gpu_layouts::Mirror {
                        name: $mirror,
                        module: gpu_structs!(@module $($module)?),
                    }),*)?],
                    size: ::core::mem::size_of::<$name>(),
                    fields: &[$($crate::gpu_layouts::Field {
                        name: stringify!($field),
                        offset: ::core::mem::offset_of!($name, $field),
                        size: ::core::mem::size_of::<$ty>(),
                    }),*],
                };
            }
        )*
    };
}

pub mod constants;

pub mod scene;
