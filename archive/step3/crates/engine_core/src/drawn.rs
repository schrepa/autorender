use std::fmt::Write as _;

impl DrawnTable {
    pub fn wgsl(&self) -> String {
        let mut text = String::new();
        for (at, decl) in self.kinds.iter().enumerate() {
            let name = engine_place::wgsl_name(decl.kind.name());
            let _ = writeln!(text, "const KIND_{name}: u32 = {}u;", at + 1);
        }
        text
    }
}
