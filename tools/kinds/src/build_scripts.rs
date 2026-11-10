use std::fs;
use std::path::Path;

use crate::techniques::{Token, mark, tokens};

const CLIMBS: &[&str] = &["parent", "ancestors"];

pub fn check(
    label: &str,
    script: &Path,
    shown: impl Fn(&Path) -> String,
    problems: &mut Vec<String>,
) -> Result<(), String> {
    let text =
        fs::read_to_string(script).map_err(|error| format!("{}: {error}", script.display()))?;
    let located = tokens(&text);
    for (at, found) in located.iter().enumerate() {
        let reason = match &found.token {
            Token::Text(path) if leaves(path) => {
                format!("names the path {path:?}, which leaves the crate")
            }
            Token::Word(name)
                if CLIMBS.contains(&name.as_str()) && mark(&located, at.wrapping_sub(1), '.') =>
            {
                format!("climbs out of the crate with .{name}()")
            }
            Token::Text(_) | Token::Word(_) | Token::Mark(_) => continue,
        };
        problems.push(format!(
            "{label}: {}:{} {reason}; a core crate's build script reads no file outside its own crate",
            shown(script),
            found.line
        ));
    }
    Ok(())
}

fn leaves(path: &str) -> bool {
    let mut characters = path.chars();
    let drive = matches!(
        (characters.next(), characters.next(), characters.next()),
        (Some(letter), Some(':'), None | Some('/' | '\\')) if letter.is_ascii_alphabetic()
    );
    drive || path.starts_with(['/', '\\']) || path.split(['/', '\\']).any(|part| part == "..")
}
