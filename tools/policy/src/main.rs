//! Checks the repository's policy (tools/policy) and prints each problem.

use std::process::ExitCode;

fn main() -> ExitCode {
    let result = policy::repository_root().and_then(|root| policy::check(&root));
    match result {
        Ok(()) => {
            println!("Policy holds.");
            ExitCode::SUCCESS
        }
        Err(problems) => {
            eprintln!("policy: {problems}");
            ExitCode::FAILURE
        }
    }
}
