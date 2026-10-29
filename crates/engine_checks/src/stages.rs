use std::time::{Duration, Instant};

pub struct Stage<C: ?Sized + 'static> {
    pub name: &'static str,
    pub needs: &'static [&'static str],
    pub run: fn(&C) -> Result<(), String>,
}

enum Outcome {
    Passed,
    Failed(String),
    Skipped(String),
}

pub fn run<C: ?Sized + 'static>(context: &C, stages: &[Stage<C>]) -> Result<(), String> {
    let mut results: Vec<(&str, Outcome, Duration)> = Vec::with_capacity(stages.len());
    for stage in stages {
        let blocked: Vec<&str> = stage
            .needs
            .iter()
            .copied()
            .filter(|need| {
                !results
                    .iter()
                    .any(|(name, outcome, _)| name == need && matches!(outcome, Outcome::Passed))
            })
            .collect();
        if !blocked.is_empty() {
            results.push((
                stage.name,
                Outcome::Skipped(blocked.join(", ")),
                Duration::ZERO,
            ));
            continue;
        }
        println!("\n== {} ==", stage.name);
        let start = Instant::now();
        let outcome = match (stage.run)(context) {
            Ok(()) => Outcome::Passed,
            Err(error) => {
                eprintln!("{} failed: {error}", stage.name);
                Outcome::Failed(error)
            }
        };
        results.push((stage.name, outcome, start.elapsed()));
    }
    let width = results
        .iter()
        .map(|(name, _, _)| name.len())
        .max()
        .unwrap_or(0)
        .max("stage".len());
    println!("\n{:width$}  result   seconds", "stage");
    for (name, outcome, time) in &results {
        let result = match outcome {
            Outcome::Passed => "ok",
            Outcome::Failed(_) => "FAILED",
            Outcome::Skipped(_) => "skipped",
        };
        println!("{name:width$}  {result:7}  {:7.1}", time.as_secs_f64());
    }
    let total: Duration = results.iter().map(|(_, _, time)| *time).sum();
    println!("{:width$}           {:7.1}", "total", total.as_secs_f64());
    let problems: Vec<String> = results
        .iter()
        .filter_map(|(name, outcome, _)| match outcome {
            Outcome::Passed => None,
            Outcome::Failed(error) => Some(format!("{name} failed: {error}")),
            Outcome::Skipped(blocked) => Some(format!("{name} skipped: needs {blocked}")),
        })
        .collect();
    if problems.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{} of {} stages did not pass:\n{}",
        problems.len(),
        results.len(),
        problems.join("\n")
    ))
}
