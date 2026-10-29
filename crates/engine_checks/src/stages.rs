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
        println!(
            "{}",
            serde_json::json!({"event": "stage_started", "stage": stage.name})
        );
        let start = Instant::now();
        let outcome = match (stage.run)(context) {
            Ok(()) => Outcome::Passed,
            Err(error) => {
                let record = serde_json::json!({
                    "event": "stage_failed",
                    "stage": stage.name,
                    "error": error,
                });
                eprintln!("{record}");
                Outcome::Failed(error)
            }
        };
        results.push((stage.name, outcome, start.elapsed()));
    }
    for (name, outcome, time) in &results {
        let result = match outcome {
            Outcome::Passed => "ok",
            Outcome::Failed(_) => "failed",
            Outcome::Skipped(_) => "skipped",
        };
        let record = serde_json::json!({
            "event": "stage_result",
            "stage": name,
            "result": result,
            "seconds": time.as_secs_f64(),
        });
        println!("{record}");
    }
    let total: Duration = results.iter().map(|(_, _, time)| *time).sum();
    println!(
        "{}",
        serde_json::json!({"event": "stages_total", "seconds": total.as_secs_f64()})
    );
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
