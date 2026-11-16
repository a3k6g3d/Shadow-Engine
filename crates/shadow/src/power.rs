//! GPU power limits at server start (`shadow serve --power-limit "5070=200,5060 Ti=150"`, or the launcher's `extraFlags`; env `SHADOW_POWER_LIMITS`).
//!
//! LLM inference leaves a card well under its power limit (STATUS 150: an RTX 5070 peaks at 196 W of 250 W in a 20k-token prompt, an RTX 5060 Ti at 113 W of 180 W), so a
//! lower limit costs nothing measurable and keeps a small power supply and the fans calm.  `nvidia-smi -pl` needs an elevated process; Shadow runs as the user, so when a
//! limit has to change this asks Windows for an elevated `nvidia-smi` (`Start-Process -Verb RunAs`), which goes through without a prompt when the account's UAC setting is
//! "elevate without prompting" and otherwise shows the prompt.  A limit already at the wanted value needs no elevation; a failure is reported and never stops the server.
//! The limit lasts until the next reboot or driver reload; the next `shadow serve` sets it again.

use std::process::Command;

/// One card as `nvidia-smi` reports it.
struct Card {
    uuid: String,
    name: String,
    limit: f64,
    min: f64,
    max: f64,
}

fn query() -> Result<Vec<Card>, String> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=uuid,name,power.limit,power.min_limit,power.max_limit", "--format=csv,noheader,nounits"])
        .output()
        .map_err(|e| format!("nvidia-smi: {e}"))?;
    if !out.status.success() {
        return Err(format!("nvidia-smi: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    let mut cards = vec![];
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        if f.len() < 5 {
            continue;
        }
        let num = |s: &str| s.parse::<f64>().unwrap_or(0.0);
        cards.push(Card { uuid: f[0].to_string(), name: f[1].to_string(), limit: num(f[2]), min: num(f[3]), max: num(f[4]) });
    }
    Ok(cards)
}

/// `"5070=200,5060 Ti=150"`: (name part, watts) pairs; the name part is matched case-insensitively inside the card's name, `all` matches every card.
fn parse(spec: &str) -> Result<Vec<(String, f64)>, String> {
    let mut v = vec![];
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (name, w) = part.rsplit_once('=').ok_or_else(|| format!("{part:?}: expected NAME=WATTS"))?;
        let w: f64 = w.trim().parse().map_err(|_| format!("{part:?}: {w:?} is not a number of watts"))?;
        v.push((name.trim().to_lowercase(), w));
    }
    Ok(v)
}

/// Sets the limits `spec` names (clamped to each card's allowed range) and logs what happened.
pub fn apply(spec: &str) {
    let wants = match parse(spec) {
        Ok(w) => w,
        Err(e) => return eprintln!("shadow serve: --power-limit: {e}"),
    };
    let cards = match query() {
        Ok(c) => c,
        Err(e) => return eprintln!("shadow serve: power limit not set: {e}"),
    };
    let mut todo: Vec<(&Card, f64)> = vec![];
    for c in &cards {
        let lname = c.name.to_lowercase();
        let Some(&(_, w)) = wants.iter().find(|(n, _)| n == "all" || lname.contains(n.as_str())) else { continue };
        let target = w.clamp(c.min, c.max);
        if (target - w).abs() > 0.5 {
            eprintln!("shadow serve: power limit: {} allows {:.0}-{:.0} W, {w:.0} W becomes {target:.0} W", c.name, c.min, c.max);
        }
        if (c.limit - target).abs() < 0.5 {
            eprintln!("shadow serve: power limit: {} already at {:.0} W", c.name, c.limit);
        } else {
            todo.push((c, target));
        }
    }
    if todo.is_empty() {
        return;
    }
    // one elevated PowerShell for all the cards: a single elevation
    let mut script = String::new();
    for (c, w) in &todo {
        script += &format!("& nvidia-smi -i {} -pl {w:.0}; ", c.uuid);
    }
    let cmd = format!(
        "$p = Start-Process powershell -Verb RunAs -WindowStyle Hidden -PassThru -ArgumentList '-NoProfile','-Command','{}'; if (-not $p.WaitForExit(30000)) {{ exit 2 }}",
        script.replace('\'', "''")
    );
    let ran = Command::new("powershell").args(["-NoProfile", "-Command", &cmd]).status();
    match ran {
        Ok(s) if s.success() => {}
        Ok(s) => eprintln!("shadow serve: power limit: the elevated nvidia-smi did not finish ({s}); it needs administrator rights"),
        Err(e) => eprintln!("shadow serve: power limit: could not start PowerShell: {e}"),
    }
    match query() {
        Ok(after) => {
            for (c, w) in &todo {
                let now = after.iter().find(|a| a.uuid == c.uuid).map_or(f64::NAN, |a| a.limit);
                if (now - w).abs() < 0.5 {
                    eprintln!("shadow serve: power limit: {} {:.0} W -> {now:.0} W", c.name, c.limit);
                } else {
                    eprintln!("shadow serve: power limit: {} is still {now:.0} W (wanted {w:.0} W): setting it needs an elevated process", c.name);
                }
            }
        }
        Err(e) => eprintln!("shadow serve: power limit: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn a_spec_names_cards_and_watts() {
        assert_eq!(parse("5070=200, 5060 Ti=150").unwrap(), vec![("5070".to_string(), 200.0), ("5060 ti".to_string(), 150.0)]);
        assert_eq!(parse("all=180").unwrap(), vec![("all".to_string(), 180.0)]);
        assert!(parse("5070").is_err());
        assert!(parse("5070=lots").is_err());
    }
}
