//! The web tools of the agent layer: `web_search` (DuckDuckGo's HTML results) and `web_fetch` (a page as readable text).  The HTTP itself
//! is `curl.exe`, which ships with Windows 10 and later with its own TLS, so Shadow needs no HTTP client library; the HTML is turned into
//! text here (scripts, styles and tags dropped, entities decoded), so a page costs the model its words, not its markup.

use std::process::Command;

const UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36";

/// GETs a URL (http or https only) and returns the body; an HTTP error status is an error.
fn http_get(url: &str) -> Result<String, String> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err("only http:// and https:// URLs can be fetched".into());
    }
    let out = Command::new("curl.exe")
        .args(["-sL", "--max-time", "25", "--max-filesize", "6000000", "--compressed", "-A", UA, "-H", "Accept-Language: en", "-w", "\n%{http_code}", "--", url])
        .output()
        .map_err(|e| format!("could not run curl.exe: {e}"))?;
    let all = String::from_utf8_lossy(&out.stdout).into_owned();
    let (body, code) = all.rsplit_once('\n').unwrap_or((&all, "000"));
    match code.trim() {
        c if c.starts_with('2') => Ok(body.to_string()),
        "000" => Err(format!("no answer from {url} ({})", String::from_utf8_lossy(&out.stderr).trim())),
        c => Err(format!("HTTP {c} from {url}")),
    }
}

/// Removes `<tag ...> ... </tag>` blocks (case-insensitively), e.g. scripts and styles.
fn strip_blocks(s: &str, tag: &str) -> String {
    let lower = s.to_ascii_lowercase();
    let (open, close) = (format!("<{tag}"), format!("</{tag}>"));
    let mut out = String::with_capacity(s.len());
    let mut at = 0;
    while let Some(i) = lower[at..].find(&open) {
        out.push_str(&s[at..at + i]);
        match lower[at + i..].find(&close) {
            Some(j) => at += i + j + close.len(),
            None => return out,
        }
    }
    out.push_str(&s[at..]);
    out
}

fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        let end = rest.find(';').filter(|&e| e <= 10);
        let rep = end.and_then(|e| match &rest[1..e] {
            "amp" => Some('&'), "lt" => Some('<'), "gt" => Some('>'), "quot" => Some('"'), "apos" | "#39" => Some('\''), "nbsp" => Some(' '),
            n if n.starts_with("#x") => u32::from_str_radix(&n[2..], 16).ok().and_then(char::from_u32),
            n if n.starts_with('#') => n[1..].parse().ok().and_then(char::from_u32),
            _ => None,
        });
        match (rep, end) {
            (Some(c), Some(e)) => { out.push(c); rest = &rest[e + 1..]; }
            _ => { out.push('&'); rest = &rest[1..]; }
        }
    }
    out.push_str(rest);
    out
}

/// A page's readable text: no scripts, styles or comments; block tags become line breaks; blank runs collapse.
pub fn html_to_text(html: &str) -> String {
    let mut s = html.to_string();
    for t in ["script", "style", "noscript", "svg", "head"] {
        s = strip_blocks(&s, t);
    }
    let mut out = String::with_capacity(s.len() / 2);
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if c != '<' {
            out.push(c);
            continue;
        }
        if s[i..].starts_with("<!--") {
            match s[i..].find("-->") {
                Some(j) => { while chars.peek().is_some_and(|&(k, _)| k < i + j + 3) { chars.next(); } }
                None => break,
            }
            continue;
        }
        let end = s[i..].find('>').map_or(s.len(), |j| i + j);
        let name = s[i + 1..end].trim_start_matches('/').split(|c: char| c.is_whitespace() || c == '/').next().unwrap_or("").to_ascii_lowercase();
        if matches!(name.as_str(), "p" | "br" | "div" | "li" | "tr" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "section" | "article" | "table" | "ul" | "ol" | "pre" | "blockquote") {
            out.push('\n');
        } else if matches!(name.as_str(), "td" | "th") && !s[i + 1..].starts_with('/') {
            out.push_str(" | ");
        }
        while chars.peek().is_some_and(|&(k, _)| k <= end) { chars.next(); }
    }
    let text = decode_entities(&out);
    let mut clean = String::with_capacity(text.len());
    let mut blank = 0;
    for line in text.lines() {
        let l = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if l.is_empty() {
            blank += 1;
            if blank == 1 { clean.push('\n'); }
        } else {
            blank = 0;
            clean.push_str(&l);
            clean.push('\n');
        }
    }
    clean.trim().to_string()
}

fn percent_encode(s: &str) -> String {
    s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let (mut out, mut i) = (Vec::with_capacity(b.len()), 0);
    while i < b.len() {
        match (b[i], b.get(i + 1..i + 3).and_then(|h| std::str::from_utf8(h).ok()).and_then(|h| u8::from_str_radix(h, 16).ok())) {
            (b'%', Some(v)) => { out.push(v); i += 3; }
            (b'+', _) => { out.push(b' '); i += 1; }
            (c, _) => { out.push(c); i += 1; }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The text between the next `>` after `from` and the following `</a>`, as plain text.
fn anchor_text(html: &str, from: usize) -> (String, usize) {
    let Some(gt) = html[from..].find('>') else { return (String::new(), html.len()) };
    let start = from + gt + 1;
    let end = html[start..].find("</a>").map_or(html.len(), |j| start + j);
    (html_to_text(&html[start..end]), end)
}

/// The results of a DuckDuckGo HTML results page: (title, url, snippet).
pub fn parse_results(html: &str, max: usize) -> Vec<(String, String, String)> {
    let mut out = vec![];
    let mut at = 0;
    while out.len() < max {
        let Some(i) = html[at..].find("class=\"result__a\"") else { break };
        let tag_at = at + i;
        let tag_end = html[tag_at..].find('>').map_or(html.len(), |j| tag_at + j);
        let tag = &html[tag_at..tag_end];
        let href = tag.split("href=\"").nth(1).and_then(|h| h.split('"').next()).unwrap_or("");
        let href = decode_entities(href);
        let url = match href.split("uddg=").nth(1) {
            Some(u) => percent_decode(u.split('&').next().unwrap_or(u)),
            None if href.starts_with("//") => format!("https:{href}"),
            None => href.clone(),
        };
        let (title, after) = anchor_text(html, tag_at);
        let next = html[after..].find("class=\"result__a\"").map_or(html.len(), |j| after + j);
        let snippet = html[after..next].find("class=\"result__snippet\"").map(|j| anchor_text(html, after + j).0).unwrap_or_default();
        if !url.is_empty() && !title.is_empty() {
            out.push((title, url, snippet));
        }
        at = after.max(tag_at + 1);
    }
    out
}

pub fn web_search(query: &str, max_results: usize) -> Result<String, String> {
    if query.trim().is_empty() {
        return Err("missing required argument 'query'".into());
    }
    let html = http_get(&format!("https://html.duckduckgo.com/html/?q={}", percent_encode(query.trim())))?;
    let results = parse_results(&html, max_results.clamp(1, 15));
    if results.is_empty() {
        return Err("the search returned no results (try different words)".into());
    }
    let mut out = format!("Search results for: {}\n", query.trim());
    for (i, (title, url, snippet)) in results.iter().enumerate() {
        out.push_str(&format!("\n{}. {}\n   {}\n", i + 1, title, url));
        if !snippet.is_empty() {
            out.push_str(&format!("   {}\n", snippet));
        }
    }
    out.push_str("\nUse web_fetch with one of these URLs to read the page.");
    Ok(out)
}

pub fn web_fetch(url: &str, start: usize, max_chars: usize) -> Result<String, String> {
    let body = http_get(url.trim())?;
    let looks_html = body.trim_start().starts_with('<') || body.contains("<html") || body.contains("<body");
    let text = if looks_html { html_to_text(&body) } else { body };
    let total = text.chars().count();
    let max = max_chars.clamp(500, 40_000);
    let shown: String = text.chars().skip(start).take(max).collect();
    let end = (start + shown.chars().count()).min(total);
    let more = if end < total { format!(" Call web_fetch again with start={end} to continue.") } else { String::new() };
    Ok(format!("{url}\n{total} characters of text; showing {start} to {end}.{more}\n\n{shown}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_becomes_text() {
        let t = html_to_text("<html><head><title>x</title><style>p{}</style></head><body><h1>Hi &amp; bye</h1><script>var a=1;</script><p>One<br>two &#65;</p><!-- no --><table><tr><td>a</td><td>b</td></tr></table></body></html>");
        assert_eq!(t, "Hi & bye\n\nOne\ntwo A\n\n| a | b");
    }

    #[test]
    fn percent_round_trip() {
        assert_eq!(percent_decode(&percent_encode("rtx 5070 & gb205/ü")), "rtx 5070 & gb205/ü");
        assert_eq!(percent_decode("https%3A%2F%2Fa.b%2Fc%2Dd"), "https://a.b/c-d");
    }

    #[test]
    fn results_are_parsed() {
        let h = r#"<div class="result"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.example%2Fx%2Dy&amp;rut=ff">First <b>hit</b></a>
            <a class="result__snippet" href="x">About &amp; more</a></div>
            <div class="result"><a class="result__a" href="https://b.example/">Second</a></div>"#;
        let r = parse_results(h, 5);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0], ("First hit".to_string(), "https://a.example/x-y".to_string(), "About & more".to_string()));
        assert_eq!(r[1].1, "https://b.example/");
    }

    #[test]
    fn only_http_is_fetched() {
        assert!(http_get("file:///c:/windows/win.ini").is_err());
        assert!(http_get("-o x http://a").is_err());
    }
}
