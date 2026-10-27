// Shadow desktop window (Tauri 2): native desktop shell for the Shadow web UI.
// Opens http://127.0.0.1:8081 when ready, displays loading/status splash while starting,
// keeps a system tray icon, and stores settings in %APPDATA%\Shadow\desktop.json.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};
use tauri_plugin_opener::OpenerExt;

const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[derive(Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct Config {
    url: String,
    start_server: bool,
    launcher: String,
    console_log: String,
    stop_on_quit: bool,
    close_to_tray: bool,
    load_timeout_minutes: u64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            url: "http://127.0.0.1:8081".into(),
            start_server: false,
            launcher: String::new(),
            console_log: String::new(),
            stop_on_quit: false,
            close_to_tray: true,
            load_timeout_minutes: 5,
        }
    }
}

struct State {
    cfg: Config,
    cfg_path: PathBuf,
    child: Mutex<Option<Child>>,
    started_by_us: AtomicBool,
    quitting: AtomicBool,
    generation: AtomicU64,
}

#[derive(Clone, Serialize)]
struct Status {
    state: &'static str,
    secs: u64,
    lines: Vec<String>,
}

fn config_path() -> PathBuf {
    if let Ok(p) = std::env::var("SHADOW_DESKTOP_CONFIG") {
        return PathBuf::from(p);
    }
    let base = std::env::var("APPDATA").unwrap_or_else(|_| ".".into());
    PathBuf::from(base).join("Shadow").join("desktop.json")
}

fn load_config(path: &PathBuf) -> Config {
    match std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<Config>(t.trim_start_matches('\u{feff}')).ok())
    {
        Some(c) => c,
        None => {
            let c = Config::default();
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            if !path.exists() {
                let _ = std::fs::write(path, serde_json::to_string_pretty(&c).unwrap_or_default());
            }
            c
        }
    }
}

fn log(state: &State, msg: &str) {
    if let Some(dir) = state.cfg_path.parent() {
        let _ = std::fs::create_dir_all(dir);
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("desktop.log")) {
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let _ = writeln!(f, "{} {}", secs, msg);
        }
    }
}

fn tail_log(path: &str, n: usize) -> Vec<String> {
    if path.is_empty() {
        return vec![];
    }
    let Ok(mut f) = std::fs::File::open(path) else { return vec![] };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let take = len.min(16384);
    let mut buf = vec![0u8; take as usize];
    use std::io::Seek;
    if f.seek(std::io::SeekFrom::Start(len - take)).is_err() || f.read_exact(&mut buf).is_err() {
        return vec![];
    }
    let text = String::from_utf8_lossy(&buf).to_string();
    let lines: Vec<String> = text.lines().filter(|l| !l.trim().is_empty()).map(|l| l.chars().take(160).collect()).collect();
    lines[lines.len().saturating_sub(n)..].to_vec()
}

fn host_port(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("http://")?;
    let hostport = rest.split('/').next()?.to_string();
    let host = hostport.split(':').next()?.to_string();
    Some((if hostport.contains(':') { hostport } else { format!("{}:80", hostport) }, host))
}

fn probe(url: &str) -> bool {
    probe_path(url, "/health") || probe_path(url, "/v1/status")
}

fn probe_path(url: &str, path: &str) -> bool {
    let Some((addr, host)) = host_port(url) else { return false };
    let Some(sock) = addr.to_socket_addrs().ok().and_then(|mut a| a.next()) else { return false };
    let Ok(mut s) = TcpStream::connect_timeout(&sock, Duration::from_secs(2)) else { return false };
    let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
    let req = format!("GET {} HTTP/1.0\r\nHost: {}\r\nConnection: close\r\n\r\n", path, host);
    if s.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut body = Vec::new();
    let _ = s.take(65536).read_to_end(&mut body);
    let text = String::from_utf8_lossy(&body);
    (text.starts_with("HTTP/1.0 200") || text.starts_with("HTTP/1.1 200")) && !text.replace(' ', "").contains("\"state\":\"loading\"")
}

fn start_server(state: &State) {
    let mut child = state.child.lock().unwrap();
    if child.is_some() || !state.cfg.start_server || state.cfg.launcher.is_empty() {
        return;
    }
    if !std::path::Path::new(&state.cfg.launcher).exists() {
        log(state, &format!("launcher does not exist: {}", state.cfg.launcher));
        return;
    }
    match Command::new("powershell.exe")
        .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", &state.cfg.launcher])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
    {
        Ok(c) => {
            log(state, &format!("started launcher, pid {}", c.id()));
            state.started_by_us.store(true, Ordering::SeqCst);
            *child = Some(c);
        }
        Err(e) => log(state, &format!("launcher failed: {}", e)),
    }
}

fn stop_server(state: &State) {
    *state.child.lock().unwrap() = None;
    state.started_by_us.store(false, Ordering::SeqCst);
}

fn navigate(app: &AppHandle, url: &str) {
    if let Some(w) = app.get_webview_window("main") {
        if let Ok(u) = url.parse() {
            let _ = w.navigate(u);
        }
    }
}

fn open_chat(app: &AppHandle) {
    let state = app.state::<State>();
    let gen = state.generation.fetch_add(1, Ordering::SeqCst) + 1;
    let app = app.clone();
    std::thread::spawn(move || {
        let state = app.state::<State>();
        let url = state.cfg.url.clone();
        log(&state, &format!("connecting to {}", url));
        if probe(&url) {
            navigate(&app, &url);
            return;
        }
        if let Some(w) = app.get_webview_window("main") {
            let _ = w.eval("if (!location.href.includes('tauri.localhost') && !location.href.startsWith('tauri://')) location.replace('http://tauri.localhost/');");
        }
        start_server(&state);
        let started = Instant::now();
        loop {
            std::thread::sleep(Duration::from_millis(1500));
            if state.generation.load(Ordering::SeqCst) != gen {
                return;
            }
            if probe(&url) {
                navigate(&app, &url);
                return;
            }
            let secs = started.elapsed().as_secs();
            if secs > state.cfg.load_timeout_minutes * 60 {
                let _ = app.emit("status", Status { state: "failed", secs, lines: tail_log(&state.cfg.console_log, 14) });
                return;
            }
            let st = if state.cfg.start_server { "loading" } else { "waiting" };
            let _ = app.emit("status", Status { state: st, secs, lines: tail_log(&state.cfg.console_log, 8) });
        }
    });
}

fn show_window(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.unminimize();
        let _ = w.show();
        let _ = w.set_focus();
    }
}

fn open_path(app: &AppHandle, path: &str) {
    if std::path::Path::new(path).exists() {
        let _ = app.opener().open_path(path, None::<&str>);
    }
}

#[tauri::command]
fn retry(app: AppHandle) {
    open_chat(&app);
}

#[tauri::command]
fn open_log(app: AppHandle) {
    let state = app.state::<State>();
    open_path(&app, &state.cfg.console_log);
}

#[tauri::command]
fn quit_app(app: AppHandle) {
    app.exit(0);
}

fn main() {
    let cfg_path = config_path();
    let cfg = load_config(&cfg_path);
    let origin = cfg.url.trim_end_matches('/').to_string();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| show_window(app)))
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .plugin(tauri_plugin_opener::init())
        .manage(State {
            cfg,
            cfg_path,
            child: Mutex::new(None),
            started_by_us: AtomicBool::new(false),
            quitting: AtomicBool::new(false),
            generation: AtomicU64::new(0),
        })
        .invoke_handler(tauri::generate_handler![retry, open_log, quit_app])
        .setup(move |app| {
            let handle = app.handle().clone();
            let allowed = origin.clone();
            let nav_handle = handle.clone();
            WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("Shadow")
                .inner_size(1280.0, 860.0)
                .min_inner_size(720.0, 520.0)
                .disable_drag_drop_handler()
                .on_navigation(move |url| {
                    let s = url.as_str();
                    if s.starts_with(&allowed) || url.host_str() == Some("tauri.localhost") || url.scheme() == "tauri" {
                        return true;
                    }
                    if url.scheme() == "http" || url.scheme() == "https" {
                        let _ = nav_handle.opener().open_url(s, None::<&str>);
                    }
                    false
                })
                .build()?;

            let show = MenuItem::with_id(app, "show", "Show Shadow", true, None::<&str>)?;
            let restart = MenuItem::with_id(app, "restart", "Reconnect to Shadow", true, None::<&str>)?;
            let settings = MenuItem::with_id(app, "settings", "Open Settings", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let sep = PredefinedMenuItem::separator(app)?;
            let menu = Menu::with_items(app, &[&show, &restart, &sep, &settings, &quit])?;

            let mut tray = TrayIconBuilder::new()
                .tooltip("Shadow")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "show" => show_window(app),
                    "restart" => {
                        let app = app.clone();
                        std::thread::spawn(move || {
                            show_window(&app);
                            open_chat(&app);
                        });
                    }
                    "settings" => {
                        let state = app.state::<State>();
                        open_path(app, &state.cfg_path.to_string_lossy());
                    }
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = event {
                        show_window(tray.app_handle());
                    }
                });
            if let Some(icon) = app.default_window_icon() {
                tray = tray.icon(icon.clone());
            }
            tray.build(app)?;

            open_chat(&handle);
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                let state = window.app_handle().state::<State>();
                if !state.quitting.load(Ordering::SeqCst) && state.cfg.close_to_tray {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building Shadow desktop UI");

    app.run(|app, event| {
        if let RunEvent::ExitRequested { api, .. } = event {
            let state = app.state::<State>();
            if state.quitting.swap(true, Ordering::SeqCst) {
                return;
            }
            if state.started_by_us.load(Ordering::SeqCst) && state.cfg.stop_on_quit {
                api.prevent_exit();
                let app = app.clone();
                std::thread::spawn(move || {
                    stop_server(&app.state::<State>());
                    app.exit(0);
                });
            }
        }
    });
}
