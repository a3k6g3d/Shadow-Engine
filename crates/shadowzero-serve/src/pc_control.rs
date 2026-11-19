//! Opt-in Windows PC Control ("Computer Use") with an always-on-top takeover notification
//! overlay, screen border glow, clickable Stop button, and global `Ctrl+Alt+Esc` kill-switch.
//!
//! Disabled by default. Every `pc_*` tool refuses to run unless PC Control is explicitly enabled
//! via `POST /v1/pc-control` (loopback-only).

use crate::events::EventHub;
use crate::orchestrator::ToolResult;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

pub const EMERGENCY_HOTKEY_LABEL: &str = "Ctrl+Alt+Esc";

struct PcControlState {
    enabled: AtomicBool,
    active: AtomicBool,
    last_action: Mutex<String>,
    active_until: Mutex<Option<Instant>>,
    banner_hwnd: AtomicIsize,
    overlay_started: AtomicBool,
    events: Mutex<Option<EventHub>>,
}

static STATE: OnceLock<PcControlState> = OnceLock::new();

fn state() -> &'static PcControlState {
    STATE.get_or_init(|| PcControlState {
        enabled: AtomicBool::new(false),
        active: AtomicBool::new(false),
        last_action: Mutex::new(String::new()),
        active_until: Mutex::new(None),
        banner_hwnd: AtomicIsize::new(0),
        overlay_started: AtomicBool::new(false),
        events: Mutex::new(None),
    })
}

/// Registers the server's `EventHub` so overlay Stop clicks and `Ctrl+Alt+Esc`
/// broadcast live `pc_control` SSE updates to all connected UI tabs.
pub fn bind_events(events: EventHub) {
    if let Ok(mut g) = state().events.lock() {
        *g = Some(events);
    }
}

fn publish_status() {
    if let Ok(g) = state().events.lock() {
        if let Some(ref hub) = *g {
            hub.publish(json!({
                "kind": "pc_control",
                "data": status_json()
            }));
        }
    }
}

pub fn is_enabled() -> bool {
    state().enabled.load(Ordering::Acquire)
}

pub fn is_active() -> bool {
    let s = state();
    if !s.enabled.load(Ordering::Acquire) {
        return false;
    }
    if s.active.load(Ordering::Acquire) {
        return true;
    }
    if let Ok(g) = s.active_until.lock() {
        if let Some(until) = *g {
            return Instant::now() < until;
        }
    }
    false
}

pub fn status_json() -> Value {
    let s = state();
    let enabled = s.enabled.load(Ordering::Acquire);
    let active = is_active();
    let last_action = s
        .last_action
        .lock()
        .map(|g| g.clone())
        .unwrap_or_default();
    json!({
        "enabled": enabled,
        "active": active,
        "last_action": last_action,
        "emergency_hotkey": EMERGENCY_HOTKEY_LABEL,
    })
}

/// Enables or disables PC Control, showing or hiding the native Windows takeover overlay.
pub fn set_enabled(enable: bool) -> Value {
    let s = state();
    s.enabled.store(enable, Ordering::Release);
    if !enable {
        s.active.store(false, Ordering::Release);
        if let Ok(mut g) = s.active_until.lock() {
            *g = None;
        }
        if let Ok(mut g) = s.last_action.lock() {
            g.clear();
        }
    } else {
        if let Ok(mut g) = s.last_action.lock() {
            *g = "Armed — waiting for command".into();
        }
        ensure_overlay_thread();
    }
    notify_overlay();
    publish_status();
    status_json()
}

struct ActionGuard;

impl ActionGuard {
    fn begin(desc: &str) -> Result<Self, String> {
        if !is_enabled() {
            return Err(
                "PC Control is OFF. Ask the user to enable PC Control in ShadowZero first.".into(),
            );
        }
        let s = state();
        s.active.store(true, Ordering::Release);
        if let Ok(mut g) = s.last_action.lock() {
            *g = desc.to_string();
        }
        if let Ok(mut g) = s.active_until.lock() {
            *g = Some(Instant::now() + Duration::from_millis(2500));
        }
        ensure_overlay_thread();
        notify_overlay();
        publish_status();
        Ok(Self)
    }
}

impl Drop for ActionGuard {
    fn drop(&mut self) {
        let s = state();
        s.active.store(false, Ordering::Release);
        if s.enabled.load(Ordering::Acquire) {
            if let Ok(mut g) = s.active_until.lock() {
                *g = Some(Instant::now() + Duration::from_millis(2500));
            }
        }
        notify_overlay();
        publish_status();
    }
}

/// Tool definitions for native Windows PC Control (`pc_*`).
pub fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "pc_screenshot",
            "description": "Capture the current Windows desktop screen and list visible top-level windows with their bounding rectangles. Requires PC Control to be enabled.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "include_windows": { "type": "boolean", "description": "Include list of visible top-level windows and their (x,y,w,h) rects (default true)" }
                }
            }
        }),
        json!({
            "name": "pc_mouse",
            "description": "Control the Windows mouse cursor: move, click, double_click, right_click, drag, or scroll at screen coordinates (x, y). Requires PC Control to be enabled.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "description": "move, click, double_click, right_click, drag, or scroll" },
                    "x": { "type": "integer", "description": "Target screen X coordinate in pixels" },
                    "y": { "type": "integer", "description": "Target screen Y coordinate in pixels" },
                    "to_x": { "type": "integer", "description": "Destination X coordinate for drag" },
                    "to_y": { "type": "integer", "description": "Destination Y coordinate for drag" },
                    "delta": { "type": "integer", "description": "Scroll wheel delta (positive = up, negative = down; default -360)" }
                },
                "required": ["action"]
            }
        }),
        json!({
            "name": "pc_keyboard",
            "description": "Send keyboard input to the active Windows window: type text, press a single key, or press a hotkey combo (e.g. ctrl+c, alt+tab, win+r). Requires PC Control to be enabled.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "description": "type, key, or hotkey" },
                    "text": { "type": "string", "description": "Text to type when action is 'type'" },
                    "key": { "type": "string", "description": "Key name (e.g. enter, tab, escape, backspace, up, down, f5) or combo string (e.g. ctrl+s)" },
                    "keys": { "type": "string", "description": "Hotkey combo like 'ctrl+shift+p' or 'alt+tab'" }
                },
                "required": ["action"]
            }
        }),
        json!({
            "name": "pc_window",
            "description": "List, focus, minimize, maximize, restore, or close visible Windows desktop windows. Requires PC Control to be enabled.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "description": "list, focus, minimize, maximize, restore, or close" },
                    "title": { "type": "string", "description": "Case-insensitive substring of the window title" },
                    "hwnd": { "type": "integer", "description": "Optional exact window handle (HWND) from pc_window list or pc_screenshot" }
                },
                "required": ["action"]
            }
        }),
        json!({
            "name": "pc_wait",
            "description": "Pause briefly (up to 10000 ms) between PC Control actions while waiting for a window or UI transition.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ms": { "type": "integer", "description": "Milliseconds to wait (50-10000, default 500)" }
                }
            }
        }),
    ]
}

pub fn execute_tool(name: &str, args: &Value, cancel: &AtomicBool) -> ToolResult {
    let err = |msg: String| ToolResult {
        name: name.to_string(),
        output: format!("error: {msg}"),
        is_error: true,
        images: vec![],
    };
    if !is_enabled() {
        return err(
            "PC Control is OFF. Turn on PC Control in ShadowZero's chat bar or Settings to allow screen and input control.".into(),
        );
    }
    match name {
        "pc_screenshot" => run_screenshot(name, args),
        "pc_mouse" => run_mouse(name, args),
        "pc_keyboard" => run_keyboard(name, args),
        "pc_window" => run_window(name, args),
        "pc_wait" => run_wait(name, args, cancel),
        _ => err(format!("unknown pc control tool '{name}'")),
    }
}

fn run_wait(name: &str, args: &Value, cancel: &AtomicBool) -> ToolResult {
    let ms = args
        .get("ms")
        .and_then(|v| v.as_u64().or_else(|| v.as_i64().filter(|&x| x > 0).map(|x| x as u64)))
        .unwrap_or(500)
        .clamp(50, 10_000);
    let _guard = match ActionGuard::begin(&format!("Waiting {ms}ms")) {
        Ok(g) => g,
        Err(e) => {
            return ToolResult {
                name: name.to_string(),
                output: format!("error: {e}"),
                is_error: true,
                images: vec![],
            };
        }
    };
    let start = Instant::now();
    let target = Duration::from_millis(ms);
    while start.elapsed() < target {
        if cancel.load(Ordering::Relaxed) || !is_enabled() {
            return ToolResult {
                name: name.to_string(),
                output: "Cancelled: PC Control stopped during wait.".into(),
                is_error: true,
                images: vec![],
            };
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    ToolResult {
        name: name.to_string(),
        output: format!("Waited {ms}ms."),
        is_error: false,
        images: vec![],
    }
}

#[cfg(not(windows))]
fn ensure_overlay_thread() {}
#[cfg(not(windows))]
fn notify_overlay() {}
#[cfg(not(windows))]
fn run_screenshot(name: &str, _args: &Value) -> ToolResult {
    ToolResult {
        name: name.to_string(),
        output: "error: PC Control is only supported on Windows.".into(),
        is_error: true,
        images: vec![],
    }
}
#[cfg(not(windows))]
fn run_mouse(name: &str, _args: &Value) -> ToolResult {
    run_screenshot(name, _args)
}
#[cfg(not(windows))]
fn run_keyboard(name: &str, _args: &Value) -> ToolResult {
    run_screenshot(name, _args)
}
#[cfg(not(windows))]
fn run_window(name: &str, _args: &Value) -> ToolResult {
    run_screenshot(name, _args)
}

#[cfg(windows)]
mod win {
    use super::*;
    use std::os::windows::process::CommandExt;
    use std::process::{Command, Stdio};

    const WM_DESTROY: u32 = 0x0002;
    const WM_PAINT: u32 = 0x000F;
    const WM_CLOSE: u32 = 0x0010;
    const WM_MOUSEACTIVATE: u32 = 0x0021;
    const WM_TIMER: u32 = 0x0113;
    const WM_LBUTTONDOWN: u32 = 0x0201;
    const WM_HOTKEY: u32 = 0x0312;
    const WM_APP_SYNC: u32 = 0x8001;

    const MA_NOACTIVATE: isize = 3;
    const HOTKEY_ID_STOP: i32 = 1;
    const TIMER_ID_PULSE: usize = 1;
    const MOD_ALT: u32 = 0x0001;
    const MOD_CONTROL: u32 = 0x0002;
    const MOD_NOREPEAT: u32 = 0x4000;
    const VK_ESCAPE: u32 = 0x1B;

    const WS_POPUP: u32 = 0x8000_0000;
    const WS_EX_TOPMOST: u32 = 0x0000_0008;
    const WS_EX_TRANSPARENT: u32 = 0x0000_0020;
    const WS_EX_TOOLWINDOW: u32 = 0x0000_0080;
    const WS_EX_LAYERED: u32 = 0x0008_0000;
    const WS_EX_NOACTIVATE: u32 = 0x0800_0000;

    const LWA_ALPHA: u32 = 0x0000_0002;
    const WDA_EXCLUDEFROMCAPTURE: u32 = 0x0000_0011;
    const SW_HIDE: i32 = 0;
    const SW_SHOWNOACTIVATE: i32 = 4;
    const SW_MINIMIZE: i32 = 6;
    const SW_MAXIMIZE: i32 = 3;
    const SW_RESTORE: i32 = 9;
    const SM_CXSCREEN: i32 = 0;
    const SM_CYSCREEN: i32 = 1;

    const INPUT_MOUSE: u32 = 0;
    const INPUT_KEYBOARD: u32 = 1;
    const MOUSEEVENTF_LEFTDOWN: u32 = 0x0002;
    const MOUSEEVENTF_LEFTUP: u32 = 0x0004;
    const MOUSEEVENTF_RIGHTDOWN: u32 = 0x0008;
    const MOUSEEVENTF_RIGHTUP: u32 = 0x0010;
    const MOUSEEVENTF_WHEEL: u32 = 0x0800;
    const KEYEVENTF_EXTENDEDKEY: u32 = 0x0001;
    const KEYEVENTF_KEYUP: u32 = 0x0002;
    const KEYEVENTF_UNICODE: u32 = 0x0004;

    const BANNER_W: i32 = 520;
    const BANNER_H: i32 = 38;
    const STOP_BTN_W: i32 = 152;
    const BORDER_THICK: i32 = 4;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Point {
        x: i32,
        y: i32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Rect {
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    }

    #[repr(C)]
    struct WndClassExW {
        cb_size: u32,
        style: u32,
        lpfn_wnd_proc: unsafe extern "system" fn(isize, u32, usize, isize) -> isize,
        cb_cls_extra: i32,
        cb_wnd_extra: i32,
        h_instance: isize,
        h_icon: isize,
        h_cursor: isize,
        hbr_background: isize,
        lpsz_menu_name: *const u16,
        lpsz_class_name: *const u16,
        h_icon_sm: isize,
    }

    #[repr(C)]
    struct Msg {
        hwnd: isize,
        message: u32,
        w_param: usize,
        l_param: isize,
        time: u32,
        pt: Point,
        l_private: u32,
    }

    #[repr(C)]
    struct PaintStruct {
        hdc: isize,
        f_erase: i32,
        rc_paint: Rect,
        f_restore: i32,
        f_inc_update: i32,
        rgb_reserved: [u8; 32],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub(super) struct MouseInput {
        dx: i32,
        dy: i32,
        mouse_data: u32,
        dw_flags: u32,
        time: u32,
        dw_extra_info: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub(super) struct KeybdInput {
        w_vk: u16,
        w_scan: u16,
        dw_flags: u32,
        time: u32,
        dw_extra_info: usize,
    }

    #[repr(C)]
    pub(super) union InputUnion {
        mi: MouseInput,
        ki: KeybdInput,
        _pad: [u8; 32],
    }

    #[repr(C)]
    pub(super) struct Input {
        r#type: u32,
        u: InputUnion,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetModuleHandleW(lp_module_name: *const u16) -> isize;
    }

    #[link(name = "user32")]
    extern "system" {
        fn RegisterClassExW(lpwcx: *const WndClassExW) -> u16;
        fn CreateWindowExW(
            dw_ex_style: u32,
            lp_class_name: *const u16,
            lp_window_name: *const u16,
            dw_style: u32,
            x: i32,
            y: i32,
            n_width: i32,
            n_height: i32,
            h_wnd_parent: isize,
            h_menu: isize,
            h_instance: isize,
            lp_param: *const u8,
        ) -> isize;
        fn DefWindowProcW(hwnd: isize, msg: u32, w: usize, l: isize) -> isize;
        fn ShowWindow(hwnd: isize, n_cmd_show: i32) -> i32;
        fn InvalidateRect(hwnd: isize, lp_rect: *const Rect, b_erase: i32) -> i32;
        fn BeginPaint(hwnd: isize, lp_paint: *mut PaintStruct) -> isize;
        fn EndPaint(hwnd: isize, lp_paint: *const PaintStruct) -> i32;
        fn PostMessageW(hwnd: isize, msg: u32, w: usize, l: isize) -> i32;
        fn GetMessageW(lp_msg: *mut Msg, hwnd: isize, min: u32, max: u32) -> i32;
        fn TranslateMessage(lp_msg: *const Msg) -> i32;
        fn DispatchMessageW(lp_msg: *const Msg) -> isize;
        fn SetTimer(hwnd: isize, id: usize, elapse: u32, func: isize) -> usize;
        fn RegisterHotKey(hwnd: isize, id: i32, fs_modifiers: u32, vk: u32) -> i32;
        fn UnregisterHotKey(hwnd: isize, id: i32) -> i32;
        fn SetLayeredWindowAttributes(hwnd: isize, cr_key: u32, b_alpha: u8, dw_flags: u32) -> i32;
        fn SetWindowDisplayAffinity(hwnd: isize, dw_affinity: u32) -> i32;
        fn GetSystemMetrics(n_index: i32) -> i32;
        fn GetCursorPos(lp_point: *mut Point) -> i32;
        fn SetCursorPos(x: i32, y: i32) -> i32;
        fn SendInput(c_inputs: u32, p_inputs: *const Input, cb_size: i32) -> u32;
        fn FillRect(hdc: isize, lprc: *const Rect, hbr: isize) -> i32;
        fn DrawTextW(hdc: isize, lpch: *const u16, cch: i32, lprc: *mut Rect, format: u32) -> i32;
        fn LoadCursorW(h_instance: isize, lp_cursor_name: *const u16) -> isize;
        fn EnumWindows(
            lp_enum_func: unsafe extern "system" fn(isize, isize) -> i32,
            l_param: isize,
        ) -> i32;
        fn IsWindowVisible(hwnd: isize) -> i32;
        fn GetWindowTextW(hwnd: isize, lp_string: *mut u16, n_max_count: i32) -> i32;
        fn GetWindowRect(hwnd: isize, lp_rect: *mut Rect) -> i32;
        fn GetForegroundWindow() -> isize;
        fn SetForegroundWindow(hwnd: isize) -> i32;
    }

    #[link(name = "gdi32")]
    extern "system" {
        fn CreateSolidBrush(color: u32) -> isize;
        fn CreatePen(i_style: i32, c_width: i32, color: u32) -> isize;
        fn SelectObject(hdc: isize, h: isize) -> isize;
        fn DeleteObject(ho: isize) -> i32;
        fn RoundRect(hdc: isize, l: i32, t: i32, r: i32, b: i32, w: i32, h: i32) -> i32;
        fn Ellipse(hdc: isize, l: i32, t: i32, r: i32, b: i32) -> i32;
        fn SetBkMode(hdc: isize, mode: i32) -> i32;
        fn SetTextColor(hdc: isize, color: u32) -> u32;
        fn CreateFontW(
            c_height: i32,
            c_width: i32,
            c_escapement: i32,
            c_orientation: i32,
            c_weight: i32,
            b_italic: u32,
            b_underline: u32,
            b_strike_out: u32,
            i_char_set: u32,
            i_out_precision: u32,
            i_clip_precision: u32,
            i_quality: u32,
            i_pitch_and_family: u32,
            psz_face_name: *const u16,
        ) -> isize;
    }

    static BORDER_HWNDS: Mutex<[isize; 4]> = Mutex::new([0; 4]);

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    fn rgb(r: u8, g: u8, b: u8) -> u32 {
        (r as u32) | ((g as u32) << 8) | ((b as u32) << 16)
    }

    fn sync_overlay_windows(banner: isize) {
        let enabled = is_enabled();
        let cmd = if enabled { SW_SHOWNOACTIVATE } else { SW_HIDE };
        unsafe {
            ShowWindow(banner, cmd);
            if enabled {
                InvalidateRect(banner, std::ptr::null(), 1);
            }
            if let Ok(borders) = BORDER_HWNDS.lock() {
                for &bh in borders.iter() {
                    if bh != 0 {
                        ShowWindow(bh, cmd);
                        if enabled {
                            InvalidateRect(bh, std::ptr::null(), 1);
                        }
                    }
                }
            }
        }
    }

    unsafe extern "system" fn border_wnd_proc(
        hwnd: isize,
        msg: u32,
        w: usize,
        l: isize,
    ) -> isize {
        if msg == WM_PAINT {
            let mut ps = std::mem::zeroed::<PaintStruct>();
            let hdc = BeginPaint(hwnd, &mut ps);
            let col = if is_active() {
                rgb(244, 63, 94) // bright coral/red during active PC control
            } else {
                rgb(99, 102, 241) // indigo while armed/idle
            };
            let brush = CreateSolidBrush(col);
            FillRect(hdc, &ps.rc_paint, brush);
            DeleteObject(brush);
            EndPaint(hwnd, &ps);
            return 0;
        }
        DefWindowProcW(hwnd, msg, w, l)
    }

    unsafe extern "system" fn banner_wnd_proc(
        hwnd: isize,
        msg: u32,
        w: usize,
        l: isize,
    ) -> isize {
        match msg {
            WM_MOUSEACTIVATE => return MA_NOACTIVATE,
            WM_APP_SYNC => {
                sync_overlay_windows(hwnd);
                return 0;
            }
            WM_TIMER => {
                if is_enabled() {
                    InvalidateRect(hwnd, std::ptr::null(), 0);
                    if let Ok(borders) = BORDER_HWNDS.lock() {
                        for &bh in borders.iter() {
                            if bh != 0 {
                                InvalidateRect(bh, std::ptr::null(), 0);
                            }
                        }
                    }
                }
                return 0;
            }
            WM_HOTKEY => {
                if w as i32 == HOTKEY_ID_STOP && is_enabled() {
                    set_enabled(false);
                }
                return 0;
            }
            WM_LBUTTONDOWN => {
                let x = (l & 0xFFFF) as i16 as i32;
                if x >= BANNER_W - STOP_BTN_W - 10 {
                    set_enabled(false);
                }
                return 0;
            }
            WM_PAINT => {
                let mut ps = std::mem::zeroed::<PaintStruct>();
                let hdc = BeginPaint(hwnd, &mut ps);
                let active = is_active();
                let bg = if active { rgb(42, 15, 23) } else { rgb(22, 27, 38) };
                let border = if active { rgb(244, 63, 94) } else { rgb(99, 102, 241) };
                let dot = if active { rgb(251, 113, 133) } else { rgb(129, 140, 248) };

                let bg_brush = CreateSolidBrush(bg);
                let pen = CreatePen(0, 2, border);
                let old_brush = SelectObject(hdc, bg_brush);
                let old_pen = SelectObject(hdc, pen);
                RoundRect(hdc, 0, 0, BANNER_W, BANNER_H, 14, 14);
                SelectObject(hdc, old_brush);
                SelectObject(hdc, old_pen);
                DeleteObject(bg_brush);
                DeleteObject(pen);

                // Status dot
                let dot_brush = CreateSolidBrush(dot);
                let dot_pen = CreatePen(0, 1, dot);
                let ob2 = SelectObject(hdc, dot_brush);
                let op2 = SelectObject(hdc, dot_pen);
                Ellipse(hdc, 14, 14, 24, 24);
                SelectObject(hdc, ob2);
                SelectObject(hdc, op2);
                DeleteObject(dot_brush);
                DeleteObject(dot_pen);

                // Stop button pill on the right
                let btn_l = BANNER_W - STOP_BTN_W - 8;
                let btn_r = BANNER_W - 8;
                let btn_brush = CreateSolidBrush(rgb(185, 28, 28));
                let btn_pen = CreatePen(0, 1, rgb(248, 113, 113));
                let ob3 = SelectObject(hdc, btn_brush);
                let op3 = SelectObject(hdc, btn_pen);
                RoundRect(hdc, btn_l, 6, btn_r, BANNER_H - 6, 10, 10);
                SelectObject(hdc, ob3);
                SelectObject(hdc, op3);
                DeleteObject(btn_brush);
                DeleteObject(btn_pen);

                let face = wide("Segoe UI");
                let font = CreateFontW(15, 0, 0, 0, 600, 0, 0, 0, 1, 0, 0, 5, 0, face.as_ptr());
                let old_font = SelectObject(hdc, font);
                SetBkMode(hdc, 1); // TRANSPARENT
                SetTextColor(hdc, rgb(248, 250, 252));

                let action = state()
                    .last_action
                    .lock()
                    .map(|g| g.clone())
                    .unwrap_or_default();
                let label = if active {
                    if action.is_empty() {
                        "ShadowZero is controlling your PC".to_string()
                    } else {
                        format!("ShadowZero controlling PC: {action}")
                    }
                } else {
                    "ShadowZero PC Control ON — Ready".to_string()
                };
                let label_w = wide(&label);
                let mut rc_txt = Rect {
                    left: 32,
                    top: 0,
                    right: btn_l - 8,
                    bottom: BANNER_H,
                };
                // DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS = 0x00008024
                DrawTextW(hdc, label_w.as_ptr(), -1, &mut rc_txt, 0x0000_8024);

                let stop_w = wide("Stop (Ctrl+Alt+Esc)");
                let mut rc_btn = Rect {
                    left: btn_l,
                    top: 6,
                    right: btn_r,
                    bottom: BANNER_H - 6,
                };
                // DT_CENTER | DT_VCENTER | DT_SINGLELINE = 0x25
                DrawTextW(hdc, stop_w.as_ptr(), -1, &mut rc_btn, 0x25);

                SelectObject(hdc, old_font);
                DeleteObject(font);
                EndPaint(hwnd, &ps);
                return 0;
            }
            WM_DESTROY => {
                UnregisterHotKey(hwnd, HOTKEY_ID_STOP);
                return 0;
            }
            _ => {}
        }
        DefWindowProcW(hwnd, msg, w, l)
    }

    pub(super) fn ensure_overlay_thread() {
        let s = state();
        if s.overlay_started.swap(true, Ordering::AcqRel) {
            return;
        }
        std::thread::spawn(|| unsafe {
            let hinst = GetModuleHandleW(std::ptr::null());
            let arrow = LoadCursorW(0, 32512usize as *const u16); // IDC_ARROW

            let banner_cls = wide("ShadowZeroPcBanner");
            let wc_banner = WndClassExW {
                cb_size: std::mem::size_of::<WndClassExW>() as u32,
                style: 0,
                lpfn_wnd_proc: banner_wnd_proc,
                cb_cls_extra: 0,
                cb_wnd_extra: 0,
                h_instance: hinst,
                h_icon: 0,
                h_cursor: arrow,
                hbr_background: 0,
                lpsz_menu_name: std::ptr::null(),
                lpsz_class_name: banner_cls.as_ptr(),
                h_icon_sm: 0,
            };
            RegisterClassExW(&wc_banner);

            let border_cls = wide("ShadowZeroPcBorder");
            let wc_border = WndClassExW {
                cb_size: std::mem::size_of::<WndClassExW>() as u32,
                style: 0,
                lpfn_wnd_proc: border_wnd_proc,
                cb_cls_extra: 0,
                cb_wnd_extra: 0,
                h_instance: hinst,
                h_icon: 0,
                h_cursor: arrow,
                hbr_background: 0,
                lpsz_menu_name: std::ptr::null(),
                lpsz_class_name: border_cls.as_ptr(),
                h_icon_sm: 0,
            };
            RegisterClassExW(&wc_border);

            let sw = GetSystemMetrics(SM_CXSCREEN).max(800);
            let sh = GetSystemMetrics(SM_CYSCREEN).max(600);
            let bx = ((sw - BANNER_W) / 2).max(0);
            let title = wide("ShadowZero PC Control");

            let banner = CreateWindowExW(
                WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_LAYERED,
                banner_cls.as_ptr(),
                title.as_ptr(),
                WS_POPUP,
                bx,
                10,
                BANNER_W,
                BANNER_H,
                0,
                0,
                hinst,
                std::ptr::null(),
            );
            if banner == 0 {
                return;
            }
            SetLayeredWindowAttributes(banner, 0, 240, LWA_ALPHA);
            SetWindowDisplayAffinity(banner, WDA_EXCLUDEFROMCAPTURE);
            RegisterHotKey(
                banner,
                HOTKEY_ID_STOP,
                MOD_CONTROL | MOD_ALT | MOD_NOREPEAT,
                VK_ESCAPE,
            );
            SetTimer(banner, TIMER_ID_PULSE, 500, 0);

            // 4 click-through screen border glow strips (top, bottom, left, right)
            let rects = [
                (0, 0, sw, BORDER_THICK),
                (0, sh - BORDER_THICK, sw, BORDER_THICK),
                (0, 0, BORDER_THICK, sh),
                (sw - BORDER_THICK, 0, BORDER_THICK, sh),
            ];
            let mut bhs = [0isize; 4];
            for (idx, &(rx, ry, rw, rh)) in rects.iter().enumerate() {
                let bh = CreateWindowExW(
                    WS_EX_TOPMOST
                        | WS_EX_TOOLWINDOW
                        | WS_EX_NOACTIVATE
                        | WS_EX_TRANSPARENT
                        | WS_EX_LAYERED,
                    border_cls.as_ptr(),
                    title.as_ptr(),
                    WS_POPUP,
                    rx,
                    ry,
                    rw,
                    rh,
                    0,
                    0,
                    hinst,
                    std::ptr::null(),
                );
                if bh != 0 {
                    SetLayeredWindowAttributes(bh, 0, 210, LWA_ALPHA);
                    SetWindowDisplayAffinity(bh, WDA_EXCLUDEFROMCAPTURE);
                }
                bhs[idx] = bh;
            }
            if let Ok(mut g) = BORDER_HWNDS.lock() {
                *g = bhs;
            }
            state().banner_hwnd.store(banner, Ordering::Release);
            sync_overlay_windows(banner);

            let mut msg = std::mem::zeroed::<Msg>();
            while GetMessageW(&mut msg, 0, 0, 0) > 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        });
    }

    pub(super) fn notify_overlay() {
        let hwnd = state().banner_hwnd.load(Ordering::Acquire);
        if hwnd != 0 {
            unsafe {
                PostMessageW(hwnd, WM_APP_SYNC, 0, 0);
            }
        }
    }

    #[derive(Clone)]
    struct WinInfo {
        hwnd: isize,
        title: String,
        rect: Rect,
    }

    unsafe extern "system" fn enum_windows_cb(hwnd: isize, l_param: isize) -> i32 {
        if IsWindowVisible(hwnd) == 0 {
            return 1;
        }
        let mut buf = [0u16; 256];
        let len = GetWindowTextW(hwnd, buf.as_mut_ptr(), buf.len() as i32);
        if len <= 0 {
            return 1;
        }
        let title = String::from_utf16_lossy(&buf[..len as usize]);
        if title == "ShadowZero PC Control" || title == "Program Manager" {
            return 1;
        }
        let mut rect = Rect::default();
        GetWindowRect(hwnd, &mut rect);
        if rect.right - rect.left < 40 || rect.bottom - rect.top < 24 {
            return 1;
        }
        let out = &mut *(l_param as *mut Vec<WinInfo>);
        out.push(WinInfo { hwnd, title, rect });
        1
    }

    fn list_visible_windows() -> Vec<WinInfo> {
        let mut out: Vec<WinInfo> = Vec::new();
        unsafe {
            EnumWindows(enum_windows_cb, (&mut out as *mut Vec<WinInfo>) as isize);
        }
        out
    }

    fn foreground_title() -> String {
        unsafe {
            let fg = GetForegroundWindow();
            if fg == 0 {
                return String::new();
            }
            let mut buf = [0u16; 256];
            let len = GetWindowTextW(fg, buf.as_mut_ptr(), buf.len() as i32);
            if len <= 0 {
                return String::new();
            }
            String::from_utf16_lossy(&buf[..len as usize])
        }
    }

    pub(super) fn run_screenshot(name: &str, args: &Value) -> ToolResult {
        let _guard = match ActionGuard::begin("Capturing screen") {
            Ok(g) => g,
            Err(e) => {
                return ToolResult {
                    name: name.to_string(),
                    output: format!("error: {e}"),
                    is_error: true,
                    images: vec![],
                };
            }
        };
        let include_windows = args
            .get("include_windows")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let (sw, sh, cur) = unsafe {
            let mut pt = Point::default();
            GetCursorPos(&mut pt);
            (GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN), pt)
        };
        let fg = foreground_title();
        let ps = r#"
Add-Type -AssemblyName System.Windows.Forms,System.Drawing
$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap($b.Width, $b.Height)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
$g.Dispose()
$maxW = 1280
if ($bmp.Width -gt $maxW) {
  $newH = [int]([Math]::Round($bmp.Height * ($maxW / [double]$bmp.Width)))
  $scaled = New-Object System.Drawing.Bitmap($maxW, $newH)
  $sg = [System.Drawing.Graphics]::FromImage($scaled)
  $sg.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
  $sg.DrawImage($bmp, 0, 0, $maxW, $newH)
  $sg.Dispose()
  $bmp.Dispose()
  $bmp = $scaled
}
$ms = New-Object System.IO.MemoryStream
$bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()
[Convert]::ToBase64String($ms.ToArray())
$ms.Dispose()
"#;
        let out = Command::new("powershell")
            .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .creation_flags(0x0800_0000)
            .output();
        let mut images = Vec::new();
        let mut capture_note = String::new();
        match out {
            Ok(o) if o.status.success() => {
                let b64 = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if !b64.is_empty() {
                    images.push(format!("data:image/png;base64,{b64}"));
                }
            }
            Ok(o) => {
                capture_note = format!(
                    " (image capture warning: {})",
                    String::from_utf8_lossy(&o.stderr).trim()
                );
            }
            Err(e) => {
                capture_note = format!(" (image capture error: {e})");
            }
        }

        let mut text = format!(
            "Screen: {sw}x{sh} | Cursor: ({}, {}) | Foreground: {}{capture_note}",
            cur.x,
            cur.y,
            if fg.is_empty() { "(none)" } else { &fg }
        );
        if include_windows {
            let wins = list_visible_windows();
            if !wins.is_empty() {
                text.push_str("\nVisible windows:");
                for w in wins.iter().take(25) {
                    text.push_str(&format!(
                        "\n- [hwnd={}] {} ({}, {}, {}x{})",
                        w.hwnd,
                        w.title,
                        w.rect.left,
                        w.rect.top,
                        w.rect.right - w.rect.left,
                        w.rect.bottom - w.rect.top
                    ));
                }
            }
        }
        ToolResult {
            name: name.to_string(),
            output: text,
            is_error: false,
            images,
        }
    }

    fn send_mouse_flags(flags: u32, mouse_data: u32) {
        let inp = Input {
            r#type: INPUT_MOUSE,
            u: InputUnion {
                mi: MouseInput {
                    dx: 0,
                    dy: 0,
                    mouse_data,
                    dw_flags: flags,
                    time: 0,
                    dw_extra_info: 0,
                },
            },
        };
        unsafe {
            SendInput(1, &inp, std::mem::size_of::<Input>() as i32);
        }
    }

    pub(super) fn run_mouse(name: &str, args: &Value) -> ToolResult {
        let action = args
            .get("action")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_lowercase();
        let x = args.get("x").and_then(Value::as_i64).map(|v| v as i32);
        let y = args.get("y").and_then(Value::as_i64).map(|v| v as i32);
        let desc = match (x, y) {
            (Some(px), Some(py)) => format!("{action} ({px}, {py})"),
            _ => format!("mouse {action}"),
        };
        let _guard = match ActionGuard::begin(&desc) {
            Ok(g) => g,
            Err(e) => {
                return ToolResult {
                    name: name.to_string(),
                    output: format!("error: {e}"),
                    is_error: true,
                    images: vec![],
                };
            }
        };

        if let (Some(px), Some(py)) = (x, y) {
            unsafe {
                SetCursorPos(px, py);
            }
            std::thread::sleep(Duration::from_millis(20));
        }

        match action.as_str() {
            "move" => {}
            "click" => {
                send_mouse_flags(MOUSEEVENTF_LEFTDOWN, 0);
                std::thread::sleep(Duration::from_millis(20));
                send_mouse_flags(MOUSEEVENTF_LEFTUP, 0);
            }
            "double_click" => {
                for _ in 0..2 {
                    send_mouse_flags(MOUSEEVENTF_LEFTDOWN, 0);
                    std::thread::sleep(Duration::from_millis(18));
                    send_mouse_flags(MOUSEEVENTF_LEFTUP, 0);
                    std::thread::sleep(Duration::from_millis(45));
                }
            }
            "right_click" => {
                send_mouse_flags(MOUSEEVENTF_RIGHTDOWN, 0);
                std::thread::sleep(Duration::from_millis(20));
                send_mouse_flags(MOUSEEVENTF_RIGHTUP, 0);
            }
            "drag" => {
                let (Some(tx), Some(ty)) = (
                    args.get("to_x").and_then(Value::as_i64).map(|v| v as i32),
                    args.get("to_y").and_then(Value::as_i64).map(|v| v as i32),
                ) else {
                    return ToolResult {
                        name: name.to_string(),
                        output: "error: drag requires to_x and to_y".into(),
                        is_error: true,
                        images: vec![],
                    };
                };
                send_mouse_flags(MOUSEEVENTF_LEFTDOWN, 0);
                std::thread::sleep(Duration::from_millis(40));
                unsafe {
                    SetCursorPos(tx, ty);
                }
                std::thread::sleep(Duration::from_millis(40));
                send_mouse_flags(MOUSEEVENTF_LEFTUP, 0);
            }
            "scroll" => {
                let delta = args
                    .get("delta")
                    .and_then(Value::as_i64)
                    .unwrap_or(-360) as i32;
                send_mouse_flags(MOUSEEVENTF_WHEEL, delta as u32);
            }
            _ => {
                return ToolResult {
                    name: name.to_string(),
                    output: format!(
                        "error: unknown mouse action '{action}' (use move, click, double_click, right_click, drag, or scroll)"
                    ),
                    is_error: true,
                    images: vec![],
                };
            }
        }
        ToolResult {
            name: name.to_string(),
            output: format!("Performed mouse {desc}."),
            is_error: false,
            images: vec![],
        }
    }

    fn key_vk(name: &str) -> Option<(u16, bool)> {
        let k = name.trim().to_lowercase();
        if k.len() == 1 {
            let b = k.as_bytes()[0];
            if b.is_ascii_alphanumeric() {
                return Some((b.to_ascii_uppercase() as u16, false));
            }
        }
        let hit = match k.as_str() {
            "ctrl" | "control" => (0x11, false),
            "alt" | "menu" => (0x12, false),
            "shift" => (0x10, false),
            "win" | "super" | "meta" | "lwin" => (0x5B, true),
            "enter" | "return" => (0x0D, false),
            "tab" => (0x09, false),
            "space" => (0x20, false),
            "esc" | "escape" => (0x1B, false),
            "backspace" | "bksp" => (0x08, false),
            "delete" | "del" => (0x2E, true),
            "insert" | "ins" => (0x2D, true),
            "home" => (0x24, true),
            "end" => (0x23, true),
            "pageup" | "pgup" => (0x21, true),
            "pagedown" | "pgdn" => (0x22, true),
            "left" => (0x25, true),
            "up" => (0x26, true),
            "right" => (0x27, true),
            "down" => (0x28, true),
            "f1" => (0x70, false),
            "f2" => (0x71, false),
            "f3" => (0x72, false),
            "f4" => (0x73, false),
            "f5" => (0x74, false),
            "f6" => (0x75, false),
            "f7" => (0x76, false),
            "f8" => (0x77, false),
            "f9" => (0x78, false),
            "f10" => (0x79, false),
            "f11" => (0x7A, false),
            "f12" => (0x7B, false),
            _ => return None,
        };
        Some(hit)
    }

    fn send_vk(vk: u16, extended: bool, key_up: bool) {
        let mut flags = 0u32;
        if extended {
            flags |= KEYEVENTF_EXTENDEDKEY;
        }
        if key_up {
            flags |= KEYEVENTF_KEYUP;
        }
        let inp = Input {
            r#type: INPUT_KEYBOARD,
            u: InputUnion {
                ki: KeybdInput {
                    w_vk: vk,
                    w_scan: 0,
                    dw_flags: flags,
                    time: 0,
                    dw_extra_info: 0,
                },
            },
        };
        unsafe {
            SendInput(1, &inp, std::mem::size_of::<Input>() as i32);
        }
    }

    fn send_unicode_text(text: &str) {
        for unit in text.encode_utf16() {
            if unit == b'\n' as u16 {
                send_vk(0x0D, false, false);
                send_vk(0x0D, false, true);
                continue;
            }
            if unit == b'\r' as u16 {
                continue;
            }
            let down = Input {
                r#type: INPUT_KEYBOARD,
                u: InputUnion {
                    ki: KeybdInput {
                        w_vk: 0,
                        w_scan: unit,
                        dw_flags: KEYEVENTF_UNICODE,
                        time: 0,
                        dw_extra_info: 0,
                    },
                },
            };
            let up = Input {
                r#type: INPUT_KEYBOARD,
                u: InputUnion {
                    ki: KeybdInput {
                        w_vk: 0,
                        w_scan: unit,
                        dw_flags: KEYEVENTF_UNICODE | KEYEVENTF_KEYUP,
                        time: 0,
                        dw_extra_info: 0,
                    },
                },
            };
            unsafe {
                let pair = [down, up];
                SendInput(2, pair.as_ptr(), std::mem::size_of::<Input>() as i32);
            }
        }
    }

    pub(super) fn run_keyboard(name: &str, args: &Value) -> ToolResult {
        let action = args
            .get("action")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_lowercase();
        match action.as_str() {
            "type" => {
                let text = args.get("text").and_then(Value::as_str).unwrap_or("");
                if text.is_empty() {
                    return ToolResult {
                        name: name.to_string(),
                        output: "error: 'type' requires non-empty 'text'".into(),
                        is_error: true,
                        images: vec![],
                    };
                }
                let _guard = match ActionGuard::begin(&format!("Typing {} chars", text.chars().count())) {
                    Ok(g) => g,
                    Err(e) => {
                        return ToolResult {
                            name: name.to_string(),
                            output: format!("error: {e}"),
                            is_error: true,
                            images: vec![],
                        };
                    }
                };
                send_unicode_text(text);
                ToolResult {
                    name: name.to_string(),
                    output: format!("Typed {} character(s).", text.chars().count()),
                    is_error: false,
                    images: vec![],
                }
            }
            "key" | "hotkey" => {
                let raw = args
                    .get("keys")
                    .or_else(|| args.get("key"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim();
                if raw.is_empty() {
                    return ToolResult {
                        name: name.to_string(),
                        output: "error: missing 'key' or 'keys'".into(),
                        is_error: true,
                        images: vec![],
                    };
                }
                let parts: Vec<&str> = raw.split('+').map(str::trim).filter(|s| !s.is_empty()).collect();
                let mut vks = Vec::with_capacity(parts.len());
                for p in &parts {
                    let Some(pair) = key_vk(p) else {
                        return ToolResult {
                            name: name.to_string(),
                            output: format!("error: unrecognized key '{p}'"),
                            is_error: true,
                            images: vec![],
                        };
                    };
                    vks.push(pair);
                }
                // Reserve Ctrl+Alt+Esc and Ctrl+Alt+Del for user emergency stop / OS security.
                let has_ctrl = vks.iter().any(|&(vk, _)| vk == 0x11);
                let has_alt = vks.iter().any(|&(vk, _)| vk == 0x12);
                let has_blocked = vks.iter().any(|&(vk, _)| vk == 0x1B || vk == 0x2E);
                if has_ctrl && has_alt && has_blocked {
                    return ToolResult {
                        name: name.to_string(),
                        output: "error: Ctrl+Alt+Esc and Ctrl+Alt+Del are reserved for the user.".into(),
                        is_error: true,
                        images: vec![],
                    };
                }
                let _guard = match ActionGuard::begin(&format!("Pressing {raw}")) {
                    Ok(g) => g,
                    Err(e) => {
                        return ToolResult {
                            name: name.to_string(),
                            output: format!("error: {e}"),
                            is_error: true,
                            images: vec![],
                        };
                    }
                };
                for &(vk, ext) in &vks {
                    send_vk(vk, ext, false);
                    std::thread::sleep(Duration::from_millis(12));
                }
                for &(vk, ext) in vks.iter().rev() {
                    send_vk(vk, ext, true);
                    std::thread::sleep(Duration::from_millis(12));
                }
                ToolResult {
                    name: name.to_string(),
                    output: format!("Pressed {raw}."),
                    is_error: false,
                    images: vec![],
                }
            }
            _ => ToolResult {
                name: name.to_string(),
                output: format!("error: unknown keyboard action '{action}' (use type, key, or hotkey)"),
                is_error: true,
                images: vec![],
            },
        }
    }

    pub(super) fn run_window(name: &str, args: &Value) -> ToolResult {
        let action = args
            .get("action")
            .and_then(Value::as_str)
            .unwrap_or("list")
            .trim()
            .to_lowercase();
        let _guard = match ActionGuard::begin(&format!("Window {action}")) {
            Ok(g) => g,
            Err(e) => {
                return ToolResult {
                    name: name.to_string(),
                    output: format!("error: {e}"),
                    is_error: true,
                    images: vec![],
                };
            }
        };
        let wins = list_visible_windows();
        if action == "list" {
            let fg = foreground_title();
            let mut out = format!("Foreground: {}\nVisible windows ({}):", if fg.is_empty() { "(none)" } else { &fg }, wins.len());
            for w in wins.iter().take(40) {
                out.push_str(&format!(
                    "\n- [hwnd={}] {} ({}, {}, {}x{})",
                    w.hwnd,
                    w.title,
                    w.rect.left,
                    w.rect.top,
                    w.rect.right - w.rect.left,
                    w.rect.bottom - w.rect.top
                ));
            }
            return ToolResult {
                name: name.to_string(),
                output: out,
                is_error: false,
                images: vec![],
            };
        }
        let target_hwnd = args.get("hwnd").and_then(Value::as_i64).map(|v| v as isize);
        let title_q = args
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_lowercase();
        let found = wins.iter().find(|w| {
            target_hwnd.is_some_and(|h| w.hwnd == h)
                || (!title_q.is_empty() && w.title.to_lowercase().contains(&title_q))
        });
        let Some(w) = found else {
            return ToolResult {
                name: name.to_string(),
                output: "error: no matching visible window found".into(),
                is_error: true,
                images: vec![],
            };
        };
        unsafe {
            match action.as_str() {
                "focus" => {
                    ShowWindow(w.hwnd, SW_RESTORE);
                    SetForegroundWindow(w.hwnd);
                }
                "minimize" => {
                    ShowWindow(w.hwnd, SW_MINIMIZE);
                }
                "maximize" => {
                    ShowWindow(w.hwnd, SW_MAXIMIZE);
                }
                "restore" => {
                    ShowWindow(w.hwnd, SW_RESTORE);
                }
                "close" => {
                    PostMessageW(w.hwnd, WM_CLOSE, 0, 0);
                }
                _ => {
                    return ToolResult {
                        name: name.to_string(),
                        output: format!("error: unknown window action '{action}'"),
                        is_error: true,
                        images: vec![],
                    };
                }
            }
        }
        ToolResult {
            name: name.to_string(),
            output: format!("Window '{}' ({}): {action} ok.", w.title, w.hwnd),
            is_error: false,
            images: vec![],
        }
    }
}

#[cfg(windows)]
use win::{ensure_overlay_thread, notify_overlay, run_keyboard, run_mouse, run_screenshot, run_window};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pc_control_is_off_by_default_and_refuses_tools() {
        assert!(!is_enabled());
        let never = AtomicBool::new(false);
        let res = execute_tool("pc_mouse", &json!({"action": "click", "x": 10, "y": 10}), &never);
        assert!(res.is_error);
        assert!(res.output.contains("PC Control is OFF"));
    }

    #[cfg(all(windows, target_pointer_width = "64"))]
    #[test]
    fn win32_input_struct_matches_x64_abi() {
        assert_eq!(std::mem::size_of::<win::Input>(), 40);
    }
}
