use crate::api::{self, ChatMessage, ChatRequest, StreamEvent};
use crate::theme;
use egui::{Align, Color32, CornerRadius, Layout, Margin, RichText, ScrollArea, Vec2};
use std::sync::mpsc;
use std::time::Instant;

#[derive(PartialEq, Clone, Copy)]
enum Role {
    User,
    Assistant,
}

struct DisplayMessage {
    role: Role,
    content: String,
}

#[derive(Clone, Copy, PartialEq)]
enum PaneSide {
    Left,
    Right,
}

struct PaneState {
    label: String,
    endpoint: String,
    model: String,
    available_models: Vec<String>,
    messages: Vec<DisplayMessage>,
    input: String,
    streaming: bool,
    current_tps: Option<f64>,
    last_stats: Option<String>,
    stream_rx: Option<mpsc::Receiver<StreamEvent>>,
    model_rx: Option<mpsc::Receiver<Vec<String>>>,
    scroll_to_bottom: bool,
    // Race tracking per-pane
    race_token_count: u32,
    race_finish_time: Option<Instant>,
    race_elapsed_secs: Option<f64>,
}

impl PaneState {
    fn new(label: &str, endpoint: &str) -> Self {
        Self {
            label: label.to_string(),
            endpoint: endpoint.to_string(),
            model: String::new(),
            available_models: vec![],
            messages: vec![],
            input: String::new(),
            streaming: false,
            current_tps: None,
            last_stats: None,
            stream_rx: None,
            model_rx: None,
            scroll_to_bottom: false,
            race_token_count: 0,
            race_finish_time: None,
            race_elapsed_secs: None,
        }
    }
}

struct RaceResult {
    left_tokens: u32,
    left_elapsed: f64,
    right_tokens: u32,
    right_elapsed: f64,
}

pub struct ChatApp {
    // Shared config
    temperature: f32,
    max_tokens: u32,
    system_prompt: String,
    show_system_prompt: bool,

    // Race state
    race_prompt: String,
    race_active: bool,
    race_start: Option<Instant>,
    race_left_done: bool,
    race_right_done: bool,
    race_result: Option<RaceResult>,

    // Panes
    left: PaneState,
    right: PaneState,

    // Runtime
    runtime: tokio::runtime::Handle,
    theme_applied: bool,
}

impl ChatApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, runtime: tokio::runtime::Handle) -> Self {
        let mut app = Self {
            temperature: 0.7,
            max_tokens: 2048,
            system_prompt: "You are a helpful assistant.".to_string(),
            show_system_prompt: false,
            race_prompt: String::new(),
            race_active: false,
            race_start: None,
            race_left_done: false,
            race_right_done: false,
            race_result: None,
            left: PaneState::new("GPU", "http://localhost:8081/v1/chat/completions"),
            right: PaneState::new("TPU", "http://localhost:8080/v1/chat/completions"),
            runtime,
            theme_applied: false,
        };
        app.refresh_models_for(PaneSide::Left);
        app.refresh_models_for(PaneSide::Right);
        app
    }

    fn pane(&self, side: PaneSide) -> &PaneState {
        match side {
            PaneSide::Left => &self.left,
            PaneSide::Right => &self.right,
        }
    }

    fn pane_mut(&mut self, side: PaneSide) -> &mut PaneState {
        match side {
            PaneSide::Left => &mut self.left,
            PaneSide::Right => &mut self.right,
        }
    }

    fn refresh_models_for(&mut self, side: PaneSide) {
        let (tx, rx) = mpsc::channel();
        let endpoint = self.pane(side).endpoint.clone();
        self.pane_mut(side).model_rx = Some(rx);
        let _guard = self.runtime.enter();
        api::fetch_models(endpoint, tx);
    }

    fn build_api_messages(&self, pane: &PaneState) -> Vec<ChatMessage> {
        let mut api_messages = Vec::new();
        let sys = self.system_prompt.trim();
        if !sys.is_empty() {
            api_messages.push(ChatMessage {
                role: "system".to_string(),
                content: sys.to_string(),
            });
        }
        for msg in &pane.messages {
            api_messages.push(ChatMessage {
                role: match msg.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                }
                .to_string(),
                content: msg.content.clone(),
            });
        }
        api_messages
    }

    fn send_message_for(&mut self, side: PaneSide) {
        let pane = self.pane(side);
        let text = pane.input.trim().to_string();
        if text.is_empty() || pane.streaming {
            return;
        }

        let pane = self.pane_mut(side);
        pane.messages.push(DisplayMessage {
            role: Role::User,
            content: text.clone(),
        });
        pane.input.clear();

        let api_messages = self.build_api_messages(self.pane(side));
        let model = {
            let p = self.pane(side);
            if p.model.is_empty() { "default".to_string() } else { p.model.clone() }
        };

        let request = ChatRequest {
            model,
            messages: api_messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            stream: true,
        };

        let pane = self.pane_mut(side);
        pane.messages.push(DisplayMessage {
            role: Role::Assistant,
            content: String::new(),
        });

        let (tx, rx) = mpsc::channel();
        pane.stream_rx = Some(rx);
        pane.streaming = true;
        pane.current_tps = None;
        pane.last_stats = None;
        pane.scroll_to_bottom = true;

        let endpoint = pane.endpoint.clone();
        let _guard = self.runtime.enter();
        api::stream_chat(endpoint, request, tx);
    }

    fn send_race_prompt_to(&mut self, side: PaneSide, prompt: &str) {
        let pane = self.pane_mut(side);
        pane.messages.push(DisplayMessage {
            role: Role::User,
            content: prompt.to_string(),
        });

        let api_messages = self.build_api_messages(self.pane(side));
        let model = {
            let p = self.pane(side);
            if p.model.is_empty() { "default".to_string() } else { p.model.clone() }
        };

        let request = ChatRequest {
            model,
            messages: api_messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            stream: true,
        };

        let pane = self.pane_mut(side);
        pane.messages.push(DisplayMessage {
            role: Role::Assistant,
            content: String::new(),
        });

        let (tx, rx) = mpsc::channel();
        pane.stream_rx = Some(rx);
        pane.streaming = true;
        pane.current_tps = None;
        pane.last_stats = None;
        pane.scroll_to_bottom = true;
        pane.race_token_count = 0;
        pane.race_finish_time = None;
        pane.race_elapsed_secs = None;

        let endpoint = pane.endpoint.clone();
        let _guard = self.runtime.enter();
        api::stream_chat(endpoint, request, tx);
    }

    fn start_race(&mut self) {
        let prompt = self.race_prompt.trim().to_string();
        if prompt.is_empty() || self.left.streaming || self.right.streaming {
            return;
        }

        self.race_active = true;
        self.race_start = Some(Instant::now());
        self.race_left_done = false;
        self.race_right_done = false;
        self.race_result = None;

        // Send to both panes
        self.send_race_prompt_to(PaneSide::Left, &prompt);
        self.send_race_prompt_to(PaneSide::Right, &prompt);
    }

    fn poll_pane(&mut self, side: PaneSide) {
        // Poll model list
        let pane = self.pane_mut(side);
        if let Some(rx) = &pane.model_rx {
            if let Ok(models) = rx.try_recv() {
                if !models.is_empty() {
                    let current = pane.model.clone();
                    pane.available_models = models;
                    if current.is_empty() || !pane.available_models.contains(&current) {
                        pane.model = pane.available_models[0].clone();
                    }
                }
                pane.model_rx = None;
            }
        }

        // Poll stream events
        let pane = self.pane_mut(side);
        if pane.stream_rx.is_none() {
            return;
        }

        let mut got_token = false;
        loop {
            let event = {
                if let Some(rx) = &pane.stream_rx {
                    match rx.try_recv() {
                        Ok(e) => Some(e),
                        Err(mpsc::TryRecvError::Empty) => None,
                        Err(mpsc::TryRecvError::Disconnected) => {
                            pane.streaming = false;
                            pane.stream_rx = None;
                            break;
                        }
                    }
                } else {
                    break;
                }
            };

            match event {
                Some(StreamEvent::Token(tok)) => {
                    if let Some(last) = pane.messages.last_mut() {
                        last.content.push_str(&tok);
                    }
                    pane.race_token_count += 1;
                    got_token = true;
                }
                Some(StreamEvent::TokPerSec(tps)) => {
                    pane.current_tps = Some(tps);
                }
                Some(StreamEvent::Done { tokens, elapsed_secs }) => {
                    let final_tps = if elapsed_secs > 0.0 {
                        tokens as f64 / elapsed_secs
                    } else {
                        0.0
                    };
                    pane.last_stats = Some(format!(
                        "{} tokens in {:.1}s ({:.1} tok/s)",
                        tokens, elapsed_secs, final_tps
                    ));
                    pane.streaming = false;
                    pane.current_tps = None;
                    pane.stream_rx = None;
                    pane.race_token_count = tokens;
                    pane.race_finish_time = Some(Instant::now());
                    pane.race_elapsed_secs = Some(elapsed_secs);
                    break;
                }
                Some(StreamEvent::Error(e)) => {
                    if let Some(last) = pane.messages.last_mut() {
                        if last.role == Role::Assistant && last.content.is_empty() {
                            last.content = format!("[Error: {}]", e);
                        }
                    }
                    pane.streaming = false;
                    pane.current_tps = None;
                    pane.stream_rx = None;
                    pane.race_finish_time = Some(Instant::now());
                    pane.race_elapsed_secs = Some(0.0);
                    break;
                }
                None => break,
            }
        }

        if got_token {
            self.pane_mut(side).scroll_to_bottom = true;
        }
    }

    fn check_race_completion(&mut self) {
        if !self.race_active {
            return;
        }

        if !self.race_left_done && !self.left.streaming {
            self.race_left_done = true;
        }
        if !self.race_right_done && !self.right.streaming {
            self.race_right_done = true;
        }

        if self.race_left_done && self.race_right_done {
            self.race_active = false;
            self.race_result = Some(RaceResult {
                left_tokens: self.left.race_token_count,
                left_elapsed: self.left.race_elapsed_secs.unwrap_or(0.0),
                right_tokens: self.right.race_token_count,
                right_elapsed: self.right.race_elapsed_secs.unwrap_or(0.0),
            });
        }
    }

    fn render_top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("rvLLM Race").size(18.0).strong().color(theme::ACCENT));
            ui.add_space(16.0);

            // Race prompt input
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.race_prompt)
                    .desired_width(ui.available_width() - 400.0)
                    .hint_text("Enter race prompt... (Ctrl+R to race)")
                    .font(egui::TextStyle::Body),
            );

            let ctrl_r = ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::R));
            if resp.has_focus() {
                let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
                if enter && !self.race_active {
                    self.start_race();
                }
            }

            let can_race = !self.race_prompt.trim().is_empty()
                && !self.race_active
                && !self.left.streaming
                && !self.right.streaming;

            let race_btn = ui.add_enabled(
                can_race,
                egui::Button::new(
                    RichText::new("Race!").color(Color32::WHITE).size(15.0).strong(),
                )
                .fill(if can_race { theme::ACCENT } else { theme::BG_INPUT })
                .corner_radius(CornerRadius::same(8))
                .min_size(Vec2::new(70.0, 32.0)),
            );
            if race_btn.clicked() || (ctrl_r && can_race) {
                self.start_race();
            }

            ui.add_space(8.0);

            // Race timer
            if self.race_active {
                if let Some(start) = self.race_start {
                    let elapsed = start.elapsed().as_secs_f64();
                    ui.label(
                        RichText::new(format!("{:.3}s", elapsed))
                            .size(20.0)
                            .strong()
                            .color(theme::RACE_TIMER)
                            .family(egui::FontFamily::Monospace),
                    );
                }
            } else if let Some(start) = self.race_start {
                // Show final time
                let final_time = if let (Some(lt), Some(rt)) = (self.left.race_finish_time, self.right.race_finish_time) {
                    let later = if lt > rt { lt } else { rt };
                    later.duration_since(start).as_secs_f64()
                } else {
                    0.0
                };
                if final_time > 0.0 {
                    ui.label(
                        RichText::new(format!("{:.3}s", final_time))
                            .size(20.0)
                            .strong()
                            .color(theme::TEXT_SECONDARY)
                            .family(egui::FontFamily::Monospace),
                    );
                }
            }
        });

        // Shared controls row
        ui.horizontal(|ui| {
            // Temperature
            ui.label(
                RichText::new(format!("Temp: {:.2}", self.temperature))
                    .size(12.0)
                    .color(theme::TEXT_SECONDARY),
            );
            ui.add(egui::Slider::new(&mut self.temperature, 0.0..=1.0).show_value(false));

            ui.add_space(12.0);

            // Max tokens
            ui.label(
                RichText::new(format!("Max Tokens: {}", self.max_tokens))
                    .size(12.0)
                    .color(theme::TEXT_SECONDARY),
            );
            ui.add(
                egui::Slider::new(&mut self.max_tokens, 64..=4096)
                    .show_value(false)
                    .logarithmic(true),
            );

            ui.add_space(12.0);

            // System prompt toggle
            let arrow = if self.show_system_prompt { "v" } else { ">" };
            if ui
                .add(
                    egui::Button::new(
                        RichText::new(format!("{} System Prompt", arrow))
                            .size(12.0)
                            .color(theme::TEXT_SECONDARY),
                    )
                    .frame(false),
                )
                .clicked()
            {
                self.show_system_prompt = !self.show_system_prompt;
            }
        });

        if self.show_system_prompt {
            ui.add(
                egui::TextEdit::singleline(&mut self.system_prompt)
                    .desired_width(ui.available_width())
                    .font(egui::TextStyle::Small),
            );
        }

        // Race result summary
        if let Some(result) = &self.race_result {
            let left_tps = if result.left_elapsed > 0.0 {
                result.left_tokens as f64 / result.left_elapsed
            } else {
                0.0
            };
            let right_tps = if result.right_elapsed > 0.0 {
                result.right_tokens as f64 / result.right_elapsed
            } else {
                0.0
            };

            let (winner, margin) = if result.left_elapsed < result.right_elapsed {
                (self.left.label.as_str(), (result.right_elapsed - result.left_elapsed) * 1000.0)
            } else if result.right_elapsed < result.left_elapsed {
                (self.right.label.as_str(), (result.left_elapsed - result.right_elapsed) * 1000.0)
            } else {
                ("Tie", 0.0)
            };

            // Need to clone to avoid borrow issues
            let left_label = self.left.label.clone();
            let right_label = self.right.label.clone();

            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!(
                        "{}: {} tok in {:.1}s ({:.1} tok/s)  |  {}: {} tok in {:.1}s ({:.1} tok/s)  |  Winner: {} by {:.0}ms",
                        left_label, result.left_tokens, result.left_elapsed, left_tps,
                        right_label, result.right_tokens, result.right_elapsed, right_tps,
                        winner, margin
                    ))
                    .size(13.0)
                    .color(theme::WINNER_GOLD),
                );
            });
        }

        ui.separator();
    }

    fn render_pane(
        pane: &mut PaneState,
        ui: &mut egui::Ui,
        side: PaneSide,
        accent: Color32,
        border_color: Color32,
        header_bg: Color32,
        runtime: &tokio::runtime::Handle,
    ) {
        let id_suffix = match side {
            PaneSide::Left => "left",
            PaneSide::Right => "right",
        };

        // Pane border
        egui::Frame::new()
            .stroke(egui::Stroke::new(1.5, border_color))
            .corner_radius(CornerRadius::same(8))
            .inner_margin(Margin::same(0))
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    // Header
                    egui::Frame::new()
                        .fill(header_bg)
                        .corner_radius(CornerRadius {
                            nw: 8,
                            ne: 8,
                            sw: 0,
                            se: 0,
                        })
                        .inner_margin(Margin::symmetric(10, 6))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new(&pane.label)
                                        .size(16.0)
                                        .strong()
                                        .color(accent),
                                );

                                ui.add_space(8.0);

                                // Streaming tok/s in header
                                if pane.streaming {
                                    if let Some(tps) = pane.current_tps {
                                        ui.label(
                                            RichText::new(format!("{:.1} tok/s", tps))
                                                .size(14.0)
                                                .strong()
                                                .color(theme::SUCCESS),
                                        );
                                    }
                                }

                                // Clear button on right
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if ui
                                        .add(
                                            egui::Button::new(
                                                RichText::new("Clear").size(11.0).color(theme::ERROR),
                                            )
                                            .corner_radius(CornerRadius::same(4)),
                                        )
                                        .clicked()
                                    {
                                        pane.messages.clear();
                                        pane.last_stats = None;
                                        pane.current_tps = None;
                                    }
                                });
                            });

                            // Endpoint + model row
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new("Endpoint:").size(11.0).color(theme::TEXT_DIM),
                                );
                                let resp = ui.add(
                                    egui::TextEdit::singleline(&mut pane.endpoint)
                                        .desired_width(ui.available_width() - 10.0)
                                        .font(egui::TextStyle::Small),
                                );
                                if resp.lost_focus()
                                    && ui.input(|i| i.key_pressed(egui::Key::Enter))
                                {
                                    let (tx, rx) = mpsc::channel();
                                    pane.model_rx = Some(rx);
                                    let endpoint = pane.endpoint.clone();
                                    let _guard = runtime.enter();
                                    api::fetch_models(endpoint, tx);
                                }
                            });

                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new("Model:").size(11.0).color(theme::TEXT_DIM),
                                );
                                if pane.available_models.is_empty() {
                                    ui.label(
                                        RichText::new("No models")
                                            .size(11.0)
                                            .color(theme::TEXT_DIM),
                                    );
                                    if ui
                                        .small_button("Refresh")
                                        .clicked()
                                    {
                                        let (tx, rx) = mpsc::channel();
                                        pane.model_rx = Some(rx);
                                        let endpoint = pane.endpoint.clone();
                                        let _guard = runtime.enter();
                                        api::fetch_models(endpoint, tx);
                                    }
                                } else {
                                    egui::ComboBox::from_id_salt(format!("model_{}", id_suffix))
                                        .selected_text(&pane.model)
                                        .width(ui.available_width() - 10.0)
                                        .show_ui(ui, |ui| {
                                            for m in pane.available_models.clone() {
                                                ui.selectable_value(
                                                    &mut pane.model,
                                                    m.clone(),
                                                    &m,
                                                );
                                            }
                                        });
                                }
                            });
                        });

                    // Chat area
                    let avail = ui.available_size();
                    let input_height = 50.0;
                    let chat_height = (avail.y - input_height - 8.0).max(100.0);

                    ui.allocate_ui(Vec2::new(avail.x, chat_height), |ui| {
                        ScrollArea::vertical()
                            .id_salt(format!("chat_scroll_{}", id_suffix))
                            .auto_shrink([false, false])
                            .stick_to_bottom(true)
                            .show(ui, |ui| {
                                ui.add_space(4.0);

                                if pane.messages.is_empty() {
                                    ui.vertical_centered(|ui| {
                                        ui.add_space(chat_height * 0.3);
                                        ui.label(
                                            RichText::new("Start a conversation")
                                                .size(16.0)
                                                .color(theme::TEXT_DIM),
                                        );
                                    });
                                } else {
                                    let max_w = (ui.available_width() * 0.85).min(500.0);
                                    for msg in &pane.messages {
                                        render_message_bubble(ui, msg, max_w);
                                        ui.add_space(2.0);
                                    }
                                }

                                // Streaming indicator
                                if pane.streaming {
                                    ui.horizontal(|ui| {
                                        ui.add_space(8.0);
                                        let t = ui.input(|i| i.time);
                                        let alpha = ((t * 3.0).sin() * 0.5 + 0.5) as u8;
                                        let color = Color32::from_rgba_premultiplied(
                                            accent.r(),
                                            accent.g(),
                                            accent.b(),
                                            (alpha as u16 * 200 / 255) as u8,
                                        );
                                        ui.label(RichText::new("|").size(16.0).color(color));
                                    });
                                }

                                if let Some(stats) = &pane.last_stats {
                                    ui.horizontal(|ui| {
                                        ui.add_space(8.0);
                                        ui.label(
                                            RichText::new(stats.as_str())
                                                .size(11.0)
                                                .color(theme::TEXT_DIM),
                                        );
                                    });
                                }

                                ui.add_space(4.0);
                            });
                    });

                    // Input bar
                    ui.horizontal(|ui| {
                        ui.add_space(4.0);
                        let input_resp = ui.add(
                            egui::TextEdit::singleline(&mut pane.input)
                                .desired_width(ui.available_width() - 56.0)
                                .hint_text("Type a message...")
                                .font(egui::TextStyle::Body),
                        );

                        let enter_pressed = input_resp.has_focus()
                            && ui.input(|i| i.key_pressed(egui::Key::Enter))
                            && !ui.input(|i| i.modifiers.shift);

                        let send_enabled = !pane.input.trim().is_empty() && !pane.streaming;
                        let btn_text = if pane.streaming { "..." } else { ">" };
                        let btn = ui.add_enabled(
                            send_enabled,
                            egui::Button::new(
                                RichText::new(btn_text).color(Color32::WHITE).size(14.0),
                            )
                            .fill(if send_enabled { accent } else { theme::BG_INPUT })
                            .corner_radius(CornerRadius::same(6))
                            .min_size(Vec2::new(40.0, 28.0)),
                        );

                        // Return whether to send
                        if (btn.clicked() || enter_pressed) && send_enabled {
                            // Mark for send -- handled via flag
                            pane.scroll_to_bottom = true;
                            // We set a sentinel: put a NUL at start of input
                            // Actually we can't mutate and send from here because we don't have ChatApp.
                            // Instead, use a simple flag approach: prefix input with a magic marker
                            // No, let's just check enter_pressed in the caller. We'll store a "send_requested" flag.
                        }

                        // Store whether send was requested in scroll_to_bottom as dual-purpose
                        // Actually let's be cleaner -- just check from outside if enter was pressed
                        // We'll handle this differently: store the send request
                        if (btn.clicked() || enter_pressed) && send_enabled {
                            // Tag: we use a special approach. We'll trim and mark.
                            // Since we can't call send_message_for from here, we embed a signal.
                            pane.input.insert(0, '\x01'); // SOH byte as send signal
                        }
                    });
                });
            });
    }
}

fn render_message_bubble(ui: &mut egui::Ui, msg: &DisplayMessage, max_width: f32) {
    let (bg, align, label_text) = match msg.role {
        Role::User => (theme::BG_USER_BUBBLE, Align::RIGHT, "You"),
        Role::Assistant => (theme::BG_ASSISTANT_BUBBLE, Align::LEFT, "AI"),
    };

    ui.with_layout(Layout::top_down(align), |ui| {
        ui.allocate_ui(Vec2::new(max_width, 0.0), |ui| {
            ui.label(RichText::new(label_text).size(10.0).color(theme::TEXT_DIM));
            egui::Frame::new()
                .fill(bg)
                .corner_radius(CornerRadius::same(8))
                .inner_margin(Margin::symmetric(10, 6))
                .show(ui, |ui| {
                    ui.set_max_width(max_width - 20.0);
                    let text = if msg.content.is_empty() && msg.role == Role::Assistant {
                        "..."
                    } else {
                        &msg.content
                    };
                    ui.label(RichText::new(text).size(14.0).color(theme::TEXT_PRIMARY));
                });
        });
    });
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.theme_applied {
            theme::apply_theme(ctx);
            self.theme_applied = true;
        }

        self.poll_pane(PaneSide::Left);
        self.poll_pane(PaneSide::Right);
        self.check_race_completion();

        if self.left.streaming || self.right.streaming || self.race_active {
            ctx.request_repaint();
        }

        // Top bar
        egui::TopBottomPanel::top("top_bar")
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_DARK)
                    .inner_margin(Margin::symmetric(12, 8))
                    .stroke(egui::Stroke::new(1.0, theme::BORDER)),
            )
            .show(ctx, |ui| {
                self.render_top_bar(ui);
            });

        // Main area: two columns
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_PANEL)
                    .inner_margin(Margin::symmetric(8, 8)),
            )
            .show(ctx, |ui| {
                let avail = ui.available_size();
                let pane_width = (avail.x - 12.0) / 2.0;

                ui.horizontal(|ui| {
                    // Left pane
                    ui.allocate_ui(Vec2::new(pane_width, avail.y), |ui| {
                        Self::render_pane(
                            &mut self.left,
                            ui,
                            PaneSide::Left,
                            theme::GPU_ACCENT,
                            theme::GPU_BORDER,
                            theme::GPU_HEADER_BG,
                            &self.runtime,
                        );
                    });

                    ui.add_space(4.0);

                    // Right pane
                    ui.allocate_ui(Vec2::new(pane_width, avail.y), |ui| {
                        Self::render_pane(
                            &mut self.right,
                            ui,
                            PaneSide::Right,
                            theme::TPU_ACCENT,
                            theme::TPU_BORDER,
                            theme::TPU_HEADER_BG,
                            &self.runtime,
                        );
                    });
                });
            });

        // Check send signals from pane inputs (SOH byte prefix)
        for side in [PaneSide::Left, PaneSide::Right] {
            let pane = self.pane(side);
            if pane.input.starts_with('\x01') {
                self.pane_mut(side).input.remove(0);
                self.send_message_for(side);
            }
        }
    }
}
