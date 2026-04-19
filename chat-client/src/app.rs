use crate::api::{self, ChatMessage, ChatRequest, StreamEvent};
use crate::theme;
use egui::{Align, Color32, CornerRadius, Layout, Margin, RichText, ScrollArea, Vec2};
use std::sync::mpsc;

#[derive(PartialEq)]
enum Role {
    User,
    Assistant,
}

struct DisplayMessage {
    role: Role,
    content: String,
}

pub struct ChatApp {
    // Config
    endpoint: String,
    model: String,
    available_models: Vec<String>,
    temperature: f32,
    max_tokens: u32,
    system_prompt: String,
    show_system_prompt: bool,

    // Chat state
    messages: Vec<DisplayMessage>,
    input: String,
    streaming: bool,
    current_tps: Option<f64>,
    last_stats: Option<String>,

    // Channels
    stream_rx: Option<mpsc::Receiver<StreamEvent>>,
    model_rx: Option<mpsc::Receiver<Vec<String>>>,

    // Runtime
    runtime: tokio::runtime::Handle,

    // Scroll
    scroll_to_bottom: bool,
    theme_applied: bool,
}

impl ChatApp {
    pub fn new(cc: &eframe::CreationContext<'_>, runtime: tokio::runtime::Handle) -> Self {
        let _ = &cc.egui_ctx;
        let mut app = Self {
            endpoint: "http://localhost:8080/v1/chat/completions".to_string(),
            model: String::new(),
            available_models: vec![],
            temperature: 0.7,
            max_tokens: 2048,
            system_prompt: "You are a helpful assistant.".to_string(),
            show_system_prompt: false,
            messages: vec![],
            input: String::new(),
            streaming: false,
            current_tps: None,
            last_stats: None,
            stream_rx: None,
            model_rx: None,
            runtime,
            scroll_to_bottom: false,
            theme_applied: false,
        };
        app.refresh_models();
        app
    }

    fn refresh_models(&mut self) {
        let (tx, rx) = mpsc::channel();
        self.model_rx = Some(rx);
        let endpoint = self.endpoint.clone();
        let _guard = self.runtime.enter();
        api::fetch_models(endpoint, tx);
    }

    fn send_message(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() || self.streaming {
            return;
        }

        self.messages.push(DisplayMessage {
            role: Role::User,
            content: text.clone(),
        });
        self.input.clear();

        // Build message history for API
        let mut api_messages = Vec::new();
        let sys = self.system_prompt.trim();
        if !sys.is_empty() {
            api_messages.push(ChatMessage {
                role: "system".to_string(),
                content: sys.to_string(),
            });
        }
        for msg in &self.messages {
            api_messages.push(ChatMessage {
                role: match msg.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                }
                .to_string(),
                content: msg.content.clone(),
            });
        }

        let request = ChatRequest {
            model: if self.model.is_empty() {
                "default".to_string()
            } else {
                self.model.clone()
            },
            messages: api_messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            stream: true,
        };

        // Start streaming
        self.messages.push(DisplayMessage {
            role: Role::Assistant,
            content: String::new(),
        });

        let (tx, rx) = mpsc::channel();
        self.stream_rx = Some(rx);
        self.streaming = true;
        self.current_tps = None;
        self.last_stats = None;
        self.scroll_to_bottom = true;

        let endpoint = self.endpoint.clone();
        let _guard = self.runtime.enter();
        api::stream_chat(endpoint, request, tx);
    }

    fn poll_stream(&mut self) {
        // Poll model list
        if let Some(rx) = &self.model_rx {
            if let Ok(models) = rx.try_recv() {
                if !models.is_empty() {
                    self.available_models = models;
                    if self.model.is_empty() || !self.available_models.contains(&self.model) {
                        self.model = self.available_models[0].clone();
                    }
                }
                self.model_rx = None;
            }
        }

        // Poll stream events
        if let Some(rx) = &self.stream_rx {
            let mut got_token = false;
            // Drain all available events this frame
            loop {
                match rx.try_recv() {
                    Ok(event) => match event {
                        StreamEvent::Token(tok) => {
                            if let Some(last) = self.messages.last_mut() {
                                last.content.push_str(&tok);
                            }
                            got_token = true;
                        }
                        StreamEvent::TokPerSec(tps) => {
                            self.current_tps = Some(tps);
                        }
                        StreamEvent::Done { tokens, elapsed_secs } => {
                            let final_tps = if elapsed_secs > 0.0 {
                                tokens as f64 / elapsed_secs
                            } else {
                                0.0
                            };
                            self.last_stats = Some(format!(
                                "{} tokens in {:.1}s ({:.1} tok/s)",
                                tokens, elapsed_secs, final_tps
                            ));
                            self.streaming = false;
                            self.current_tps = None;
                            self.stream_rx = None;
                            return;
                        }
                        StreamEvent::Error(e) => {
                            if let Some(last) = self.messages.last_mut() {
                                if last.role == Role::Assistant && last.content.is_empty() {
                                    last.content = format!("[Error: {}]", e);
                                }
                            }
                            self.streaming = false;
                            self.current_tps = None;
                            self.stream_rx = None;
                            return;
                        }
                    },
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        self.streaming = false;
                        self.stream_rx = None;
                        return;
                    }
                }
            }
            if got_token {
                self.scroll_to_bottom = true;
            }
        }
    }

    fn render_sidebar(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.add_space(8.0);
            ui.label(RichText::new("rvLLM Chat").size(18.0).strong().color(theme::ACCENT));
            ui.add_space(12.0);
            ui.separator();
            ui.add_space(8.0);

            // Endpoint
            ui.label(RichText::new("API Endpoint").size(12.0).color(theme::TEXT_SECONDARY));
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.endpoint)
                    .desired_width(f32::INFINITY)
                    .font(egui::TextStyle::Small),
            );
            if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                self.refresh_models();
            }
            ui.add_space(4.0);
            if ui.button("Refresh Models").clicked() {
                self.refresh_models();
            }
            ui.add_space(12.0);

            // Model selector
            ui.label(RichText::new("Model").size(12.0).color(theme::TEXT_SECONDARY));
            if self.available_models.is_empty() {
                ui.label(RichText::new("No models loaded").size(12.0).color(theme::TEXT_DIM));
            } else {
                egui::ComboBox::from_id_salt("model_select")
                    .selected_text(&self.model)
                    .width(ui.available_width() - 8.0)
                    .show_ui(ui, |ui| {
                        for m in self.available_models.clone() {
                            ui.selectable_value(&mut self.model, m.clone(), &m);
                        }
                    });
            }
            ui.add_space(12.0);

            // Temperature
            ui.label(
                RichText::new(format!("Temperature: {:.2}", self.temperature))
                    .size(12.0)
                    .color(theme::TEXT_SECONDARY),
            );
            ui.add(egui::Slider::new(&mut self.temperature, 0.0..=1.0).show_value(false));
            ui.add_space(8.0);

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

            // System prompt
            let arrow = if self.show_system_prompt { "v" } else { ">" };
            if ui
                .add(egui::Button::new(
                    RichText::new(format!("{} System Prompt", arrow))
                        .size(12.0)
                        .color(theme::TEXT_SECONDARY),
                ).frame(false))
                .clicked()
            {
                self.show_system_prompt = !self.show_system_prompt;
            }
            if self.show_system_prompt {
                ui.add_space(4.0);
                ui.add(
                    egui::TextEdit::multiline(&mut self.system_prompt)
                        .desired_width(f32::INFINITY)
                        .desired_rows(4)
                        .font(egui::TextStyle::Small),
                );
            }
            ui.add_space(16.0);

            // Clear chat
            if ui
                .add(
                    egui::Button::new(RichText::new("Clear Chat").color(theme::ERROR))
                        .corner_radius(CornerRadius::same(4)),
                )
                .clicked()
            {
                self.messages.clear();
                self.last_stats = None;
                self.current_tps = None;
            }
        });
    }

    fn render_chat(&mut self, ui: &mut egui::Ui) {
        let panel_rect = ui.available_rect_before_wrap();
        // Reserve space for input bar at bottom
        let input_height = 64.0;
        let chat_height = panel_rect.height() - input_height - 16.0;

        // Chat messages area
        ui.allocate_ui(Vec2::new(panel_rect.width(), chat_height), |ui| {
            ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.add_space(8.0);

                    if self.messages.is_empty() {
                        ui.vertical_centered(|ui| {
                            ui.add_space(chat_height * 0.3);
                            ui.label(
                                RichText::new("Start a conversation")
                                    .size(20.0)
                                    .color(theme::TEXT_DIM),
                            );
                        });
                    } else {
                        let max_width = (ui.available_width() * 0.75).min(700.0);
                        for msg in &self.messages {
                            self.render_message_bubble(ui, msg, max_width);
                            ui.add_space(4.0);
                        }
                    }

                    // Streaming indicator
                    if self.streaming {
                        ui.horizontal(|ui| {
                            ui.add_space(12.0);
                            if let Some(tps) = self.current_tps {
                                ui.label(
                                    RichText::new(format!("{:.1} tok/s", tps))
                                        .size(12.0)
                                        .color(theme::SUCCESS),
                                );
                            }
                            // Blinking cursor
                            let t = ui.input(|i| i.time);
                            let alpha = ((t * 3.0).sin() * 0.5 + 0.5) as u8;
                            let color = Color32::from_rgba_premultiplied(
                                theme::ACCENT.r(),
                                theme::ACCENT.g(),
                                theme::ACCENT.b(),
                                (alpha as u16 * 200 / 255) as u8,
                            );
                            ui.label(RichText::new("|").size(16.0).color(color));
                        });
                    }

                    // Last generation stats
                    if let Some(stats) = &self.last_stats {
                        ui.horizontal(|ui| {
                            ui.add_space(12.0);
                            ui.label(
                                RichText::new(stats.as_str())
                                    .size(11.0)
                                    .color(theme::TEXT_DIM),
                            );
                        });
                    }

                    ui.add_space(8.0);
                });
        });

        ui.add_space(4.0);

        // Input bar
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            let input_resp = ui.add(
                egui::TextEdit::multiline(&mut self.input)
                    .desired_width(ui.available_width() - 70.0)
                    .desired_rows(1)
                    .hint_text("Type a message... (Enter to send, Shift+Enter for newline)")
                    .font(egui::TextStyle::Body)
                    .margin(Margin::symmetric(10, 8)),
            );

            // Handle Enter to send (without Shift)
            if input_resp.has_focus() {
                let enter_pressed = ui.input(|i| i.key_pressed(egui::Key::Enter));
                let shift_held = ui.input(|i| i.modifiers.shift);
                if enter_pressed && !shift_held {
                    // Remove the newline that was just inserted
                    if self.input.ends_with('\n') {
                        self.input.pop();
                    }
                    self.send_message();
                }
            }

            let send_enabled = !self.input.trim().is_empty() && !self.streaming;
            let btn_text = if self.streaming { "..." } else { "Send" };
            let btn = ui.add_enabled(
                send_enabled,
                egui::Button::new(
                    RichText::new(btn_text).color(Color32::WHITE).size(14.0),
                )
                .fill(if send_enabled { theme::ACCENT } else { theme::BG_INPUT })
                .corner_radius(CornerRadius::same(8))
                .min_size(Vec2::new(56.0, 36.0)),
            );
            if btn.clicked() {
                self.send_message();
            }
        });
    }

    fn render_message_bubble(&self, ui: &mut egui::Ui, msg: &DisplayMessage, max_width: f32) {
        let (bg, align, label_text) = match msg.role {
            Role::User => (theme::BG_USER_BUBBLE, Align::RIGHT, "You"),
            Role::Assistant => (theme::BG_ASSISTANT_BUBBLE, Align::LEFT, "Assistant"),
        };

        ui.with_layout(Layout::top_down(align), |ui| {
            ui.allocate_ui(Vec2::new(max_width, 0.0), |ui| {
                // Role label
                ui.label(RichText::new(label_text).size(11.0).color(theme::TEXT_DIM));

                // Bubble
                egui::Frame::new()
                    .fill(bg)
                    .corner_radius(CornerRadius::same(10))
                    .inner_margin(Margin::symmetric(12, 8))
                    .show(ui, |ui| {
                        ui.set_max_width(max_width - 24.0);
                        let text = if msg.content.is_empty() && msg.role == Role::Assistant {
                            "..."
                        } else {
                            &msg.content
                        };
                        ui.label(RichText::new(text).size(14.5).color(theme::TEXT_PRIMARY));
                    });
            });
        });
    }
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.theme_applied {
            theme::apply_theme(ctx);
            self.theme_applied = true;
        }

        self.poll_stream();

        // Request repaint while streaming for smooth updates
        if self.streaming {
            ctx.request_repaint();
        }

        // Sidebar
        egui::SidePanel::left("sidebar")
            .resizable(true)
            .default_width(240.0)
            .min_width(200.0)
            .max_width(400.0)
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_DARK)
                    .inner_margin(Margin::symmetric(12, 8))
                    .stroke(egui::Stroke::new(1.0, theme::BORDER)),
            )
            .show(ctx, |ui| {
                self.render_sidebar(ui);
            });

        // Main chat area
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_PANEL)
                    .inner_margin(Margin::symmetric(16, 8)),
            )
            .show(ctx, |ui| {
                self.render_chat(ui);
            });
    }
}
