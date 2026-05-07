"""Standalone GUI application bootstrap."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import re
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Callable
from urllib.parse import urlparse, urlunparse
import structlog

from autocad_mcp.agent.planner import ActionPlanner
from autocad_mcp.agent.scene import SceneGraphBuilder, SceneGraphCache
from autocad_mcp.backends.base import CommandResult
from autocad_mcp.backends.file_ipc import FileIPCBackend
from autocad_mcp.gui.theme import apply_ttk_theme, default_gui_theme, style_canvas, style_scrolled_text
from autocad_mcp.llm.lmstudio_client import LMStudioClient, LMStudioConfig
from PIL import Image, ImageTk
log = structlog.get_logger()


class AutoCADStandaloneApp:
    """Desktop GUI that plans with LM Studio and executes approved actions."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AutoCAD LT + LM Studio")
        self.root.geometry("980x760")
        default_lmstudio_url = "http://10.0.0.28:1234"
        normalized_default_lmstudio_url = self._normalize_lmstudio_base_url(default_lmstudio_url)
        self.lm_client = LMStudioClient(LMStudioConfig(base_url=normalized_default_lmstudio_url))
        self.planner = ActionPlanner(self.lm_client)
        self.backend = FileIPCBackend()
        self.scene_graph_builder = SceneGraphBuilder(self.backend)
        self.scene_graph_cache = SceneGraphCache()

        self.current_plan: dict[str, Any] | None = None
        self.chat_history: list[dict[str, str]] = []
        self.reference_image_paths: list[str] = []
        self.captured_canvas_b64s: list[str] = []
        self._thumbnail_images: list[ImageTk.PhotoImage] = []
        self._chat_stream_start_index: str | None = None
        self._chat_stream_has_content = False
        self._planning_stream_start_index: str | None = None
        self._planning_stream_has_content = False

        self.lm_status = tk.StringVar(value="LM Studio: not checked")
        self.cad_status = tk.StringVar(value="AutoCAD LT: not checked")
        self.plan_status = tk.StringVar(value="Plan: none")
        self.visual_context_status = tk.StringVar(value="Visual context: no images")
        self.auto_capture_before_plan = tk.BooleanVar(value=False)
        self.auto_zoom_after_execute = tk.BooleanVar(value=True)
        self.use_live_autocad_context = tk.BooleanVar(value=True)
        self.chat_auto_execute = tk.BooleanVar(value=True)
        self.planning_mode_profile = tk.StringVar(value="fast")
        self.lm_base_url_var = tk.StringVar(value=default_lmstudio_url)
        self.theme = default_gui_theme()
        self.style = apply_ttk_theme(self.root, self.theme)

        self._build_ui()
        active_log_file = os.environ.get("AUTOCAD_MCP_ACTIVE_LOG_FILE", "").strip()
        if active_log_file:
            self._append_log(f"[diagnostics] log file: {active_log_file}")
            log.info("gui_diagnostics_log_file", path=active_log_file)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, style="App.TFrame")
        container.pack(fill="both", expand=True)
        self.main_canvas = tk.Canvas(container, highlightthickness=0)
        style_canvas(self.main_canvas, self.theme, background=self.theme.bg)
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.main_scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.main_canvas.yview)
        self.main_scrollbar.pack(side="right", fill="y")
        self.main_canvas.configure(yscrollcommand=self.main_scrollbar.set)

        frame = ttk.Frame(self.main_canvas, padding=14, style="Surface.TFrame")
        self.main_canvas_window = self.main_canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind(
            "<Configure>",
            lambda _e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all")),
        )
        self.main_canvas.bind(
            "<Configure>",
            lambda e: self.main_canvas.itemconfigure(self.main_canvas_window, width=e.width),
        )
        self.main_canvas.bind("<Enter>", self._bind_main_canvas_mousewheel)
        self.main_canvas.bind("<Leave>", self._unbind_main_canvas_mousewheel)

        status_card = ttk.Frame(frame, style="Card.TFrame", padding=(12, 10))
        status_card.pack(fill="x", pady=(0, 10))
        ttk.Label(status_card, text="AICAD Control Center", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            status_card,
            text="LM Studio planning and AutoCAD LT execution workflow",
            style="Subtle.TLabel",
        ).pack(anchor="w", pady=(0, 8))
        status_row = ttk.Frame(status_card, style="Card.TFrame")
        status_row.pack(fill="x")
        ttk.Label(status_row, textvariable=self.lm_status, style="Status.TLabel").pack(side="left", padx=(0, 6))
        ttk.Label(status_row, textvariable=self.cad_status, style="Status.TLabel").pack(side="left", padx=6)
        ttk.Label(status_row, textvariable=self.plan_status, style="Status.TLabel").pack(side="left", padx=6)
        ttk.Label(status_card, textvariable=self.visual_context_status, style="Caption.TLabel").pack(anchor="w", pady=(8, 0))

        lm_url_row = ttk.Frame(status_card, style="Card.TFrame")
        lm_url_row.pack(fill="x", pady=(8, 0))
        ttk.Label(lm_url_row, text="LM Studio URL", style="Subtle.TLabel").pack(side="left")
        ttk.Entry(
            lm_url_row,
            textvariable=self.lm_base_url_var,
            width=52,
            style="App.TEntry",
        ).pack(side="left", padx=(8, 8), fill="x", expand=True)
        ttk.Button(
            lm_url_row,
            text="Apply URL",
            style="Secondary.TButton",
            command=self.apply_lmstudio_url,
        ).pack(side="left")

        controls = ttk.LabelFrame(frame, text="Controls", style="App.TLabelframe", padding=(10, 10))
        controls.pack(fill="x", pady=(0, 10))
        action_row = ttk.Frame(controls, style="Surface.TFrame")
        action_row.pack(fill="x")
        ttk.Button(
            action_row,
            text="Check Connections",
            style="Secondary.TButton",
            command=self.check_connections,
        ).pack(side="left")
        ttk.Button(
            action_row,
            text="Add Reference Image(s)",
            style="Secondary.TButton",
            command=self.add_reference_images,
        ).pack(side="left", padx=8)
        ttk.Button(
            action_row,
            text="Capture AutoCAD Canvas",
            style="Secondary.TButton",
            command=self.capture_canvas_screenshot,
        ).pack(side="left")
        ttk.Button(
            action_row,
            text="Clear Visual Context",
            style="Danger.TButton",
            command=self.clear_visual_context,
        ).pack(side="left", padx=8)
        ttk.Button(
            action_row,
            text="Generate Plan",
            style="Primary.TButton",
            command=self.generate_plan,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            action_row,
            text="Execute Approved Plan",
            style="Primary.TButton",
            command=self.execute_plan,
        ).pack(side="left", padx=(8, 0))

        toggle_row = ttk.Frame(controls, style="Surface.TFrame")
        toggle_row.pack(fill="x", pady=(10, 2))
        ttk.Checkbutton(
            toggle_row,
            text="Auto-capture canvas before planning",
            variable=self.auto_capture_before_plan,
            style="App.TCheckbutton",
        ).pack(side="left", padx=8)
        ttk.Checkbutton(
            toggle_row,
            text="Auto-zoom extents after execution",
            variable=self.auto_zoom_after_execute,
            style="App.TCheckbutton",
        ).pack(side="left", padx=8)
        ttk.Checkbutton(
            toggle_row,
            text="Use live AutoCAD context for planning",
            variable=self.use_live_autocad_context,
            style="App.TCheckbutton",
        ).pack(side="left", padx=8)
        planning_mode_row = ttk.Frame(controls, style="Surface.TFrame")
        planning_mode_row.pack(fill="x", pady=(8, 0))
        ttk.Label(planning_mode_row, text="Planning mode", style="Body.TLabel").pack(side="left")
        ttk.Radiobutton(
            planning_mode_row,
            text="Fast mode (text-first, minimal vision)",
            variable=self.planning_mode_profile,
            value="fast",
            style="App.TRadiobutton",
        ).pack(side="left", padx=(8, 8))
        ttk.Radiobutton(
            planning_mode_row,
            text="Accurate mode (vision-first)",
            variable=self.planning_mode_profile,
            value="accurate",
            style="App.TRadiobutton",
        ).pack(side="left")

        self.thumbnail_panel = ttk.LabelFrame(
            frame,
            text="Visual Reference Thumbnails",
            style="App.TLabelframe",
            padding=(8, 8),
        )
        self.thumbnail_panel.pack(fill="x", expand=False, pady=(0, 10))
        self.thumb_canvas = tk.Canvas(self.thumbnail_panel, height=140, highlightthickness=0)
        style_canvas(self.thumb_canvas, self.theme, background=self.theme.panel_alt)
        self.thumb_canvas.pack(side="top", fill="x", expand=True)
        self.thumb_scroll = ttk.Scrollbar(self.thumbnail_panel, orient="horizontal", command=self.thumb_canvas.xview)
        self.thumb_scroll.pack(side="bottom", fill="x")
        self.thumb_canvas.configure(xscrollcommand=self.thumb_scroll.set)
        self.thumb_inner = ttk.Frame(self.thumb_canvas, style="Card.TFrame")
        self.thumb_window = self.thumb_canvas.create_window((0, 0), window=self.thumb_inner, anchor="nw")
        self.thumb_inner.bind(
            "<Configure>",
            lambda _e: self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox("all")),
        )
        self.thumb_canvas.bind(
            "<Configure>",
            lambda e: self.thumb_canvas.itemconfigure(self.thumb_window, height=e.height),
        )

        prompt_panel = ttk.LabelFrame(frame, text="Prompt", style="App.TLabelframe", padding=(8, 8))
        prompt_panel.pack(fill="x", expand=False)
        self.prompt_input = ScrolledText(prompt_panel, height=6, wrap="word")
        style_scrolled_text(self.prompt_input, self.theme)
        self.prompt_input.pack(fill="x", expand=False)
        self.prompt_input.insert("1.0", "Draw a line from (0,0) to (100,0) on layer 0.")

        chat_frame = ttk.LabelFrame(frame, text="Agent Chat (scene edits)", style="App.TLabelframe", padding=(8, 8))
        chat_frame.pack(fill="both", expand=False, pady=(10, 0))
        self.chat_output = ScrolledText(chat_frame, height=8, wrap="word", state="disabled")
        style_scrolled_text(self.chat_output, self.theme)
        self.chat_output.pack(fill="both", expand=True)
        chat_controls = ttk.Frame(chat_frame, style="Surface.TFrame")
        chat_controls.pack(fill="x", pady=(8, 0))
        self.chat_input = ScrolledText(chat_controls, height=3, wrap="word")
        style_scrolled_text(self.chat_input, self.theme)
        self.chat_input.pack(side="left", fill="x", expand=True)
        chat_buttons = ttk.Frame(chat_controls, style="Surface.TFrame")
        chat_buttons.pack(side="left", padx=(8, 0))
        ttk.Button(chat_buttons, text="Send Chat", style="Primary.TButton", command=self.send_chat_message).pack(fill="x")
        ttk.Button(
            chat_buttons,
            text="Describe Drawing",
            style="Secondary.TButton",
            command=self.describe_drawing,
        ).pack(fill="x", pady=(4, 0))
        ttk.Button(
            chat_buttons,
            text="Clear Chat",
            style="Secondary.TButton",
            command=self.clear_chat_history,
        ).pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(
            chat_buttons,
            text="Auto-execute chat plans",
            variable=self.chat_auto_execute,
            style="App.TCheckbutton",
        ).pack(anchor="w", pady=(6, 0))

        output_notebook = ttk.Notebook(frame, style="App.TNotebook")
        output_notebook.pack(fill="both", expand=True, pady=(10, 0))
        self.output_notebook = output_notebook
        plan_tab = ttk.Frame(output_notebook, style="Surface.TFrame", padding=(8, 8))
        output_notebook.add(plan_tab, text="Plan")
        ttk.Label(plan_tab, text="Plan preview (JSON)", style="Body.TLabel").pack(anchor="w", pady=(0, 4))
        self.plan_output = ScrolledText(plan_tab, height=16, wrap="word")
        style_scrolled_text(self.plan_output, self.theme)
        self.plan_output.pack(fill="both", expand=True)

        planning_stream_tab = ttk.Frame(output_notebook, style="Surface.TFrame", padding=(8, 8))
        output_notebook.add(planning_stream_tab, text="Planning Stream")
        self.planning_stream_tab = planning_stream_tab
        ttk.Label(
            planning_stream_tab,
            text="Live planner/model output while generating plan",
            style="Body.TLabel",
        ).pack(anchor="w", pady=(0, 4))
        self.planning_stream_output = ScrolledText(
            planning_stream_tab,
            height=16,
            wrap="word",
            state="disabled",
        )
        style_scrolled_text(self.planning_stream_output, self.theme)
        self.planning_stream_output.pack(fill="both", expand=True)

        live_output_tab = ttk.Frame(output_notebook, style="Surface.TFrame", padding=(8, 8))
        output_notebook.add(live_output_tab, text="Live Output")
        self.live_output_tab = live_output_tab
        ttk.Label(live_output_tab, text="Live execution/debug output", style="Body.TLabel").pack(anchor="w", pady=(0, 4))
        self.log_output = ScrolledText(live_output_tab, height=16, wrap="word")
        style_scrolled_text(self.log_output, self.theme)
        self.log_output.pack(fill="both", expand=True)

    def _bind_main_canvas_mousewheel(self, _event: tk.Event) -> None:
        self.main_canvas.bind_all("<MouseWheel>", self._on_main_canvas_mousewheel)

    def _unbind_main_canvas_mousewheel(self, _event: tk.Event) -> None:
        self.main_canvas.unbind_all("<MouseWheel>")

    def _on_main_canvas_mousewheel(self, event: tk.Event) -> None:
        delta = 0
        if hasattr(event, "delta") and event.delta:
            delta = int(-event.delta / 120)
        if delta != 0:
            self.main_canvas.yview_scroll(delta, "units")

    def _append_log(self, text: str) -> None:
        if hasattr(self, "output_notebook") and hasattr(self, "live_output_tab"):
            self.output_notebook.select(self.live_output_tab)
        self.log_output.insert("end", text + "\n")
        self.log_output.see("end")

    def _append_planning_stream(self, text: str) -> None:
        if not hasattr(self, "planning_stream_output"):
            return
        if hasattr(self, "output_notebook") and hasattr(self, "planning_stream_tab"):
            self.output_notebook.select(self.planning_stream_tab)
        self.planning_stream_output.configure(state="normal")
        self.planning_stream_output.insert("end", text)
        self.planning_stream_output.see("end")
        self.planning_stream_output.configure(state="disabled")

    def _clear_planning_stream(self) -> None:
        if not hasattr(self, "planning_stream_output"):
            return
        self.planning_stream_output.configure(state="normal")
        self.planning_stream_output.delete("1.0", "end")
        self.planning_stream_output.configure(state="disabled")
        self._planning_stream_start_index = None
        self._planning_stream_has_content = False

    def _begin_planning_stream(self, label: str) -> None:
        self._append_planning_stream(f"=== {label} ===\n")
        if not hasattr(self, "planning_stream_output"):
            return
        self.planning_stream_output.configure(state="normal")
        self._planning_stream_start_index = self.planning_stream_output.index("end-1c")
        self._planning_stream_has_content = False
        self.planning_stream_output.insert("end", "Model: ")
        self.planning_stream_output.see("end")
        self.planning_stream_output.configure(state="disabled")

    def _append_planning_stream_chunk(self, chunk: str) -> None:
        if self._planning_stream_start_index is None or not chunk or not hasattr(self, "planning_stream_output"):
            return
        self.planning_stream_output.configure(state="normal")
        self.planning_stream_output.insert("end", chunk)
        self.planning_stream_output.see("end")
        self.planning_stream_output.configure(state="disabled")
        self._planning_stream_has_content = True

    def _queue_planning_stream_text(
        self,
        text: str,
        can_accept: Callable[[], bool] | None = None,
    ) -> None:
        if not text:
            return

        def append_chunk(chunk: str = text, accept: Callable[[], bool] | None = can_accept) -> None:
            if accept is not None and not accept():
                return
            self._append_planning_stream_chunk(chunk)

        self.root.after(0, append_chunk)

    def _end_planning_stream(self, suffix: str | None = None) -> None:
        if self._planning_stream_start_index is None or not hasattr(self, "planning_stream_output"):
            return
        self.planning_stream_output.configure(state="normal")
        self.planning_stream_output.insert("end", "\n")
        if suffix:
            self.planning_stream_output.insert("end", f"{suffix}\n")
        self.planning_stream_output.insert("end", "\n")
        self.planning_stream_output.see("end")
        self.planning_stream_output.configure(state="disabled")
        self._planning_stream_start_index = None
        self._planning_stream_has_content = False

    def _cancel_planning_stream(self, suffix: str | None = None) -> None:
        if self._planning_stream_start_index is None or not hasattr(self, "planning_stream_output"):
            return
        self.planning_stream_output.configure(state="normal")
        self.planning_stream_output.delete(self._planning_stream_start_index, "end")
        if suffix:
            self.planning_stream_output.insert("end", f"{suffix}\n\n")
        self.planning_stream_output.see("end")
        self.planning_stream_output.configure(state="disabled")
        self._planning_stream_start_index = None
        self._planning_stream_has_content = False

    def _append_chat_message(self, speaker: str, text: str) -> None:
        self.chat_output.configure(state="normal")
        self.chat_output.insert("end", f"{speaker}: {text}\n\n")
        self.chat_output.see("end")
        self.chat_output.configure(state="disabled")

    def _append_chat_thinking(self, text: str) -> None:
        self._append_chat_message("Agent (thinking)", text)

    def _append_chat_full_output(self, title: str, payload: Any) -> None:
        try:
            rendered = json.dumps(payload, indent=2, ensure_ascii=False)
        except Exception:
            rendered = str(payload)
        self._append_chat_message("Agent (full output)", f"{title}:\n{rendered}")

    def _begin_chat_stream(self, speaker: str = "Agent") -> None:
        self.chat_output.configure(state="normal")
        self._chat_stream_start_index = self.chat_output.index("end-1c")
        self._chat_stream_has_content = False
        self.chat_output.insert("end", f"{speaker}: ")
        self.chat_output.see("end")
        self.chat_output.configure(state="disabled")

    def _append_chat_stream_chunk(self, chunk: str) -> None:
        if self._chat_stream_start_index is None or not chunk:
            return
        self.chat_output.configure(state="normal")
        self.chat_output.insert("end", chunk)
        self.chat_output.see("end")
        self.chat_output.configure(state="disabled")
        self._chat_stream_has_content = True
    @staticmethod
    def _split_stream_tokens(text: str) -> list[str]:
        if not text:
            return []
        tokens = re.findall(r"\s+|[^\s]+", text)
        return tokens or [text]

    def _queue_chat_stream_text(self, text: str) -> None:
        for token in self._split_stream_tokens(text):
            self.root.after(0, lambda t=token: self._append_chat_stream_chunk(t))

    def _end_chat_stream(self) -> None:
        if self._chat_stream_start_index is None:
            return
        self.chat_output.configure(state="normal")
        self.chat_output.insert("end", "\n\n")
        self.chat_output.see("end")
        self.chat_output.configure(state="disabled")
        self._chat_stream_start_index = None
        self._chat_stream_has_content = False

    def _cancel_chat_stream(self) -> None:
        if self._chat_stream_start_index is None:
            return
        self.chat_output.configure(state="normal")
        self.chat_output.delete(self._chat_stream_start_index, "end")
        self.chat_output.see("end")
        self.chat_output.configure(state="disabled")
        self._chat_stream_start_index = None
        self._chat_stream_has_content = False

    def clear_chat_history(self) -> None:
        self.chat_history.clear()
        self.chat_output.configure(state="normal")
        self.chat_output.delete("1.0", "end")
        self.chat_output.configure(state="disabled")
        self._chat_stream_start_index = None
        self._chat_stream_has_content = False

    def _build_chat_planner_prompt(self, latest_user_message: str) -> str:
        turns = self.chat_history[-4:]
        history_lines: list[str] = []
        for turn in turns:
            role = "User" if turn.get("role") == "user" else "Assistant"
            content = str(turn.get("content", "")).strip()
            if content:
                history_lines.append(f"{role}: {content}")
        history_block = "\n".join(history_lines)
        return (
            "Apply agentic edits to the active AutoCAD scene based on the conversation.\n"
            "Use the latest user message as the primary instruction while respecting prior context.\n\n"
            f"Conversation history:\n{history_block}\n\n"
            f"Latest user message:\n{latest_user_message.strip()}"
        )

    def _build_chat_feedback_prompt(
        self,
        latest_user_message: str,
        plan: dict[str, Any],
        execution_summary: dict[str, Any] | None,
        autocad_context: str | None,
    ) -> str:
        return (
            "User message:\n"
            f"{latest_user_message.strip()}\n\n"
            "Generated action plan:\n"
            f"{json.dumps(plan, ensure_ascii=False)}\n\n"
            "Execution summary:\n"
            f"{json.dumps(execution_summary or {}, ensure_ascii=False)}\n\n"
            "Current drawing context:\n"
            f"{(autocad_context or 'unavailable').strip()}\n\n"
            "Provide a concise user-facing update of what changed, what could not be done, and suggested next command."
        )

    def _request_lm_text_feedback(
        self,
        user_prompt: str,
        image_paths: list[str] | None = None,
        image_b64_pngs: list[str] | None = None,
        system_prompt: str | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        effective_system_prompt = system_prompt or (
            "You are an AutoCAD LT assistant. Reply in plain text only. "
            "Be concise and practical. Do not output JSON."
        )
        return self.lm_client.chat_text(
            effective_system_prompt,
            user_prompt,
            image_paths=image_paths,
            image_b64_pngs=image_b64_pngs,
            on_token=on_token,
        )

    def _prepare_lm_visual_context_sync(
        self,
        ensure_fresh_capture: bool,
        max_images: int = 2,
    ) -> tuple[list[str], list[str], str]:
        capture_note = ""
        if ensure_fresh_capture:
            ok, message = self._capture_canvas_snapshot_sync()
            if ok:
                capture_note = "fresh_capture=ok"
                self.root.after(0, self._refresh_visual_status)
                self.root.after(0, lambda m=message: self._append_log(f"Vision capture: {m}"))
            else:
                capture_note = f"fresh_capture_failed={message}"
                self.root.after(0, lambda m=message: self._append_log(f"Vision capture failed: {m}"))

        image_paths = list(self.reference_image_paths)
        image_b64_pngs = list(self.captured_canvas_b64s)
        total_images = len(image_paths) + len(image_b64_pngs)
        if total_images > max_images:
            remaining_for_captures = max(0, max_images - len(image_paths))
            if remaining_for_captures == 0:
                image_paths = image_paths[-max_images:]
                image_b64_pngs = []
            else:
                image_b64_pngs = image_b64_pngs[-remaining_for_captures:]

        summary = (
            f"vision_images={len(image_paths) + len(image_b64_pngs)} "
            f"(files={len(image_paths)}, captures={len(image_b64_pngs)})"
        )
        if capture_note:
            summary = f"{summary}, {capture_note}"
        return image_paths, image_b64_pngs, summary

    def _is_fast_planning_mode(self) -> bool:
        return self.planning_mode_profile.get() != "accurate"

    def _planning_visual_image_limits(self) -> tuple[int, int]:
        if self._is_fast_planning_mode():
            return 2, 1
        return 6, 2

    def _create_plan_with_resilience(
        self,
        prompt: str,
        autocad_context: str | None,
        image_paths: list[str],
        image_b64_pngs: list[str],
        on_planning_token: Callable[[str], None] | None = None,
        on_attempt_start: Callable[[str], None] | None = None,
    ) -> tuple[Any, str]:
        attempts: list[tuple[list[str], list[str], str]] = []
        if self._is_fast_planning_mode():
            attempts.append(([], [], "text_only"))
            if image_paths or image_b64_pngs:
                attempts.append((image_paths[-1:], image_b64_pngs[-1:], "vision_minimal"))
                attempts.append((list(image_paths), list(image_b64_pngs), "vision_full"))
        else:
            attempts.append((list(image_paths), list(image_b64_pngs), "vision_full"))
            if image_paths or image_b64_pngs:
                attempts.append((image_paths[-1:], image_b64_pngs[-1:], "vision_minimal"))
            attempts.append(([], [], "text_only"))

        errors: list[str] = []
        for paths, captures, mode in attempts:
            try:
                if on_attempt_start is not None:
                    on_attempt_start(mode)
                if paths or captures:
                    result = self.planner.create_plan_with_vision(
                        prompt,
                        image_paths=paths,
                        image_b64_pngs=captures,
                        autocad_context=autocad_context,
                        on_planning_token=on_planning_token,
                        backend_name=self.backend.name,
                    )
                else:
                    result = self.planner.create_plan(
                        prompt,
                        autocad_context=autocad_context,
                        on_planning_token=on_planning_token,
                        backend_name=self.backend.name,
                    )
                return result, mode
            except Exception as ex:
                message = str(ex)
                errors.append(f"{mode}: {message}")
                continue
        fallback_result = self.planner.create_fallback_plan(
            prompt,
            autocad_context=autocad_context,
            backend_name=self.backend.name,
        )
        if fallback_result is not None:
            return fallback_result, "deterministic_fallback"
        noop_fallback = self.planner.create_noop_fallback_plan(
            prompt,
            backend_name=self.backend.name,
        )
        if errors:
            noop_fallback.plan["notes"] = (
                f"{noop_fallback.plan.get('notes', '')} "
                f"LM attempts failed: {' | '.join(errors)}"
            ).strip()
        return noop_fallback, "safe_noop_fallback"

    def describe_drawing(self) -> None:
        self._append_chat_message("User", "Describe the current drawing.")
        self.plan_status.set("Plan: describing drawing...")

        def worker() -> None:
            stream_state: dict[str, Any] = {"started": False, "chunks": 0, "parts": []}
            try:
                image_paths, image_b64_pngs, visual_summary = self._prepare_lm_visual_context_sync(
                    ensure_fresh_capture=True,
                    max_images=2,
                )
                self.root.after(0, lambda s=visual_summary: self._append_log(f"Describe drawing visual context: {s}"))
                autocad_context, summary = self._collect_autocad_context_sync()
                if autocad_context:
                    self.root.after(0, lambda s=summary: self._append_log(f"Live AutoCAD context captured: {s}"))
                else:
                    self.root.after(0, lambda s=summary: self._append_log(f"Live AutoCAD context unavailable: {s}"))
                prompt = (
                    "Describe the current AutoCAD scene using both visual evidence and structured context. "
                    "Handle complex scenes by summarizing major regions, hierarchy of objects, annotations/dimensions/text, "
                    "layer intent, and probable design purpose. Note ambiguities or missing context explicitly.\n\n"
                    f"{autocad_context or 'Drawing context unavailable.'}"
                )

                def on_stream_chunk(chunk: str) -> None:
                    stream_state["chunks"] += 1
                    stream_state["parts"].append(chunk)
                    self._queue_chat_stream_text(chunk)

                stream_state["started"] = True
                self.root.after(0, self._begin_chat_stream)
                response = self._request_lm_text_feedback(
                    prompt,
                    image_paths=image_paths,
                    image_b64_pngs=image_b64_pngs,
                    system_prompt=(
                        "You are an AutoCAD LT scene analyst. Use provided images as primary evidence and "
                        "structured drawing metadata as supporting evidence. Reply in plain text only."
                    ),
                    on_token=on_stream_chunk,
                ).strip()
                if response:
                    if stream_state["chunks"] == 0:
                        self._queue_chat_stream_text(response)
                    self.root.after(0, self._end_chat_stream)
                else:
                    self.root.after(0, self._cancel_chat_stream)
                    response = "No description generated."
                    self.root.after(0, lambda r=response: self._append_chat_message("Agent", r))
                self.chat_history.append({"role": "assistant", "content": response})
                self.root.after(0, lambda: self.plan_status.set("Plan: drawing description ready"))
            except Exception as ex:
                log.exception("describe_drawing_failed")
                if stream_state["started"]:
                    if stream_state["chunks"] > 0:
                        self.root.after(0, self._end_chat_stream)
                    else:
                        self.root.after(0, self._cancel_chat_stream)
                response = f"Drawing description failed: {ex}"
                self.chat_history.append({"role": "assistant", "content": response})
                self.root.after(0, lambda r=response: self._append_chat_message("Agent", r))
                self.root.after(0, lambda e=ex: self.plan_status.set(f"Plan: description failed | {e}"))

        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def _normalize_lmstudio_base_url(raw_value: str) -> str:
        value = raw_value.strip()
        if not value:
            raise ValueError("LM Studio URL cannot be empty.")
        if "://" not in value:
            value = f"http://{value}"

        parsed = urlparse(value)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Enter a valid LM Studio host URL, for example: http://192.168.1.10:1234/v1")

        path = (parsed.path or "").rstrip("/")
        if path.endswith("/chat/completions"):
            path = path[: -len("/chat/completions")]
        elif path.endswith("/models"):
            path = path[: -len("/models")]
        if not path:
            path = "/v1"
        elif not path.endswith("/v1"):
            path = f"{path}/v1"

        return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))

    def apply_lmstudio_url(self) -> None:
        try:
            normalized_url = self._normalize_lmstudio_base_url(self.lm_base_url_var.get())
        except ValueError as ex:
            log.warning("lmstudio_url_invalid", value=self.lm_base_url_var.get(), error=str(ex))
            messagebox.showerror("Invalid LM Studio URL", str(ex))
            return

        previous_client = self.lm_client
        previous_cfg = previous_client.config
        self.lm_client = LMStudioClient(
            LMStudioConfig(
                base_url=normalized_url,
                model=previous_cfg.model,
                timeout_seconds=previous_cfg.timeout_seconds,
                timeout_retry_count=previous_cfg.timeout_retry_count,
                timeout_retry_backoff_seconds=previous_cfg.timeout_retry_backoff_seconds,
                temperature=previous_cfg.temperature,
            )
        )
        self.planner = ActionPlanner(self.lm_client)
        self.lm_base_url_var.set(normalized_url)
        try:
            previous_client.close()
        except Exception:
            pass

        log.info("lmstudio_url_applied", url=normalized_url)

        self.lm_status.set(f"LM Studio: checking {normalized_url} ...")
        self._append_log(f"LM Studio URL set to: {normalized_url}")
        self.check_connections()

    def _set_plan_output(self, data: dict) -> None:
        self.plan_output.delete("1.0", "end")
        self.plan_output.insert("1.0", json.dumps(data, indent=2))

    def _refresh_visual_status(self) -> None:
        file_count = len(self.reference_image_paths)
        capture_count = len(self.captured_canvas_b64s)
        total = file_count + capture_count
        self.visual_context_status.set(
            f"Visual context: {total} image(s) "
            f"(files={file_count}, canvas_captures={capture_count})"
        )
        self._render_thumbnail_panel()

    def _render_thumbnail_panel(self) -> None:
        for child in self.thumb_inner.winfo_children():
            child.destroy()
        self._thumbnail_images.clear()

        entries: list[tuple[str, str]] = []
        for idx, path in enumerate(self.reference_image_paths):
            entries.append(("file", path))
        for idx, _b64 in enumerate(self.captured_canvas_b64s):
            entries.append(("capture", f"Canvas {idx + 1}"))

        if not entries:
            ttk.Label(
                self.thumb_inner,
                text="No visual references added.",
                style="Caption.TLabel",
            ).pack(anchor="w", padx=8, pady=8)
            return

        for idx, (kind, label) in enumerate(entries):
            item = ttk.Frame(self.thumb_inner, style="Card.TFrame", padding=(6, 6))
            item.grid(row=0, column=idx, padx=6, pady=6, sticky="n")

            image = self._load_thumbnail_image(kind, idx, label)
            photo = ImageTk.PhotoImage(image)
            self._thumbnail_images.append(photo)

            ttk.Label(item, image=photo).pack()
            caption = label if kind == "capture" else Path(label).name
            ttk.Label(item, text=caption, width=18, style="Caption.TLabel").pack()

    def _load_thumbnail_image(self, kind: str, index: int, label: str) -> Image.Image:
        try:
            if kind == "file":
                img = Image.open(label)
            else:
                raw = base64.b64decode(self.captured_canvas_b64s[index])
                img = Image.open(io.BytesIO(raw))
            img = img.convert("RGB")
            img.thumbnail((160, 110))
            return img
        except Exception:
            placeholder = Image.new("RGB", (160, 110), (30, 40, 62))
            return placeholder

    def add_reference_images(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Select reference screenshots",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        for path in selected:
            if path not in self.reference_image_paths:
                self.reference_image_paths.append(path)
        self._refresh_visual_status()
        self._append_log(f"Added {len(selected)} reference image(s).")

    def clear_visual_context(self) -> None:
        self.reference_image_paths.clear()
        self.captured_canvas_b64s.clear()
        self._refresh_visual_status()
        self._append_log("Cleared visual context images.")

    def capture_canvas_screenshot(self) -> None:
        self._append_log("Capturing AutoCAD canvas screenshot...")

        def worker() -> None:
            ok, message = self._capture_canvas_snapshot_sync()
            if ok:
                self.root.after(0, self._refresh_visual_status)
                self.root.after(0, lambda: self._append_log(message))
            else:
                self.root.after(0, lambda: self._append_log(f"Capture failed: {message}"))

        threading.Thread(target=worker, daemon=True).start()

    def _collect_autocad_context_sync(self) -> tuple[str | None, str]:
        """Collect compact live drawing state to ground planning decisions."""
        try:
            init_result = asyncio.run(self.backend.initialize())
            if not init_result.ok:
                return None, init_result.error or "backend init failed"

            info_result = asyncio.run(self.backend.drawing_info())
            vars_result = asyncio.run(
                self.backend.drawing_get_variables(
                    [
                        "DWGNAME",
                        "CTAB",
                        "TILEMODE",
                        "CVPORT",
                        "CLAYER",
                        "LTSCALE",
                        "CANNOSCALE",
                        "UCSNAME",
                    ]
                )
            )
            count_result = asyncio.run(self.backend.entity_count())
            entities_result = asyncio.run(self.backend.entity_list())

            info_payload = info_result.payload if info_result.ok and isinstance(info_result.payload, dict) else {}
            vars_payload = vars_result.payload if vars_result.ok and isinstance(vars_result.payload, dict) else {}
            count_payload = count_result.payload if count_result.ok and isinstance(count_result.payload, dict) else {}
            entities_payload = entities_result.payload if entities_result.ok and isinstance(entities_result.payload, dict) else {}

            total_count = self._extract_count_value(count_payload.get("count"))
            if total_count is None:
                total_count = self._extract_entity_count(info_payload)

            cached_graph = self.scene_graph_cache.maybe_get(total_count)
            if cached_graph is None:
                scene_graph = asyncio.run(
                    self.scene_graph_builder.build(
                        detail_budget=140 if self._is_fast_planning_mode() else 220,
                        sample_budget=180,
                        text_sample_budget=24,
                    )
                )
                self.scene_graph_cache.put(scene_graph, total_count)
            else:
                scene_graph = cached_graph
            scene_payload = scene_graph.to_planner_context_payload()

            context_lines = [
                "drawing_info=" + json.dumps(info_payload, ensure_ascii=False),
                "variables=" + json.dumps(vars_payload, ensure_ascii=False),
                "entity_count=" + (str(total_count) if total_count is not None else "unknown"),
                "entity_type_counts=" + json.dumps(scene_payload.get("entity_type_counts", {}), ensure_ascii=False),
                "entity_layer_counts=" + json.dumps(scene_payload.get("entity_layer_counts", {}), ensure_ascii=False),
                "text_entity_sample=" + json.dumps(scene_payload.get("text_entity_sample", []), ensure_ascii=False),
                "scene_bounds=" + json.dumps(scene_payload.get("scene_bounds"), ensure_ascii=False),
                "scene_region_counts=" + json.dumps(scene_payload.get("scene_region_counts", {}), ensure_ascii=False),
                "entity_sample=" + json.dumps(scene_payload.get("entity_sample", []), ensure_ascii=False),
                "scene_graph_compact=" + json.dumps(scene_payload, ensure_ascii=False),
            ]
            entities_raw = entities_payload.get("entities")
            entities = entities_raw if isinstance(entities_raw, list) else []
            if scene_payload.get("entity_sample_truncated") and entities:
                context_lines.append(
                    f"entity_sample_truncated=true ({len(scene_payload.get('entity_sample', []))}/{len(entities)})"
                )

            summary = (
                f"dwg={vars_payload.get('DWGNAME', 'unknown')}, "
                f"tab={vars_payload.get('CTAB', 'unknown')}, "
                f"layer={vars_payload.get('CLAYER', 'unknown')}, "
                f"entities={(total_count if total_count is not None else 'unknown')}, "
                f"regions={len(scene_payload.get('scene_region_counts', {}))}"
            )
            return "\n".join(context_lines), summary
        except Exception as ex:
            log.exception("live_autocad_context_collection_failed")
            return None, str(ex)

    def _capture_canvas_snapshot_sync(self) -> tuple[bool, str]:
        try:
            init_result = asyncio.run(self.backend.initialize())
            if not init_result.ok:
                return False, init_result.error or "Backend initialization failed."
            screenshot_result = asyncio.run(self.backend.get_screenshot())
            if screenshot_result.ok and isinstance(screenshot_result.payload, str):
                self.captured_canvas_b64s.append(screenshot_result.payload)
                if len(self.captured_canvas_b64s) > 16:
                    self.captured_canvas_b64s = self.captured_canvas_b64s[-16:]
                return True, "Captured AutoCAD canvas screenshot and added to visual context."
            return False, screenshot_result.error or "AutoCAD screenshot capture failed."
        except Exception as ex:
            log.exception("capture_canvas_snapshot_failed")
            return False, str(ex)

    def check_connections(self) -> None:
        def worker() -> None:
            lm_ok, lm_msg = self.lm_client.health()
            self.root.after(
                0,
                lambda: self.lm_status.set(f"LM Studio: {'OK' if lm_ok else 'ERROR'} | {lm_msg}"),
            )

            try:
                backend_result = asyncio.run(self.backend.initialize())
                if backend_result.ok:
                    text = f"AutoCAD LT: OK | backend={self.backend.name}"
                else:
                    text = f"AutoCAD LT: ERROR | {backend_result.error}"
            except Exception as ex:  # pragma: no cover - runtime integration path
                log.exception("autocad_check_connections_failed")
                text = f"AutoCAD LT: ERROR | {ex}"

            self.root.after(0, lambda: self.cad_status.set(text))

        threading.Thread(target=worker, daemon=True).start()

    def generate_plan(self) -> None:
        prompt = self.prompt_input.get("1.0", "end").strip()
        if not prompt:
            self.plan_status.set("Plan: prompt is empty")
            return
        self.plan_status.set("Plan: generating...")
        self._clear_planning_stream()

        def worker() -> None:
            planning_state: dict[str, Any] = {"chunks": 0, "stream_closed": False, "active_attempt": 0}
            try:
                plan_max_images, _ = self._planning_visual_image_limits()
                autocad_context: str | None = None
                if self.use_live_autocad_context.get():
                    autocad_context, summary = self._collect_autocad_context_sync()
                    if autocad_context:
                        self.root.after(0, lambda s=summary: self._append_log(f"Live AutoCAD context captured: {s}"))
                    else:
                        self.root.after(0, lambda s=summary: self._append_log(f"Live AutoCAD context unavailable: {s}"))
                if self.auto_capture_before_plan.get():
                    ok, message = self._capture_canvas_snapshot_sync()
                    if ok:
                        self.root.after(0, self._refresh_visual_status)
                        self.root.after(0, lambda: self._append_log(f"Auto-capture: {message}"))
                    else:
                        self.root.after(0, lambda: self._append_log(f"Auto-capture failed: {message}"))

                image_paths, image_b64_pngs, visual_summary = self._prepare_lm_visual_context_sync(
                    ensure_fresh_capture=False,
                    max_images=plan_max_images,
                )
                self.root.after(0, lambda s=visual_summary: self._append_log(f"Plan visual context: {s}"))
                image_count = len(image_paths) + len(image_b64_pngs)
                self.root.after(0, lambda c=image_count: self.plan_status.set(f"Plan: generating... (images={c})"))

                def on_planning_token(token: str) -> None:
                    if planning_state["stream_closed"]:
                        return
                    planning_state["chunks"] += 1
                    attempt_id = planning_state["active_attempt"]
                    self._queue_planning_stream_text(
                        token,
                        can_accept=lambda aid=attempt_id: (
                            (not planning_state["stream_closed"]) and planning_state["active_attempt"] == aid
                        ),
                    )

                def on_attempt_start(mode: str) -> None:
                    planning_state["active_attempt"] += 1
                    self.root.after(0, lambda m=mode: self._begin_planning_stream(f"Planning attempt: {m}"))
                result, planning_mode = self._create_plan_with_resilience(
                    prompt,
                    autocad_context=autocad_context,
                    image_paths=image_paths,
                    image_b64_pngs=image_b64_pngs,
                    on_planning_token=on_planning_token,
                    on_attempt_start=on_attempt_start,
                )
                if planning_state["chunks"] == 0:
                    self.root.after(
                        0,
                        lambda: self._append_planning_stream(
                            "Model stream unavailable for this attempt; using non-stream completion.\n"
                        ),
                    )
                planning_state["stream_closed"] = True
                self.root.after(
                    0,
                    lambda m=planning_mode: self._end_planning_stream(f"Plan parsed successfully ({m})."),
                )
                self.current_plan = result.plan
                self.root.after(0, lambda: self._set_plan_output(result.plan))
                if planning_mode != "vision_full":
                    self.root.after(0, lambda m=planning_mode: self._append_log(f"Planner fallback mode used: {m}"))
                if result.safety.ok:
                    self.root.after(0, lambda: self.plan_status.set("Plan: ready (safety checks passed)"))
                else:
                    err_text = "; ".join(result.safety.errors)
                    self.root.after(0, lambda: self.plan_status.set(f"Plan: blocked by safety checks | {err_text}"))
            except Exception as ex:
                log.exception("plan_generation_failed")
                error_text = f"Plan: generation failed | {ex}"
                planning_state["stream_closed"] = True
                if planning_state["chunks"] > 0:
                    self.root.after(0, lambda e=error_text: self._end_planning_stream(e))
                else:
                    self.root.after(0, lambda e=error_text: self._cancel_planning_stream(e))
                self.root.after(0, lambda t=error_text: self.plan_status.set(t))

        threading.Thread(target=worker, daemon=True).start()
    def send_chat_message(self) -> None:
        message = self.chat_input.get("1.0", "end").strip()
        if not message:
            return

        self.chat_input.delete("1.0", "end")
        self.chat_history.append({"role": "user", "content": message})
        self._append_chat_message("User", message)
        self._append_chat_thinking("Collecting context and generating a plan...")
        self.plan_status.set("Plan: generating from chat...")
        self._clear_planning_stream()

        def worker() -> None:
            try:
                plan_max_images, feedback_max_images = self._planning_visual_image_limits()
                autocad_context: str | None = None
                if self.use_live_autocad_context.get():
                    self.root.after(0, lambda: self._append_chat_thinking("Reading live AutoCAD context..."))
                    autocad_context, summary = self._collect_autocad_context_sync()
                    if autocad_context:
                        self.root.after(0, lambda s=summary: self._append_log(f"Live AutoCAD context captured: {s}"))
                    else:
                        self.root.after(0, lambda s=summary: self._append_log(f"Live AutoCAD context unavailable: {s}"))

                image_paths, image_b64_pngs, visual_summary = self._prepare_lm_visual_context_sync(
                    ensure_fresh_capture=True,
                    max_images=plan_max_images,
                )
                self.root.after(0, lambda s=visual_summary: self._append_log(f"Chat visual context: {s}"))
                self.root.after(0, lambda s=visual_summary: self._append_chat_thinking(f"Visual context ready: {s}"))
                chat_prompt = self._build_chat_planner_prompt(message)
                self.root.after(0, lambda: self._append_chat_thinking("Requesting plan from model..."))
                planning_state: dict[str, Any] = {"chunks": 0, "stream_closed": False, "active_attempt": 0}

                def on_planning_token(token: str) -> None:
                    if planning_state["stream_closed"]:
                        return
                    planning_state["chunks"] += 1
                    attempt_id = planning_state["active_attempt"]
                    self._queue_planning_stream_text(
                        token,
                        can_accept=lambda aid=attempt_id: (
                            (not planning_state["stream_closed"]) and planning_state["active_attempt"] == aid
                        ),
                    )

                def on_attempt_start(mode: str) -> None:
                    planning_state["active_attempt"] += 1
                    self.root.after(0, lambda m=mode: self._begin_planning_stream(f"Planning attempt: {m}"))
                result, planning_mode = self._create_plan_with_resilience(
                    chat_prompt,
                    autocad_context=autocad_context,
                    image_paths=image_paths,
                    image_b64_pngs=image_b64_pngs,
                    on_planning_token=on_planning_token,
                    on_attempt_start=on_attempt_start,
                )
                if planning_state["chunks"] == 0:
                    self.root.after(
                        0,
                        lambda: self._append_planning_stream(
                            "Model stream unavailable for this attempt; using non-stream completion.\n"
                        ),
                    )
                planning_state["stream_closed"] = True
                self.root.after(
                    0,
                    lambda m=planning_mode: self._end_planning_stream(f"Plan parsed successfully ({m})."),
                )
                if planning_mode != "vision_full":
                    self.root.after(0, lambda m=planning_mode: self._append_log(f"Chat planner fallback mode used: {m}"))
                self.root.after(
                    0,
                    lambda m=planning_mode: self._append_chat_thinking(f"Plan generation mode: {m}"),
                )

                self.current_plan = result.plan
                self.root.after(0, lambda p=result.plan: self._set_plan_output(p))
                self.root.after(0, lambda p=result.plan: self._append_chat_full_output("Plan JSON", p))

                if not result.safety.ok:
                    err_text = "; ".join(result.safety.errors)
                    response = f"Blocked by safety checks: {err_text}"
                    self.chat_history.append({"role": "assistant", "content": response})
                    self.root.after(0, lambda r=response: self._append_chat_message("Agent", r))
                    self.root.after(0, lambda e=err_text: self.plan_status.set(f"Plan: blocked by safety checks | {e}"))
                    return

                actions = result.plan.get("actions", [])
                action_count = len(actions) if isinstance(actions, list) else 0
                self.root.after(
                    0,
                    lambda c=action_count: self._append_chat_thinking(f"Plan ready with {c} action(s)."),
                )
                if action_count == 0:
                    response = "No executable actions were generated."
                    self.chat_history.append({"role": "assistant", "content": response})
                    self.root.after(0, lambda r=response: self._append_chat_message("Agent", r))
                    self.root.after(0, lambda: self.plan_status.set("Plan: generated with 0 actions"))
                    return
                execution_summary: dict[str, Any] | None = None

                if self.chat_auto_execute.get():
                    self.root.after(
                        0,
                        lambda c=action_count: self._append_chat_message(
                            "Agent", f"Planned {c} action(s). Auto-executing now."
                        ),
                    )
                    self.root.after(0, lambda: self._append_chat_thinking("Executing plan actions..."))
                    execution_summary = self._execute_actions_sync(actions, source="chat")
                    self.root.after(
                        0,
                        lambda s=execution_summary: self._append_chat_full_output("Execution summary", s),
                    )
                    fallback_response = (
                        f"Applied {execution_summary.get('success_count', 0)}/{execution_summary.get('total', action_count)} "
                        "action(s) to the drawing."
                        if execution_summary.get("ok")
                        else f"Execution failed: {execution_summary.get('error', 'unknown error')}"
                    )
                else:
                    execution_summary = {
                        "ok": True,
                        "total": action_count,
                        "success_count": 0,
                        "auto_executed": False,
                    }
                    self.root.after(
                        0,
                        lambda s=execution_summary: self._append_chat_full_output("Execution summary", s),
                    )
                    fallback_response = (
                        f"Planned {action_count} action(s). Review the JSON plan, then click "
                        "\"Execute Approved Plan\" to apply changes."
                    )
                    self.root.after(0, lambda: self.plan_status.set("Plan: ready from chat"))
                if self.chat_auto_execute.get():
                    response = fallback_response
                    self.chat_history.append({"role": "assistant", "content": response})
                    self.root.after(0, lambda r=response: self._append_chat_message("Agent", r))
                    self.root.after(0, lambda: self._append_chat_thinking("Task complete."))
                    return
                latest_context = autocad_context
                if self.use_live_autocad_context.get():
                    latest_context, _ = self._collect_autocad_context_sync()
                feedback_prompt = self._build_chat_feedback_prompt(
                    latest_user_message=message,
                    plan=result.plan,
                    execution_summary=execution_summary,
                    autocad_context=latest_context,
                )
                stream_state: dict[str, Any] = {"started": False, "chunks": 0, "parts": []}
                streamed_rendered = False
                response = fallback_response
                try:
                    self.root.after(0, lambda: self._append_chat_thinking("Generating final response..."))
                    feedback_image_paths, feedback_image_b64_pngs, feedback_visual_summary = (
                        self._prepare_lm_visual_context_sync(
                            ensure_fresh_capture=self.chat_auto_execute.get(),
                            max_images=feedback_max_images,
                        )
                    )
                    self.root.after(
                        0,
                        lambda s=feedback_visual_summary: self._append_log(f"Chat feedback visual context: {s}"),
                    )

                    def on_stream_chunk(chunk: str) -> None:
                        stream_state["chunks"] += 1
                        stream_state["parts"].append(chunk)
                        self._queue_chat_stream_text(chunk)

                    stream_state["started"] = True
                    self.root.after(0, self._begin_chat_stream)
                    response = self._request_lm_text_feedback(
                        feedback_prompt,
                        image_paths=feedback_image_paths,
                        image_b64_pngs=feedback_image_b64_pngs,
                        on_token=on_stream_chunk,
                    ).strip()
                    if response:
                        if stream_state["chunks"] == 0:
                            self._queue_chat_stream_text(response)
                        self.root.after(0, self._end_chat_stream)
                        streamed_rendered = True
                    else:
                        self.root.after(0, self._cancel_chat_stream)
                        response = fallback_response
                except Exception as ex:
                    log.exception("chat_text_feedback_failed")
                    self.root.after(
                        0,
                        lambda e=str(ex): self._append_chat_thinking(
                            f"Feedback response unavailable ({e}); showing execution summary instead."
                        ),
                    )
                    if stream_state["started"]:
                        if stream_state["chunks"] > 0:
                            self.root.after(0, self._end_chat_stream)
                            partial = "".join(stream_state["parts"]).strip()
                            response = partial or fallback_response
                            streamed_rendered = True
                        else:
                            self.root.after(0, self._cancel_chat_stream)
                            response = fallback_response
                    else:
                        response = fallback_response
                if not response.strip():
                    response = fallback_response
                self.chat_history.append({"role": "assistant", "content": response})
                if not streamed_rendered:
                    self.root.after(0, lambda r=response: self._append_chat_message("Agent", r))
            except Exception as ex:
                log.exception("chat_message_processing_failed")
                response = f"Chat processing failed: {ex}"
                self.chat_history.append({"role": "assistant", "content": response})
                self.root.after(
                    0,
                    lambda e=str(ex): self._append_chat_full_output("Chat failure details", {"error": e}),
                )
                self.root.after(0, lambda r=response: self._append_chat_message("Agent", r))
                self.root.after(0, lambda e=ex: self.plan_status.set(f"Plan: generation failed | {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def execute_plan(self) -> None:
        if not self.current_plan:
            messagebox.showwarning("No plan", "Generate a plan first.")
            return

        actions = self.current_plan.get("actions", [])
        if not isinstance(actions, list) or not actions:
            messagebox.showwarning("No actions", "Current plan has no actions to execute.")
            return

        proceed = messagebox.askyesno(
            "Confirm execution",
            f"Execute {len(actions)} approved action(s) in AutoCAD LT?",
        )
        if not proceed:
            return

        def worker() -> None:
            self._execute_actions_sync(actions, source="manual")

        threading.Thread(target=worker, daemon=True).start()

    def _execute_actions_sync(self, actions: list[dict], source: str = "manual") -> dict[str, Any]:
        before_count: int | None = None
        after_count: int | None = None
        success_count = 0
        created_or_modified_geometry = False
        loop_guard_triggered = False
        verification_failures: list[dict[str, Any]] = []
        repair_attempts = 0
        successful_repairs = 0
        original_total_actions = len(actions)
        max_actions_to_execute = 30
        repeated_action_limit = 4
        max_repair_attempts_per_action = 2
        if original_total_actions > max_actions_to_execute:
            actions = actions[:max_actions_to_execute]
            self.root.after(
                0,
                lambda o=original_total_actions, m=max_actions_to_execute: self._append_log(
                    f"[guard] Plan contains {o} actions; executing first {m} to prevent long/looping runs."
                ),
            )
        total_actions = len(actions)

        try:
            init_result = asyncio.run(self.backend.initialize())
            if not init_result.ok:
                self.root.after(
                    0,
                    lambda e=init_result.error: self._append_log(f"[init] AutoCAD backend init failed: {e}"),
                )
                self.root.after(
                    0,
                    lambda e=init_result.error: self.cad_status.set(f"AutoCAD LT: ERROR | {e}"),
                )
                return {"ok": False, "error": init_result.error or "backend init failed", "total": total_actions, "success_count": 0}
            self.root.after(0, lambda: self.cad_status.set(f"AutoCAD LT: OK | backend={self.backend.name}"))
            before_info = asyncio.run(self.backend.drawing_info())
            if before_info.ok and isinstance(before_info.payload, dict):
                before_count = self._extract_entity_count(before_info.payload)
        except Exception as ex:
            log.exception("autocad_backend_init_failed")
            self.root.after(0, lambda e=ex: self._append_log(f"[init] AutoCAD backend init error: {e}"))
            self.root.after(0, lambda e=ex: self.cad_status.set(f"AutoCAD LT: ERROR | {e}"))
            return {"ok": False, "error": str(ex), "total": total_actions, "success_count": 0}
        previous_action_signature: str | None = None
        repeated_action_count = 0

        for idx, action in enumerate(actions):
            action_signature = json.dumps(action, sort_keys=True, default=str)
            if action_signature == previous_action_signature:
                repeated_action_count += 1
            else:
                previous_action_signature = action_signature
                repeated_action_count = 1

            if repeated_action_count > repeated_action_limit:
                loop_guard_triggered = True
                self.root.after(
                    0,
                    lambda i=idx: self._append_log(
                        f"[guard] Detected repetitive action loop near step {i}; stopping remaining execution."
                    ),
                )
                break
            try:
                expected_effects = self._extract_expected_effects_from_action(action)
                pre_state = self._collect_verification_state_sync() if expected_effects else None
                result = asyncio.run(self._execute_action(action))
                payload = result.to_dict()
                self.root.after(0, lambda i=idx, p=payload: self._append_log(f"[{i}] {json.dumps(p)}"))
                if result.ok:
                    success_count += 1
                    tool = str(action.get("tool", ""))
                    operation = str(action.get("operation", ""))
                    if tool in ("entity", "annotation", "block") or (
                        tool == "drawing" and operation in ("create",)
                    ):
                        created_or_modified_geometry = True
                    if expected_effects:
                        post_state = self._collect_verification_state_sync()
                        verification = self._verify_expected_effects_sync(
                            action=action,
                            result_payload=payload,
                            expected_effects=expected_effects,
                            before_state=pre_state,
                            after_state=post_state,
                        )
                        if not verification.get("ok", False):
                            verification_errors = verification.get("errors", [])
                            self.root.after(
                                0,
                                lambda i=idx, ve=verification_errors: self._append_log(
                                    f"[verify:{i}] expected effects failed: {'; '.join(ve)}"
                                ),
                            )
                            repaired = False
                            for repair_idx in range(max_repair_attempts_per_action):
                                repair_attempts += 1
                                repair_summary = self._attempt_repair_for_failed_effects_sync(
                                    failed_action=action,
                                    failed_action_result=payload,
                                    verification=verification,
                                    attempt_index=repair_idx + 1,
                                )
                                if repair_summary.get("ok"):
                                    post_state = self._collect_verification_state_sync()
                                    verification = self._verify_expected_effects_sync(
                                        action=action,
                                        result_payload=payload,
                                        expected_effects=expected_effects,
                                        before_state=pre_state,
                                        after_state=post_state,
                                    )
                                    if verification.get("ok", False):
                                        repaired = True
                                        successful_repairs += 1
                                        self.root.after(
                                            0,
                                            lambda i=idx, a=repair_idx + 1: self._append_log(
                                                f"[verify:{i}] repaired successfully on attempt {a}."
                                            ),
                                        )
                                        break
                            if not repaired:
                                verification_failures.append(
                                    {
                                        "action_index": idx,
                                        "action": action,
                                        "errors": verification.get("errors", []),
                                        "observed": verification.get("observed", {}),
                                    }
                                )
                if (not result.ok) and isinstance(result.error, str) and ("Timeout waiting for result" in result.error):
                    hint = (
                        'Recovery: In AutoCAD LT press ESC twice, ensure `(load "mcp_dispatch.lsp")` is loaded, '
                        "and verify `*mcp-ipc-dir*` in LISP matches `C:/temp/`, then retry."
                    )
                    self.root.after(0, lambda i=idx, h=hint: self._append_log(f"[{i}] HINT: {h}"))
            except Exception as ex:
                log.exception("plan_action_execution_failed", action_index=idx, action=action)
                self.root.after(0, lambda i=idx, e=ex: self._append_log(f"[{i}] ERROR: {e}"))

        if success_count > 0 and created_or_modified_geometry and self.auto_zoom_after_execute.get():
            try:
                zoom_result = asyncio.run(self.backend.zoom_extents())
                self.root.after(
                    0,
                    lambda p=zoom_result.to_dict(): self._append_log(
                        f"[post] auto zoom extents: {json.dumps(p)}"
                    ),
                )
            except Exception as ex:
                log.exception("auto_zoom_extents_failed")
                self.root.after(0, lambda e=ex: self._append_log(f"[post] auto zoom extents ERROR: {e}"))

        try:
            after_info = asyncio.run(self.backend.drawing_info())
            if after_info.ok and isinstance(after_info.payload, dict):
                after_count = self._extract_entity_count(after_info.payload)
            if before_count is not None and after_count is not None:
                delta = after_count - before_count
                self.root.after(
                    0,
                    lambda b=before_count, a=after_count, d=delta: self._append_log(
                        f"[post] drawing entities before={b}, after={a}, delta={d}"
                    ),
                )
        except Exception as ex:
            log.exception("post_execution_drawing_info_failed")
            self.root.after(0, lambda e=ex: self._append_log(f"[post] drawing info ERROR: {e}"))

        self.root.after(
            0,
            lambda s=source, ok=success_count, total=total_actions, o=original_total_actions, lg=loop_guard_triggered, vf=len(verification_failures), ra=repair_attempts, sr=successful_repairs: self._append_log(
                f"[post] execution summary ({s}): success={ok}/{total}, planned={o}, loop_guard={lg}, verification_failures={vf}, repairs={sr}/{ra}"
            ),
        )
        if success_count > 0:
            self.scene_graph_cache.invalidate()
        return {
            "ok": True,
            "total": total_actions,
            "planned_total": original_total_actions,
            "success_count": success_count,
            "before_count": before_count,
            "after_count": after_count,
            "loop_guard_triggered": loop_guard_triggered,
            "verification_failures": verification_failures,
            "repair_attempts": repair_attempts,
            "successful_repairs": successful_repairs,
        }

    def _extract_expected_effects_from_action(self, action: dict[str, Any]) -> dict[str, Any] | None:
        expected = action.get("expected_effects")
        if isinstance(expected, dict):
            return expected
        data = action.get("data")
        if isinstance(data, dict):
            nested = data.get("expected_effects")
            if isinstance(nested, dict):
                return nested
        tool, operation = self._normalize_action_aliases(action.get("tool"), action.get("operation"))
        return self._derive_default_expected_effects(tool, operation)

    @staticmethod
    def _derive_default_expected_effects(tool: Any, operation: Any) -> dict[str, Any] | None:
        if not isinstance(tool, str) or not isinstance(operation, str):
            return None
        if tool == "entity" and operation.startswith("create_"):
            entity_type = operation.removeprefix("create_").replace("_", "").upper()
            if entity_type == "MTEXT":
                entity_type = "MTEXT"
            return {"entity_count_delta": 1, "entity_type_created": entity_type}
        if tool == "entity" and operation in {"copy", "offset", "array", "mirror"}:
            return {"min_entity_count_delta": 1}
        if tool == "entity" and operation == "erase":
            return {"entity_count_delta": -1}
        return None

    def _collect_verification_state_sync(self) -> dict[str, Any]:
        state: dict[str, Any] = {"entity_count": None, "entities": []}
        try:
            count_result = asyncio.run(self.backend.entity_count())
            if count_result.ok and isinstance(count_result.payload, dict):
                state["entity_count"] = self._extract_count_value(count_result.payload.get("count"))
        except Exception as ex:
            log.exception("verification_entity_count_failed")
            self.root.after(0, lambda e=ex: self._append_log(f"[verify] entity_count probe failed: {e}"))
        try:
            list_result = asyncio.run(self.backend.entity_list())
            if list_result.ok and isinstance(list_result.payload, dict):
                entities = list_result.payload.get("entities")
                if isinstance(entities, list):
                    state["entities"] = entities
        except Exception as ex:
            log.exception("verification_entity_list_failed")
            self.root.after(0, lambda e=ex: self._append_log(f"[verify] entity_list probe failed: {e}"))
        return state

    @staticmethod
    def _handle_exists_in_state(handle: str, state: dict[str, Any]) -> bool:
        entities = state.get("entities")
        if not isinstance(entities, list):
            return False
        target = handle.strip().upper()
        for entry in entities:
            if not isinstance(entry, dict):
                continue
            candidate = str(entry.get("handle") or "").strip().upper()
            if candidate == target:
                return True
        return False

    def _verify_expected_effects_sync(
        self,
        action: dict[str, Any],
        result_payload: dict[str, Any],
        expected_effects: dict[str, Any],
        before_state: dict[str, Any] | None,
        after_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        errors: list[str] = []
        observed: dict[str, Any] = {}
        if not result_payload.get("ok"):
            errors.append("action failed")
            return {"ok": False, "errors": errors, "observed": observed}

        if not before_state or not after_state:
            return {"ok": True, "errors": [], "observed": {"skipped": "verification_state_unavailable"}}

        before_count = self._extract_count_value(before_state.get("entity_count"))
        after_count = self._extract_count_value(after_state.get("entity_count"))
        observed["before_count"] = before_count
        observed["after_count"] = after_count

        if before_count is not None and after_count is not None:
            delta = after_count - before_count
            observed["entity_count_delta"] = delta
            expected_delta = expected_effects.get("entity_count_delta")
            if isinstance(expected_delta, (int, float)) and delta != int(expected_delta):
                errors.append(f"entity_count_delta expected {int(expected_delta)} got {delta}")
            expected_min_delta = expected_effects.get("min_entity_count_delta")
            if isinstance(expected_min_delta, (int, float)) and delta < float(expected_min_delta):
                errors.append(f"entity_count_delta expected >= {float(expected_min_delta)} got {delta}")
            expected_max_delta = expected_effects.get("max_entity_count_delta")
            if isinstance(expected_max_delta, (int, float)) and delta > float(expected_max_delta):
                errors.append(f"entity_count_delta expected <= {float(expected_max_delta)} got {delta}")

        created_type = expected_effects.get("entity_type_created")
        if isinstance(created_type, str) and created_type.strip():
            entities = after_state.get("entities")
            has_type = False
            if isinstance(entities, list):
                target_type = created_type.strip().upper()
                has_type = any(
                    isinstance(entry, dict) and str(entry.get("type") or "").upper() == target_type
                    for entry in entities
                )
            observed["entity_type_created_found"] = has_type
            if not has_type:
                errors.append(f"entity_type_created {created_type} not found after execution")

        target_exists = expected_effects.get("target_exists") or expected_effects.get("entity_exists")
        if isinstance(target_exists, str) and target_exists.strip():
            handle_hint = target_exists.strip()
            if handle_hint in {"$result.handle", "result.handle"}:
                payload = result_payload.get("payload")
                if isinstance(payload, dict):
                    handle_hint = str(payload.get("handle") or "").strip()
            if not handle_hint:
                errors.append("target_exists requested but no handle available")
            else:
                exists = self._handle_exists_in_state(handle_hint, after_state)
                observed["target_exists"] = {handle_hint: exists}
                if not exists:
                    errors.append(f"target handle '{handle_hint}' not found after execution")

        return {"ok": len(errors) == 0, "errors": errors, "observed": observed}

    def _attempt_repair_for_failed_effects_sync(
        self,
        failed_action: dict[str, Any],
        failed_action_result: dict[str, Any],
        verification: dict[str, Any],
        attempt_index: int,
    ) -> dict[str, Any]:
        autocad_context, context_summary = self._collect_autocad_context_sync()
        self.root.after(
            0,
            lambda a=attempt_index, s=context_summary: self._append_log(
                f"[repair:{a}] preparing corrective mini-plan ({s})"
            ),
        )
        repair_prompt = (
            "Generate a minimal corrective mini-plan (<=3 actions) that satisfies expected effects "
            "for the failed action. Keep operations safe and explicit.\n\n"
            f"Failed action:\n{json.dumps(failed_action, ensure_ascii=False)}\n\n"
            f"Execution result:\n{json.dumps(failed_action_result, ensure_ascii=False)}\n\n"
            f"Verification failure:\n{json.dumps(verification, ensure_ascii=False)}\n\n"
            "Return corrective actions only."
        )
        try:
            plan_result = self.planner.create_plan(
                repair_prompt,
                autocad_context=autocad_context,
                backend_name=self.backend.name,
            )
        except Exception as ex:
            log.exception("repair_plan_generation_failed")
            self.root.after(0, lambda e=ex, a=attempt_index: self._append_log(f"[repair:{a}] plan generation failed: {e}"))
            return {"ok": False, "error": str(ex)}

        if not plan_result.safety.ok:
            err_text = "; ".join(plan_result.safety.errors)
            self.root.after(
                0,
                lambda a=attempt_index, e=err_text: self._append_log(f"[repair:{a}] blocked by safety: {e}"),
            )
            return {"ok": False, "error": err_text}

        repair_actions = plan_result.plan.get("actions")
        if not isinstance(repair_actions, list) or not repair_actions:
            self.root.after(0, lambda a=attempt_index: self._append_log(f"[repair:{a}] no corrective actions generated."))
            return {"ok": False, "error": "no corrective actions"}

        repair_actions = repair_actions[:3]
        success = 0
        for ridx, repair_action in enumerate(repair_actions):
            try:
                repair_result = asyncio.run(self._execute_action(repair_action))
                payload = repair_result.to_dict()
                self.root.after(
                    0,
                    lambda a=attempt_index, i=ridx, p=payload: self._append_log(
                        f"[repair:{a}.{i}] {json.dumps(p)}"
                    ),
                )
                if repair_result.ok:
                    success += 1
            except Exception as ex:
                log.exception("repair_action_execution_failed", repair_index=ridx)
                self.root.after(
                    0,
                    lambda a=attempt_index, i=ridx, e=ex: self._append_log(
                        f"[repair:{a}.{i}] ERROR: {e}"
                    ),
                )

        if success > 0:
            self.scene_graph_cache.invalidate()
        return {"ok": success > 0, "success_count": success, "total": len(repair_actions)}
    async def _execute_action(self, action: dict):
        tool = action.get("tool")
        operation = action.get("operation")
        data = (
            action.get("data")
            or action.get("params")
            or action.get("arguments")
            or {}
        )
        if not isinstance(data, dict):
            raise ValueError(f"Action payload must be an object for {tool}.{operation}")
        tool, operation = self._normalize_action_aliases(tool, operation)

        if tool == "entity" and operation == "create_line":
            x1, y1, x2, y2, layer = self._extract_line_args(data, action)
            return await self.backend.create_line(
                x1, y1, x2, y2, layer
            )
        if tool == "entity" and operation == "create_circle":
            try:
                cx, cy, radius, layer = self._extract_circle_args(data, action)
                return await self.backend.create_circle(
                    cx, cy, radius, layer
                )
            except ValueError:
                placement = await self._resolve_placement_point(data, action=action, prefer_scene_center=False)
                radius = self._extract_float_value(data, keys=("radius", "r"), default=None)
                if radius is None:
                    diameter = self._extract_float_value(data, keys=("diameter", "d"), default=None)
                    if diameter is not None:
                        radius = diameter / 2.0
                if not placement or radius is None:
                    return CommandResult(
                        ok=False,
                        error=(
                            "Unable to resolve circle placement. Provide {cx,cy,radius} or "
                            "a spatial hint (e.g. target_area/location) plus radius."
                        ),
                    )
                layer = self._extract_string_value(data, keys=("layer", "target_layer", "entity_layer"), default=None)
                return await self.backend.create_circle(float(placement[0]), float(placement[1]), float(radius), layer)
        if tool == "entity" and operation == "create_rectangle":
            try:
                x1, y1, x2, y2, layer = self._extract_rectangle_args(data, action)
                return await self.backend.create_rectangle(x1, y1, x2, y2, layer)
            except ValueError:
                placement = await self._resolve_placement_point(data, action=action, prefer_scene_center=False)
                width = self._extract_float_value(data, keys=("width", "w"), default=None)
                height = self._extract_float_value(data, keys=("height", "h"), default=None)
                if not placement or width is None or height is None:
                    return CommandResult(
                        ok=False,
                        error=(
                            "Unable to resolve rectangle placement. Provide corner coordinates or "
                            "a spatial hint with width/height."
                        ),
                    )
                x1, y1 = placement
                x2 = x1 + float(width)
                y2 = y1 + float(height)
                layer = self._extract_string_value(data, keys=("layer", "target_layer", "entity_layer"), default=None)
                return await self.backend.create_rectangle(float(x1), float(y1), float(x2), float(y2), layer)
        if tool == "entity" and operation == "create_polyline":
            points = self._extract_points_list(
                data,
                keys=("points", "vertices", "polyline_points", "path"),
            )
            if len(points) < 2:
                return CommandResult(
                    ok=False,
                    error="create_polyline requires at least two points.",
                )
            closed = self._extract_bool_value(data, keys=("closed", "is_closed"), default=False)
            layer = self._extract_string_value(data, keys=("layer", "target_layer", "entity_layer"), default=None)
            return await self.backend.create_polyline(points, bool(closed), layer)
        if tool == "entity" and operation == "create_arc":
            placement = await self._resolve_placement_point(data, action=action, prefer_scene_center=False)
            cx = self._extract_float_value(data, keys=("cx", "x", "center_x"), default=placement[0] if placement else None)
            cy = self._extract_float_value(data, keys=("cy", "y", "center_y"), default=placement[1] if placement else None)
            radius = self._extract_float_value(data, keys=("radius", "r"), default=None)
            if radius is None:
                diameter = self._extract_float_value(data, keys=("diameter", "d"), default=None)
                if diameter is not None:
                    radius = diameter / 2.0
            start_angle = self._extract_float_value(
                data,
                keys=("start_angle", "angle_start", "from_angle", "start"),
                default=0.0,
            )
            end_angle = self._extract_float_value(
                data,
                keys=("end_angle", "angle_end", "to_angle", "end"),
                default=None,
            )
            if end_angle is None:
                sweep_angle = self._extract_float_value(data, keys=("sweep_angle", "delta_angle"), default=None)
                if sweep_angle is not None:
                    end_angle = float(start_angle if start_angle is not None else 0.0) + float(sweep_angle)
            if cx is None or cy is None or radius is None or end_angle is None:
                return CommandResult(
                    ok=False,
                    error=(
                        "create_arc requires center, radius, start_angle, and end_angle "
                        "(or sweep_angle)."
                    ),
                )
            layer = self._extract_string_value(data, keys=("layer", "target_layer", "entity_layer"), default=None)
            return await self.backend.create_arc(
                float(cx),
                float(cy),
                float(radius),
                float(start_angle if start_angle is not None else 0.0),
                float(end_angle),
                layer,
            )
        if tool == "entity" and operation == "create_ellipse":
            placement = await self._resolve_placement_point(data, action=action, prefer_scene_center=False)
            cx = self._extract_float_value(data, keys=("cx", "x", "center_x"), default=placement[0] if placement else None)
            cy = self._extract_float_value(data, keys=("cy", "y", "center_y"), default=placement[1] if placement else None)
            major_x = self._extract_float_value(data, keys=("major_x", "axis_x"), default=None)
            major_y = self._extract_float_value(data, keys=("major_y", "axis_y"), default=None)
            if major_x is None or major_y is None:
                major_axis = self._parse_point(data.get("major_axis"))
                if major_axis is not None:
                    major_x = float(major_axis[0])
                    major_y = float(major_axis[1])
            ratio = self._extract_float_value(data, keys=("ratio", "minor_major_ratio"), default=0.5)
            if cx is None or cy is None or major_x is None or major_y is None or ratio is None:
                return CommandResult(
                    ok=False,
                    error="create_ellipse requires center, major_x, major_y, and ratio.",
                )
            layer = self._extract_string_value(data, keys=("layer", "target_layer", "entity_layer"), default=None)
            return await self.backend.create_ellipse(
                float(cx),
                float(cy),
                float(major_x),
                float(major_y),
                float(ratio),
                layer,
            )
        if tool == "entity" and operation == "create_mtext":
            text_value = data.get("text") or data.get("content") or data.get("value") or data.get("label")
            if not isinstance(text_value, str) or not text_value.strip():
                return CommandResult(ok=False, error="create_mtext requires a non-empty text value.")
            placement = await self._resolve_placement_point(data, action=action, prefer_scene_center=True)
            width = self._extract_float_value(data, keys=("width", "w"), default=40.0)
            height = self._extract_float_value(data, keys=("height", "text_height"), default=2.5)
            if not placement or width is None or height is None:
                return CommandResult(
                    ok=False,
                    error="create_mtext requires placement and width.",
                )
            layer = self._extract_string_value(data, keys=("layer", "target_layer", "entity_layer"), default=None)
            return await self.backend.create_mtext(
                float(placement[0]),
                float(placement[1]),
                float(width),
                text_value.strip(),
                float(height),
                layer,
            )
        if tool == "entity" and operation == "create_hatch":
            entity_id = await self._resolve_entity_id(data, action, fallback_to_last=True)
            if not entity_id:
                return CommandResult(ok=False, error="Unable to resolve target entity for hatch operation.")
            pattern = self._extract_string_value(data, keys=("pattern", "hatch_pattern"), default="ANSI31")
            return await self.backend.create_hatch(entity_id, pattern or "ANSI31")
        if tool == "entity" and operation in ("get", "get_entity_properties", "entity_get_properties"):
            entity_id = await self._resolve_entity_id(data, action, fallback_to_last=True)
            if not entity_id:
                return CommandResult(ok=False, error="Unable to resolve target entity for get operation.")
            return await self.backend.entity_get(entity_id)
        if tool == "entity" and operation in ("list", "list_entities", "query_by_type"):
            return await self._query_entities(data)
        if tool == "entity" and operation in ("count", "get_entity_count"):
            return await self.backend.entity_count(data.get("layer"))
        if tool == "entity" and operation == "erase":
            entity_id = await self._resolve_entity_id(data, action, fallback_to_last=True)
            if not entity_id:
                return CommandResult(ok=False, error="Unable to resolve target entity for erase operation.")
            return await self.backend.entity_erase(entity_id)
        if tool == "entity" and operation == "copy":
            entity_id = await self._resolve_entity_id(data, action, fallback_to_last=True)
            if not entity_id:
                return CommandResult(ok=False, error="Unable to resolve target entity for copy operation.")
            displacement = self._extract_displacement(data, action=action)
            if displacement is None:
                return CommandResult(ok=False, error="copy requires a displacement (dx,dy or from/to points).")
            return await self.backend.entity_copy(entity_id, float(displacement[0]), float(displacement[1]))
        if tool == "entity" and operation == "move":
            entity_id = await self._resolve_entity_id(data, action, fallback_to_last=True)
            if not entity_id:
                return CommandResult(ok=False, error="Unable to resolve target entity for move operation.")
            displacement = self._extract_displacement(data, action=action)
            if displacement is None:
                return CommandResult(ok=False, error="move requires a displacement (dx,dy or from/to points).")
            return await self.backend.entity_move(entity_id, float(displacement[0]), float(displacement[1]))
        if tool == "entity" and operation == "rotate":
            entity_id = await self._resolve_entity_id(data, action, fallback_to_last=True)
            if not entity_id:
                return CommandResult(ok=False, error="Unable to resolve target entity for rotate operation.")
            center = await self._resolve_placement_point(data, action=action, prefer_scene_center=True)
            if center is None:
                center = (0.0, 0.0)
            angle = self._extract_float_value(data, keys=("angle", "rotation"), default=None)
            if angle is None:
                return CommandResult(ok=False, error="rotate requires an angle.")
            return await self.backend.entity_rotate(
                entity_id,
                float(center[0]),
                float(center[1]),
                float(angle),
            )
        if tool == "entity" and operation == "scale":
            entity_id = await self._resolve_entity_id(data, action, fallback_to_last=True)
            if not entity_id:
                return CommandResult(ok=False, error="Unable to resolve target entity for scale operation.")
            center = await self._resolve_placement_point(data, action=action, prefer_scene_center=True)
            if center is None:
                center = (0.0, 0.0)
            factor = self._extract_float_value(data, keys=("factor", "scale", "scale_factor"), default=None)
            if factor is None:
                return CommandResult(ok=False, error="scale requires a scale factor.")
            return await self.backend.entity_scale(
                entity_id,
                float(center[0]),
                float(center[1]),
                float(factor),
            )
        if tool == "entity" and operation == "mirror":
            entity_id = await self._resolve_entity_id(data, action, fallback_to_last=True)
            if not entity_id:
                return CommandResult(ok=False, error="Unable to resolve target entity for mirror operation.")
            line_points = self._extract_points_list(
                data,
                keys=("points", "mirror_line", "axis", "line"),
            )
            if len(line_points) >= 2:
                p1, p2 = line_points[0], line_points[1]
                x1, y1 = float(p1[0]), float(p1[1])
                x2, y2 = float(p2[0]), float(p2[1])
            elif all(
                value is not None
                for value in (
                    self._extract_float_value(data, keys=("x1",), default=None),
                    self._extract_float_value(data, keys=("y1",), default=None),
                    self._extract_float_value(data, keys=("x2",), default=None),
                    self._extract_float_value(data, keys=("y2",), default=None),
                )
            ):
                x1 = float(self._extract_float_value(data, keys=("x1",), default=0.0))
                y1 = float(self._extract_float_value(data, keys=("y1",), default=0.0))
                x2 = float(self._extract_float_value(data, keys=("x2",), default=0.0))
                y2 = float(self._extract_float_value(data, keys=("y2",), default=0.0))
            else:
                return CommandResult(ok=False, error="mirror requires mirror-line coordinates.")
            return await self.backend.entity_mirror(
                entity_id,
                x1,
                y1,
                x2,
                y2,
            )
        if tool == "entity" and operation == "offset":
            entity_id = await self._resolve_entity_id(data, action, fallback_to_last=True)
            if not entity_id:
                return CommandResult(ok=False, error="Unable to resolve target entity for offset operation.")
            distance = self._extract_float_value(data, keys=("distance", "offset"), default=None)
            if distance is None:
                return CommandResult(ok=False, error="offset requires a distance.")
            return await self.backend.entity_offset(entity_id, float(distance))
        if tool == "entity" and operation == "array":
            entity_id = await self._resolve_entity_id(data, action, fallback_to_last=True)
            if not entity_id:
                return CommandResult(ok=False, error="Unable to resolve target entity for array operation.")
            rows = int(self._extract_float_value(data, keys=("rows",), default=1) or 1)
            cols = int(self._extract_float_value(data, keys=("cols", "columns"), default=1) or 1)
            row_dist = self._extract_float_value(data, keys=("row_dist", "row_spacing"), default=0.0) or 0.0
            col_dist = self._extract_float_value(data, keys=("col_dist", "col_spacing", "column_spacing"), default=0.0) or 0.0
            return await self.backend.entity_array(
                entity_id,
                rows,
                cols,
                float(row_dist),
                float(col_dist),
            )
        if tool == "entity" and operation == "fillet":
            id1 = await self._resolve_entity_id(
                data,
                action,
                preferred_keys=("id1", "entity_id1", "entity1_id", "entity_a", "first_entity_id", "handle1"),
                point_hint_keys=("point1", "first_point", "p1"),
            )
            id2 = await self._resolve_entity_id(
                data,
                action,
                preferred_keys=("id2", "entity_id2", "entity2_id", "entity_b", "second_entity_id", "handle2"),
                point_hint_keys=("point2", "second_point", "p2"),
                exclude_handles={id1} if id1 else None,
            )
            if not id1 or not id2:
                return CommandResult(ok=False, error="Unable to resolve both entities for fillet operation.")
            radius = self._extract_float_value(data, keys=("radius", "r"), default=None)
            if radius is None:
                return CommandResult(ok=False, error="fillet requires a radius.")
            return await self.backend.entity_fillet(id1, id2, float(radius))
        if tool == "entity" and operation == "chamfer":
            id1 = await self._resolve_entity_id(
                data,
                action,
                preferred_keys=("id1", "entity_id1", "entity1_id", "entity_a", "first_entity_id", "handle1"),
                point_hint_keys=("point1", "first_point", "p1"),
            )
            id2 = await self._resolve_entity_id(
                data,
                action,
                preferred_keys=("id2", "entity_id2", "entity2_id", "entity_b", "second_entity_id", "handle2"),
                point_hint_keys=("point2", "second_point", "p2"),
                exclude_handles={id1} if id1 else None,
            )
            if not id1 or not id2:
                return CommandResult(ok=False, error="Unable to resolve both entities for chamfer operation.")
            dist1 = self._extract_float_value(data, keys=("dist1", "distance1", "d1"), default=None)
            dist2 = self._extract_float_value(data, keys=("dist2", "distance2", "d2"), default=None)
            if dist1 is None or dist2 is None:
                return CommandResult(ok=False, error="chamfer requires dist1 and dist2.")
            return await self.backend.entity_chamfer(
                id1,
                id2,
                float(dist1),
                float(dist2),
            )
        if tool == "annotation" and operation == "create_text":
            text_value = data.get("text") or data.get("content") or data.get("value") or data.get("label")
            if not isinstance(text_value, str) or not text_value.strip():
                return CommandResult(ok=False, error="create_text requires a non-empty text value.")
            placement = await self._resolve_placement_point(data, action=action, prefer_scene_center=True)
            if not placement:
                return CommandResult(
                    ok=False,
                    error=(
                        "Unable to resolve text placement. Provide {x,y} or a spatial hint "
                        "(e.g. location/target_area such as 'top right')."
                    ),
                )
            height = self._extract_float_value(data, keys=("height", "text_height"), default=2.5)
            rotation = self._extract_float_value(data, keys=("rotation", "angle"), default=0.0)
            return await self.backend.create_text(
                float(placement[0]),
                float(placement[1]),
                text_value.strip(),
                float(height if height is not None else 2.5),
                float(rotation if rotation is not None else 0.0),
                self._extract_string_value(data, keys=("layer", "target_layer", "entity_layer"), default=None),
            )
        if tool == "annotation" and operation == "create_dimension_linear":
            try:
                x1, y1, x2, y2, _layer = self._extract_line_args(data, action)
            except ValueError:
                points = self._extract_points_list(data, keys=("points", "line_points", "extension_points"))
                if len(points) < 2:
                    return CommandResult(ok=False, error="create_dimension_linear requires two extension points.")
                x1, y1 = float(points[0][0]), float(points[0][1])
                x2, y2 = float(points[1][0]), float(points[1][1])
            dim_x = self._extract_float_value(data, keys=("dim_x", "x_dim", "dimension_x"), default=None)
            dim_y = self._extract_float_value(data, keys=("dim_y", "y_dim", "dimension_y"), default=None)
            if dim_x is None or dim_y is None:
                dim_point = self._parse_point(data.get("dim_point"))
                if dim_point is None:
                    dim_point = await self._resolve_placement_point(data, action=action, prefer_scene_center=True)
                if dim_point is not None:
                    dim_x, dim_y = float(dim_point[0]), float(dim_point[1])
                else:
                    dim_x = float((x1 + x2) / 2.0)
                    dim_y = float((y1 + y2) / 2.0 + 5.0)
            return await self.backend.create_dimension_linear(
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                float(dim_x),
                float(dim_y),
            )
        if tool == "annotation" and operation == "create_dimension_aligned":
            try:
                x1, y1, x2, y2, _layer = self._extract_line_args(data, action)
            except ValueError:
                return CommandResult(ok=False, error="create_dimension_aligned requires two points.")
            offset = self._extract_float_value(data, keys=("offset", "distance", "dim_offset"), default=5.0)
            return await self.backend.create_dimension_aligned(
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                float(offset if offset is not None else 5.0),
            )
        if tool == "annotation" and operation == "create_dimension_angular":
            center_point = self._parse_point(data.get("center"))
            cx = self._extract_float_value(data, keys=("cx", "center_x"), default=center_point[0] if center_point else None)
            cy = self._extract_float_value(data, keys=("cy", "center_y"), default=center_point[1] if center_point else None)
            try:
                x1, y1, x2, y2, _layer = self._extract_line_args(data, action)
            except ValueError:
                points = self._extract_points_list(data, keys=("points", "line_points"))
                if len(points) < 2:
                    return CommandResult(ok=False, error="create_dimension_angular requires center and two points.")
                x1, y1 = float(points[0][0]), float(points[0][1])
                x2, y2 = float(points[1][0]), float(points[1][1])
            if cx is None or cy is None:
                return CommandResult(ok=False, error="create_dimension_angular requires center coordinates.")
            return await self.backend.create_dimension_angular(
                float(cx),
                float(cy),
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            )
        if tool == "annotation" and operation == "create_dimension_radius":
            center_point = self._parse_point(data.get("center"))
            cx = self._extract_float_value(data, keys=("cx", "center_x", "x"), default=center_point[0] if center_point else None)
            cy = self._extract_float_value(data, keys=("cy", "center_y", "y"), default=center_point[1] if center_point else None)
            radius = self._extract_float_value(data, keys=("radius", "r"), default=None)
            angle = self._extract_float_value(data, keys=("angle", "theta"), default=45.0)
            if cx is None or cy is None or radius is None:
                return CommandResult(ok=False, error="create_dimension_radius requires center and radius.")
            return await self.backend.create_dimension_radius(
                float(cx),
                float(cy),
                float(radius),
                float(angle if angle is not None else 45.0),
            )
        if tool == "annotation" and operation == "create_leader":
            points = self._extract_points_list(data, keys=("points", "leader_points", "path"))
            if len(points) < 2:
                return CommandResult(ok=False, error="create_leader requires at least two points.")
            text_value = data.get("text") or data.get("content") or data.get("value") or data.get("label") or ""
            return await self.backend.create_leader(points, str(text_value))
        if tool == "layer" and operation == "list":
            return await self.backend.layer_list()
        if tool == "layer" and operation == "create":
            name = self._extract_string_value(data, keys=("name", "layer_name", "layer"), default=None)
            if not name:
                return CommandResult(ok=False, error="layer.create requires a layer name.")
            return await self.backend.layer_create(
                name,
                data.get("color", "white"),
                data.get("linetype", "CONTINUOUS"),
            )
        if tool == "layer" and operation == "set_current":
            name = self._extract_string_value(data, keys=("name", "layer_name", "layer"), default=None)
            if not name:
                return CommandResult(ok=False, error="layer.set_current requires a layer name.")
            return await self.backend.layer_set_current(name)
        if tool == "layer" and operation == "set_properties":
            name = self._extract_string_value(data, keys=("name", "layer_name", "layer"), default=None)
            if not name:
                return CommandResult(ok=False, error="layer.set_properties requires a layer name.")
            return await self.backend.layer_set_properties(
                name,
                data.get("color"),
                data.get("linetype"),
                data.get("lineweight"),
            )
        if tool == "layer" and operation == "freeze":
            name = self._extract_string_value(data, keys=("name", "layer_name", "layer"), default=None)
            if not name:
                return CommandResult(ok=False, error="layer.freeze requires a layer name.")
            return await self.backend.layer_freeze(name)
        if tool == "layer" and operation == "thaw":
            name = self._extract_string_value(data, keys=("name", "layer_name", "layer"), default=None)
            if not name:
                return CommandResult(ok=False, error="layer.thaw requires a layer name.")
            return await self.backend.layer_thaw(name)
        if tool == "layer" and operation == "lock":
            name = self._extract_string_value(data, keys=("name", "layer_name", "layer"), default=None)
            if not name:
                return CommandResult(ok=False, error="layer.lock requires a layer name.")
            return await self.backend.layer_lock(name)
        if tool == "layer" and operation == "unlock":
            name = self._extract_string_value(data, keys=("name", "layer_name", "layer"), default=None)
            if not name:
                return CommandResult(ok=False, error="layer.unlock requires a layer name.")
            return await self.backend.layer_unlock(name)
        if tool == "block" and operation == "list":
            return await self.backend.block_list()
        if tool == "block" and operation == "insert":
            name = self._extract_string_value(data, keys=("name", "block_name"), default=None)
            if not name:
                return CommandResult(ok=False, error="block.insert requires a block name.")
            placement = await self._resolve_placement_point(data, action=action, prefer_scene_center=True)
            if not placement:
                return CommandResult(ok=False, error="block.insert requires placement coordinates.")
            scale = self._extract_float_value(data, keys=("scale", "factor"), default=1.0)
            rotation = self._extract_float_value(data, keys=("rotation", "angle"), default=0.0)
            block_id = self._extract_string_value(data, keys=("block_id", "instance_id", "id"), default=None)
            return await self.backend.block_insert(
                name,
                float(placement[0]),
                float(placement[1]),
                float(scale if scale is not None else 1.0),
                float(rotation if rotation is not None else 0.0),
                block_id,
            )
        if tool == "block" and operation == "insert_with_attributes":
            name = self._extract_string_value(data, keys=("name", "block_name"), default=None)
            if not name:
                return CommandResult(ok=False, error="block.insert_with_attributes requires a block name.")
            placement = await self._resolve_placement_point(data, action=action, prefer_scene_center=True)
            if not placement:
                return CommandResult(ok=False, error="block.insert_with_attributes requires placement coordinates.")
            scale = self._extract_float_value(data, keys=("scale", "factor"), default=1.0)
            rotation = self._extract_float_value(data, keys=("rotation", "angle"), default=0.0)
            attributes = data.get("attributes")
            if not isinstance(attributes, dict):
                attributes = {}
            normalized_attributes = {str(k): str(v) for k, v in attributes.items()}
            return await self.backend.block_insert_with_attributes(
                name,
                float(placement[0]),
                float(placement[1]),
                float(scale if scale is not None else 1.0),
                float(rotation if rotation is not None else 0.0),
                normalized_attributes,
            )
        if tool == "block" and operation == "get_attributes":
            entity_id = await self._resolve_entity_id(data, action, fallback_to_last=True)
            if not entity_id:
                return CommandResult(ok=False, error="Unable to resolve target block insert for get_attributes.")
            return await self.backend.block_get_attributes(entity_id)
        if tool == "block" and operation == "update_attribute":
            entity_id = await self._resolve_entity_id(data, action, fallback_to_last=True)
            if not entity_id:
                return CommandResult(ok=False, error="Unable to resolve target block insert for update_attribute.")
            tag = self._extract_string_value(data, keys=("tag", "attribute", "attribute_tag"), default=None)
            if not tag:
                return CommandResult(ok=False, error="block.update_attribute requires tag.")
            value = data.get("value")
            if value is None:
                value = data.get("text")
            return await self.backend.block_update_attribute(entity_id, str(tag), str(value if value is not None else ""))
        if tool == "block" and operation == "define":
            name = self._extract_string_value(data, keys=("name", "block_name"), default=None)
            entities = data.get("entities")
            if not name or not isinstance(entities, list):
                return CommandResult(ok=False, error="block.define requires {name, entities}.")
            return await self.backend.block_define(name, entities)
        if tool == "drawing" and operation == "info":
            return await self.backend.drawing_info()
        if tool == "drawing" and operation == "create":
            return await self.backend.drawing_create(data.get("name"))
        if tool == "drawing" and operation == "save":
            return await self.backend.drawing_save(data.get("path"))
        if tool == "drawing" and operation == "save_as_dxf":
            path = self._extract_string_value(data, keys=("path", "file", "filename"), default=None)
            if not path:
                return CommandResult(ok=False, error="drawing.save_as_dxf requires a path.")
            return await self.backend.drawing_save_as_dxf(path)
        if tool == "drawing" and operation == "plot_pdf":
            path = self._extract_string_value(data, keys=("path", "file", "filename"), default=None)
            if not path:
                return CommandResult(ok=False, error="drawing.plot_pdf requires a path.")
            return await self.backend.drawing_plot_pdf(path)
        if tool == "drawing" and operation == "purge":
            return await self.backend.drawing_purge()
        if tool == "drawing" and operation == "get_variables":
            names_raw = data.get("names") or data.get("variables") or data.get("vars")
            names: list[str] | None = None
            if isinstance(names_raw, str):
                names = [part.strip() for part in re.split(r"[;,]", names_raw) if part.strip()]
            elif isinstance(names_raw, list):
                names = [str(item).strip() for item in names_raw if str(item).strip()]
            return await self.backend.drawing_get_variables(names)
        if tool == "drawing" and operation == "open":
            path = self._extract_string_value(data, keys=("path", "file", "filename"), default=None)
            if not path:
                return CommandResult(ok=False, error="drawing.open requires a path.")
            return await self.backend.drawing_open(path)
        if tool == "drawing" and operation == "undo":
            return await self.backend.undo()
        if tool == "drawing" and operation == "redo":
            return await self.backend.redo()
        if tool == "view" and operation == "zoom_extents":
            return await self.backend.zoom_extents()
        if tool == "view" and operation == "zoom_window":
            x1 = self._extract_float_value(data, keys=("x1",), default=None)
            y1 = self._extract_float_value(data, keys=("y1",), default=None)
            x2 = self._extract_float_value(data, keys=("x2",), default=None)
            y2 = self._extract_float_value(data, keys=("y2",), default=None)
            if None in (x1, y1, x2, y2):
                return CommandResult(ok=False, error="zoom_window requires x1,y1,x2,y2.")
            return await self.backend.zoom_window(float(x1), float(y1), float(x2), float(y2))

        raise ValueError(f"Unsupported action executor mapping: {tool}.{operation}")

    @staticmethod
    def _normalize_action_aliases(tool: Any, operation: Any) -> tuple[Any, Any]:
        def norm(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            return value.strip().lower().replace("-", "_").replace(" ", "_").replace(".", "_")

        tool_name = norm(tool)
        operation_name = norm(operation)

        tool_aliases = {
            "draw": "entity",
            "entities": "entity",
            "annotate": "annotation",
            "annotations": "annotation",
            "layers": "layer",
            "blocks": "block",
            "document": "drawing",
            "file": "drawing",
        }
        operation_aliases = {
            "entity_get_properties": "get",
            "get_entity_properties": "get",
            "get_properties": "get",
            "get_entity_info": "get",
            "entity_get": "get",
            "get_entity": "get",
            "query_entity": "get",
            "find_entity": "get",
            "select_entity": "get",
            "lookup_entity": "get",
            "resolve_entity": "get",
            "list_entities": "list",
            "entity_list": "list",
            "query_entities": "list",
            "find_entities": "list",
            "select_entities": "list",
            "query_by_type": "query_by_type",
            "find_by_type": "query_by_type",
            "select_by_type": "query_by_type",
            "list_by_type": "query_by_type",
            "get_entities_by_type": "query_by_type",
            "query_by_layer": "list",
            "find_by_layer": "list",
            "select_by_layer": "list",
            "list_by_layer": "list",
            "get_entities_by_layer": "list",
            "entity_count": "count",
            "count_entities": "count",
            "get_entity_count": "count",
            "delete_entity": "erase",
            "remove_entity": "erase",
            "erase_entity": "erase",
            "entity_erase": "erase",
            "move_entity": "move",
            "entity_move": "move",
            "copy_entity": "copy",
            "entity_copy": "copy",
            "rotate_entity": "rotate",
            "entity_rotate": "rotate",
            "scale_entity": "scale",
            "entity_scale": "scale",
            "mirror_entity": "mirror",
            "entity_mirror": "mirror",
            "offset_entity": "offset",
            "entity_offset": "offset",
            "array_entity": "array",
            "entity_array": "array",
            "fillet_entities": "fillet",
            "entity_fillet": "fillet",
            "chamfer_entities": "chamfer",
            "entity_chamfer": "chamfer",
            "line": "create_line",
            "draw_line": "create_line",
            "circle": "create_circle",
            "draw_circle": "create_circle",
            "polyline": "create_polyline",
            "draw_polyline": "create_polyline",
            "rectangle": "create_rectangle",
            "draw_rectangle": "create_rectangle",
            "arc": "create_arc",
            "draw_arc": "create_arc",
            "ellipse": "create_ellipse",
            "draw_ellipse": "create_ellipse",
            "mtext": "create_mtext",
            "create_multiline_text": "create_mtext",
            "hatch": "create_hatch",
            "draw_hatch": "create_hatch",
            "text": "create_text",
            "create_label": "create_text",
            "add_text": "create_text",
            "dim_linear": "create_dimension_linear",
            "dimension_linear": "create_dimension_linear",
            "add_dimension_linear": "create_dimension_linear",
            "dim_aligned": "create_dimension_aligned",
            "dimension_aligned": "create_dimension_aligned",
            "add_dimension_aligned": "create_dimension_aligned",
            "dim_angular": "create_dimension_angular",
            "dimension_angular": "create_dimension_angular",
            "add_dimension_angular": "create_dimension_angular",
            "dim_radius": "create_dimension_radius",
            "dimension_radius": "create_dimension_radius",
            "add_dimension_radius": "create_dimension_radius",
            "leader": "create_leader",
            "add_leader": "create_leader",
            "layer_list": "list",
            "list_layers": "list",
            "layer_create": "create",
            "create_layer": "create",
            "layer_set_current": "set_current",
            "set_current_layer": "set_current",
            "layer_set_properties": "set_properties",
            "set_layer_properties": "set_properties",
            "layer_freeze": "freeze",
            "freeze_layer": "freeze",
            "layer_thaw": "thaw",
            "thaw_layer": "thaw",
            "layer_lock": "lock",
            "lock_layer": "lock",
            "layer_unlock": "unlock",
            "unlock_layer": "unlock",
            "block_list": "list",
            "list_blocks": "list",
            "block_insert": "insert",
            "insert_block": "insert",
            "block_insert_with_attributes": "insert_with_attributes",
            "insert_block_with_attributes": "insert_with_attributes",
            "block_get_attributes": "get_attributes",
            "get_block_attributes": "get_attributes",
            "block_update_attribute": "update_attribute",
            "update_block_attribute": "update_attribute",
            "block_define": "define",
            "define_block": "define",
            "drawing_save_as_dxf": "save_as_dxf",
            "save_as_dxf": "save_as_dxf",
            "export_dxf": "save_as_dxf",
            "drawing_plot_pdf": "plot_pdf",
            "plot_pdf": "plot_pdf",
            "drawing_purge": "purge",
            "purge_drawing": "purge",
            "drawing_get_variables": "get_variables",
            "get_variables": "get_variables",
            "drawing_open": "open",
            "open_drawing": "open",
            "drawing_undo": "undo",
            "undo_last": "undo",
            "drawing_redo": "redo",
            "redo_last": "redo",
            "zoomextents": "zoom_extents",
        }

        if isinstance(tool_name, str):
            tool_name = tool_aliases.get(tool_name, tool_name)
        if isinstance(operation_name, str):
            operation_name = operation_aliases.get(operation_name, operation_name)
            for prefix, mapped_tool in (
                ("entity_", "entity"),
                ("annotation_", "annotation"),
                ("layer_", "layer"),
                ("block_", "block"),
                ("drawing_", "drawing"),
                ("view_", "view"),
            ):
                if operation_name.startswith(prefix):
                    stripped = operation_name.removeprefix(prefix)
                    operation_name = operation_aliases.get(stripped, stripped)
                    tool_name = mapped_tool
                    break
            if operation_name in ("find", "select", "query", "lookup", "resolve"):
                operation_name = "get"
            elif operation_name in ("find_all", "select_all", "query_all", "search"):
                operation_name = "list"
            if tool_name == "drawing" and operation_name in (
                "create_line",
                "create_circle",
                "create_polyline",
                "create_rectangle",
                "create_arc",
                "create_ellipse",
            ):
                tool_name = "entity"

        return tool_name, operation_name

    async def _query_entities(self, data: dict[str, Any]) -> CommandResult:
        layer = self._extract_string_value(
            data,
            keys=("layer", "target_layer", "entity_layer", "layer_name", "on_layer"),
            default=None,
        )
        filter_payload = data.get("filter")
        if isinstance(filter_payload, dict) and not layer:
            layer = self._extract_string_value(
                filter_payload,
                keys=("layer", "target_layer", "entity_layer", "layer_name", "on_layer"),
                default=None,
            )

        desired_type = self._extract_string_value(
            data,
            keys=("type", "entity_type", "geometry_type", "object_type"),
            default=None,
        )
        if not desired_type and isinstance(filter_payload, dict):
            desired_type = self._extract_string_value(
                filter_payload,
                keys=("type", "entity_type", "geometry_type", "object_type"),
                default=None,
            )
        desired_type_upper = desired_type.upper() if isinstance(desired_type, str) else None

        result = await self.backend.entity_list(layer)
        if not result.ok or not isinstance(result.payload, dict):
            return result
        entities_raw = result.payload.get("entities")
        if not isinstance(entities_raw, list):
            return result
        type_equivalences: dict[str, set[str]] = {
            "POLYLINE": {"POLYLINE", "LWPOLYLINE"},
            "LWPOLYLINE": {"POLYLINE", "LWPOLYLINE"},
        }

        filtered_entities: list[dict[str, Any]] = []
        for entity in entities_raw:
            if not isinstance(entity, dict):
                continue
            if desired_type_upper:
                entity_type = str(entity.get("type") or "").upper()
                accepted_types = type_equivalences.get(desired_type_upper, {desired_type_upper})
                if entity_type not in accepted_types:
                    continue
            filtered_entities.append(entity)

        return CommandResult(
            ok=True,
            payload={
                "entities": filtered_entities,
                "count": len(filtered_entities),
                "layer": layer,
                "type": desired_type_upper,
            },
        )

    async def _resolve_placement_point(
        self,
        data: dict[str, Any],
        action: dict[str, Any] | None = None,
        prefer_scene_center: bool = False,
    ) -> tuple[float, float] | None:
        point_keys = (
            "point",
            "target_point",
            "placement_point",
            "insertion_point",
            "insert_point",
            "origin",
            "base_point",
            "center",
            "location",
            "position",
            "at",
        )
        for key in point_keys:
            parsed = self._parse_point(data.get(key))
            if parsed is not None:
                return self._apply_point_offset(parsed, data)

        x = self._extract_float_value(data, keys=("x", "cx", "center_x"), default=None)
        y = self._extract_float_value(data, keys=("y", "cy", "center_y"), default=None)
        if x is not None and y is not None:
            return self._apply_point_offset((x, y), data)

        for nested_key in ("entity", "target", "target_entity", "selection", "object", "geometry"):
            nested_value = data.get(nested_key)
            if isinstance(nested_value, dict):
                nested_point = await self._resolve_placement_point(
                    nested_value,
                    action=action,
                    prefer_scene_center=False,
                )
                if nested_point is not None:
                    return self._apply_point_offset(nested_point, data)

        phrase = self._extract_string_value(
            data,
            keys=(
                "target_area",
                "area",
                "region",
                "location",
                "position",
                "where",
                "placement",
                "anchor",
                "spatial_hint",
                "description",
            ),
            default=None,
        )
        layer_hint = self._extract_string_value(
            data,
            keys=("layer", "target_layer", "entity_layer", "layer_name", "on_layer"),
            default=None,
        )
        if phrase:
            scene_points = await self._collect_scene_anchor_points(layer_hint=layer_hint)
            inferred = self._infer_relative_point_from_phrase(phrase, scene_points)
            if inferred is not None:
                return self._apply_point_offset(inferred, data)

        anchor_entity_id = await self._resolve_entity_id(data, action=action, fallback_to_last=False)
        if anchor_entity_id:
            detail = await self.backend.entity_get(anchor_entity_id)
            if detail.ok and isinstance(detail.payload, dict):
                anchors = self._extract_entity_anchor_points(detail.payload)
                if anchors:
                    centroid = (
                        sum(point[0] for point in anchors) / len(anchors),
                        sum(point[1] for point in anchors) / len(anchors),
                    )
                    return self._apply_point_offset(centroid, data)

        if prefer_scene_center:
            scene_points = await self._collect_scene_anchor_points(layer_hint=layer_hint)
            inferred = self._infer_relative_point_from_phrase("center", scene_points)
            if inferred is not None:
                return self._apply_point_offset(inferred, data)
        return None

    async def _collect_scene_anchor_points(
        self,
        layer_hint: str | None = None,
        max_entities: int = 140,
    ) -> list[tuple[float, float]]:
        list_result = await self.backend.entity_list(layer_hint)
        if not list_result.ok or not isinstance(list_result.payload, dict):
            return []
        entities_raw = list_result.payload.get("entities")
        if not isinstance(entities_raw, list) or not entities_raw:
            return []

        handles: list[str] = []
        points: list[tuple[float, float]] = []
        for entry in entities_raw:
            if not isinstance(entry, dict):
                continue
            points.extend(self._extract_entity_anchor_points(entry))
            handle = entry.get("handle")
            if isinstance(handle, str) and handle.strip():
                handles.append(handle.strip())

        if handles:
            if len(handles) > max_entities:
                step = max(1, len(handles) // max_entities)
                handles = handles[::step][:max_entities]
            for handle in handles:
                detail = await self.backend.entity_get(handle)
                if not detail.ok or not isinstance(detail.payload, dict):
                    continue
                points.extend(self._extract_entity_anchor_points(detail.payload))

        deduped: list[tuple[float, float]] = []
        seen: set[tuple[float, float]] = set()
        for x_val, y_val in points:
            rounded = (round(float(x_val), 6), round(float(y_val), 6))
            if rounded in seen:
                continue
            seen.add(rounded)
            deduped.append((float(x_val), float(y_val)))
        return deduped

    @staticmethod
    def _infer_relative_point_from_phrase(
        phrase: str,
        scene_points: list[tuple[float, float]],
    ) -> tuple[float, float] | None:
        if not scene_points:
            return None
        phrase_norm = phrase.strip().lower()
        if not phrase_norm:
            return None

        min_x = min(point[0] for point in scene_points)
        max_x = max(point[0] for point in scene_points)
        min_y = min(point[1] for point in scene_points)
        max_y = max(point[1] for point in scene_points)

        fx: float | None = None
        fy: float | None = None
        if any(token in phrase_norm for token in ("left", "west")):
            fx = 0.15
        elif any(token in phrase_norm for token in ("right", "east")):
            fx = 0.85
        elif any(token in phrase_norm for token in ("center", "centre", "middle", "mid")):
            fx = 0.5

        if any(token in phrase_norm for token in ("top", "upper", "north")):
            fy = 0.85
        elif any(token in phrase_norm for token in ("bottom", "lower", "south")):
            fy = 0.15
        elif any(token in phrase_norm for token in ("center", "centre", "middle", "mid")):
            fy = 0.5

        if fx is None and any(token in phrase_norm for token in ("quarter", "quadrant", "side")):
            fx = 0.5
        if fy is None and any(token in phrase_norm for token in ("quarter", "quadrant", "band", "row")):
            fy = 0.5

        if fx is None and fy is None:
            if any(token in phrase_norm for token in ("inside", "within", "around", "near", "interior")):
                fx, fy = 0.5, 0.5
            else:
                return None
        if fx is None:
            fx = 0.5
        if fy is None:
            fy = 0.5

        span_x = max_x - min_x
        span_y = max_y - min_y
        x = min_x + (span_x * fx if span_x > 0 else 0.0)
        y = min_y + (span_y * fy if span_y > 0 else 0.0)
        return float(x), float(y)

    @staticmethod
    def _apply_point_offset(point: tuple[float, float], data: dict[str, Any]) -> tuple[float, float]:
        offset_point = AutoCADStandaloneApp._parse_point(data.get("offset"))
        dx = AutoCADStandaloneApp._extract_float_value(data, keys=("dx", "offset_x", "x_offset"), default=0.0) or 0.0
        dy = AutoCADStandaloneApp._extract_float_value(data, keys=("dy", "offset_y", "y_offset"), default=0.0) or 0.0
        if offset_point is not None:
            dx += float(offset_point[0])
            dy += float(offset_point[1])
        return float(point[0] + dx), float(point[1] + dy)

    async def _resolve_entity_id(
        self,
        data: dict[str, Any],
        action: dict[str, Any] | None = None,
        preferred_keys: tuple[str, ...] | None = None,
        point_hint_keys: tuple[str, ...] | None = None,
        fallback_to_last: bool = False,
        exclude_handles: set[str] | None = None,
    ) -> str | None:
        exclusions = exclude_handles or set()
        candidate_ids = self._extract_entity_reference_candidates(data, preferred_keys=preferred_keys)

        for candidate in candidate_ids:
            normalized = candidate.strip()
            if not normalized or normalized in exclusions:
                continue
            check = await self.backend.entity_get(normalized)
            if check.ok:
                payload = check.payload if isinstance(check.payload, dict) else {}
                return str(payload.get("handle") or normalized)

        desired_type = self._extract_string_value(
            data,
            keys=("entity_type", "type", "geometry_type", "object_type"),
            default=None,
        )
        desired_layer = self._extract_string_value(
            data,
            keys=("entity_layer", "target_layer", "layer"),
            default=None,
        )
        hint_points = self._extract_hint_points(data, action=action, preferred_keys=point_hint_keys)

        list_result = await self.backend.entity_list(desired_layer)
        if not list_result.ok or not isinstance(list_result.payload, dict):
            if fallback_to_last:
                return "last"
            return None
        entities = list_result.payload.get("entities")
        if not isinstance(entities, list):
            if fallback_to_last:
                return "last"
            return None

        candidates: list[dict[str, Any]] = []
        desired_type_normalized = desired_type.upper() if isinstance(desired_type, str) else None
        type_equivalences: dict[str, set[str]] = {
            "POLYLINE": {"POLYLINE", "LWPOLYLINE"},
            "LWPOLYLINE": {"POLYLINE", "LWPOLYLINE"},
        }
        for entry in entities:
            if not isinstance(entry, dict):
                continue
            handle = entry.get("handle")
            if not isinstance(handle, str) or not handle or handle in exclusions:
                continue
            etype = str(entry.get("type") or "").upper()
            if desired_type_normalized:
                accepted_types = type_equivalences.get(desired_type_normalized, {desired_type_normalized})
                if etype not in accepted_types:
                    continue
            candidates.append(entry)

        if not candidates:
            if fallback_to_last:
                return "last"
            return None

        if not hint_points:
            return str(candidates[-1].get("handle"))

        scored: list[tuple[float, str]] = []
        for entry in candidates:
            handle = str(entry.get("handle"))
            detail = await self.backend.entity_get(handle)
            if not detail.ok or not isinstance(detail.payload, dict):
                continue
            anchors = self._extract_entity_anchor_points(detail.payload)
            if not anchors:
                continue
            min_distance = min(
                math.dist(hint_point, anchor)
                for hint_point in hint_points
                for anchor in anchors
            )
            scored.append((min_distance, handle))

        if scored:
            scored.sort(key=lambda item: item[0])
            return scored[0][1]

        return str(candidates[-1].get("handle"))

    @staticmethod
    def _extract_entity_reference_candidates(
        data: dict[str, Any],
        preferred_keys: tuple[str, ...] | None = None,
    ) -> list[str]:
        keys: list[str] = []
        if preferred_keys:
            keys.extend(preferred_keys)
        keys.extend(
            (
                "entity_id",
                "handle",
                "id",
                "entity_handle",
                "target_entity_id",
                "target_handle",
                "selected_entity_id",
                "selected_handle",
            )
        )

        candidates: list[str] = []

        def add_candidate(value: Any) -> None:
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())

        for key in keys:
            value = data.get(key)
            add_candidate(value)
            if isinstance(value, dict):
                for nested_key in ("entity_id", "handle", "id"):
                    add_candidate(value.get(nested_key))

        for nested in ("entity", "target", "target_entity", "selection", "object", "geometry"):
            nested_value = data.get(nested)
            if isinstance(nested_value, dict):
                for nested_key in keys:
                    add_candidate(nested_value.get(nested_key))
                for nested_key in ("entity_id", "handle", "id"):
                    add_candidate(nested_value.get(nested_key))

        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _extract_string_value(data: dict[str, Any], keys: tuple[str, ...], default: str | None = None) -> str | None:
        for key in keys:
            value = data.get(key)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped
            if isinstance(value, dict):
                for nested_key in keys:
                    nested_value = value.get(nested_key)
                    if isinstance(nested_value, str) and nested_value.strip():
                        return nested_value.strip()
        for nested in ("entity", "target", "target_entity", "selection", "object", "geometry"):
            nested_value = data.get(nested)
            if not isinstance(nested_value, dict):
                continue
            for key in keys:
                value = nested_value.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return default

    @staticmethod
    def _extract_float_value(data: dict[str, Any], keys: tuple[str, ...], default: float | None = None) -> float | None:
        for key in keys:
            value = data.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value.strip())
                except ValueError:
                    continue
            if isinstance(value, dict):
                for nested_key in keys:
                    nested_value = value.get(nested_key)
                    if isinstance(nested_value, (int, float)):
                        return float(nested_value)
                    if isinstance(nested_value, str):
                        try:
                            return float(nested_value.strip())
                        except ValueError:
                            continue
        return default

    @staticmethod
    def _extract_bool_value(data: dict[str, Any], keys: tuple[str, ...], default: bool | None = None) -> bool | None:
        truthy = {"1", "true", "yes", "y", "on"}
        falsy = {"0", "false", "no", "n", "off"}
        for key in keys:
            value = data.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in truthy:
                    return True
                if normalized in falsy:
                    return False
        return default

    @staticmethod
    def _extract_points_list(
        data: dict[str, Any],
        keys: tuple[str, ...] = ("points",),
        max_points: int | None = None,
    ) -> list[list[float]]:
        points: list[list[float]] = []

        def append_point(value: Any) -> None:
            parsed = AutoCADStandaloneApp._parse_point(value)
            if parsed is not None:
                points.append([float(parsed[0]), float(parsed[1])])

        for key in keys:
            value = data.get(key)
            if isinstance(value, list):
                for item in value:
                    append_point(item)
            elif isinstance(value, str):
                for chunk in re.split(r"[;|]", value):
                    append_point(chunk)
            elif isinstance(value, dict):
                nested_value = value.get("points")
                if isinstance(nested_value, list):
                    for item in nested_value:
                        append_point(item)

        if not points:
            pairs = AutoCADStandaloneApp._extract_coordinate_pairs_from_values(data)
            for x_val, y_val in pairs:
                points.append([float(x_val), float(y_val)])

        deduped: list[list[float]] = []
        seen: set[tuple[float, float]] = set()
        for x_val, y_val in points:
            rounded = (round(float(x_val), 6), round(float(y_val), 6))
            if rounded in seen:
                continue
            seen.add(rounded)
            deduped.append([float(x_val), float(y_val)])
            if max_points is not None and len(deduped) >= max_points:
                break
        return deduped

    @staticmethod
    def _extract_displacement(
        data: dict[str, Any],
        action: dict[str, Any] | None = None,
    ) -> tuple[float, float] | None:
        dx = AutoCADStandaloneApp._extract_float_value(
            data,
            keys=("dx", "delta_x", "offset_x", "x_offset"),
            default=None,
        )
        dy = AutoCADStandaloneApp._extract_float_value(
            data,
            keys=("dy", "delta_y", "offset_y", "y_offset"),
            default=None,
        )
        if dx is not None or dy is not None:
            return float(dx or 0.0), float(dy or 0.0)

        vector = AutoCADStandaloneApp._parse_point(data.get("offset"))
        if vector is None:
            vector = AutoCADStandaloneApp._parse_point(data.get("delta"))
        if vector is None:
            vector = AutoCADStandaloneApp._parse_point(data.get("vector"))
        if vector is not None:
            return float(vector[0]), float(vector[1])

        start = AutoCADStandaloneApp._parse_point(data.get("from")) or AutoCADStandaloneApp._parse_point(data.get("start"))
        end = AutoCADStandaloneApp._parse_point(data.get("to")) or AutoCADStandaloneApp._parse_point(data.get("end"))
        if start is not None and end is not None:
            return float(end[0] - start[0]), float(end[1] - start[1])

        x1 = AutoCADStandaloneApp._extract_float_value(data, keys=("x1",), default=None)
        y1 = AutoCADStandaloneApp._extract_float_value(data, keys=("y1",), default=None)
        x2 = AutoCADStandaloneApp._extract_float_value(data, keys=("x2",), default=None)
        y2 = AutoCADStandaloneApp._extract_float_value(data, keys=("y2",), default=None)
        if None not in (x1, y1, x2, y2):
            return float((x2 or 0.0) - (x1 or 0.0)), float((y2 or 0.0) - (y1 or 0.0))

        pairs = AutoCADStandaloneApp._extract_coordinate_pairs_from_values(data)
        if len(pairs) < 2 and isinstance(action, dict):
            pairs = AutoCADStandaloneApp._extract_coordinate_pairs_from_values(action)
        if len(pairs) >= 2:
            (sx, sy), (ex, ey) = pairs[0], pairs[1]
            return float(ex - sx), float(ey - sy)
        return None

    @staticmethod
    def _extract_hint_points(
        data: dict[str, Any],
        action: dict[str, Any] | None = None,
        preferred_keys: tuple[str, ...] | None = None,
    ) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []

        def add_point(value: Any) -> None:
            parsed = AutoCADStandaloneApp._parse_point(value)
            if parsed is not None:
                points.append(parsed)

        keys: list[str] = []
        if preferred_keys:
            keys.extend(preferred_keys)
        keys.extend(("point", "target_point", "near", "near_point", "location", "at", "center", "start", "end"))

        for key in keys:
            add_point(data.get(key))

        x = data.get("x")
        y = data.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            points.append((float(x), float(y)))

        points_values = data.get("points")
        if isinstance(points_values, list):
            for value in points_values[:4]:
                add_point(value)

        for nested in ("entity", "target", "target_entity", "selection", "object", "geometry"):
            nested_value = data.get(nested)
            if not isinstance(nested_value, dict):
                continue
            for key in keys:
                add_point(nested_value.get(key))

        pairs = AutoCADStandaloneApp._extract_coordinate_pairs_from_values(data)
        if not pairs and isinstance(action, dict):
            pairs = AutoCADStandaloneApp._extract_coordinate_pairs_from_values(action)
        points.extend(pairs[:4])

        deduped: list[tuple[float, float]] = []
        seen: set[tuple[float, float]] = set()
        for x_val, y_val in points:
            rounded = (round(float(x_val), 6), round(float(y_val), 6))
            if rounded in seen:
                continue
            seen.add(rounded)
            deduped.append((float(x_val), float(y_val)))
        return deduped

    @staticmethod
    def _extract_entity_anchor_points(entity_payload: dict[str, Any]) -> list[tuple[float, float]]:
        anchors: list[tuple[float, float]] = []
        for key in ("center", "start", "end", "insert", "point"):
            parsed = AutoCADStandaloneApp._parse_point(entity_payload.get(key))
            if parsed is not None:
                anchors.append(parsed)
        points_payload = entity_payload.get("points")
        if isinstance(points_payload, list):
            for point_value in points_payload:
                parsed = AutoCADStandaloneApp._parse_point(point_value)
                if parsed is not None:
                    anchors.append(parsed)

        if len(anchors) >= 2:
            x_mean = sum(p[0] for p in anchors) / len(anchors)
            y_mean = sum(p[1] for p in anchors) / len(anchors)
            anchors.append((x_mean, y_mean))

        deduped: list[tuple[float, float]] = []
        seen: set[tuple[float, float]] = set()
        for x_val, y_val in anchors:
            rounded = (round(float(x_val), 6), round(float(y_val), 6))
            if rounded in seen:
                continue
            seen.add(rounded)
            deduped.append((float(x_val), float(y_val)))
        return deduped
    @staticmethod
    def _extract_line_args(data: dict, action: dict | None = None) -> tuple[float, float, float, float, str | None]:
        """Extract line coordinates from several common payload shapes."""
        layer = data.get("layer")

        # Canonical shape: x1,y1,x2,y2
        if all(k in data for k in ("x1", "y1", "x2", "y2")):
            return (
                float(data["x1"]),
                float(data["y1"]),
                float(data["x2"]),
                float(data["y2"]),
                layer,
            )

        # Alternate shape: start/end points (list or dict)
        if "start" in data and "end" in data:
            p1 = AutoCADStandaloneApp._parse_point(data["start"])
            p2 = AutoCADStandaloneApp._parse_point(data["end"])
            if p1 and p2:
                return float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]), layer

        # Alternate shape: from/to points
        if "from" in data and "to" in data:
            p1 = AutoCADStandaloneApp._parse_point(data["from"])
            p2 = AutoCADStandaloneApp._parse_point(data["to"])
            if p1 and p2:
                return float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]), layer

        # Alternate shape: points array [[x1,y1],[x2,y2],...]
        points = data.get("points")
        if isinstance(points, list) and len(points) >= 2:
            p1 = AutoCADStandaloneApp._parse_point(points[0])
            p2 = AutoCADStandaloneApp._parse_point(points[1])
            if p1 and p2:
                return float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]), layer

        # Alternate flat keys: start_x/start_y/end_x/end_y
        if all(k in data for k in ("start_x", "start_y", "end_x", "end_y")):
            return (
                float(data["start_x"]),
                float(data["start_y"]),
                float(data["end_x"]),
                float(data["end_y"]),
                layer,
            )

        # Additional flat aliases
        alias_sets = [
            ("x_start", "y_start", "x_end", "y_end"),
            ("from_x", "from_y", "to_x", "to_y"),
            ("x_begin", "y_begin", "x_finish", "y_finish"),
        ]
        for sx, sy, ex, ey in alias_sets:
            if all(k in data for k in (sx, sy, ex, ey)):
                return (
                    float(data[sx]),
                    float(data[sy]),
                    float(data[ex]),
                    float(data[ey]),
                    layer,
                )

        # Nested coordinate object: {"coordinates": {...}} etc.
        for nested_key in ("coordinates", "coords", "geometry", "line", "segment"):
            nested = data.get(nested_key)
            if isinstance(nested, dict):
                try:
                    return AutoCADStandaloneApp._extract_line_args(nested, action=None)
                except ValueError:
                    pass

        # Alternate point key aliases
        point_alias_pairs = [
            ("start_point", "end_point"),
            ("point1", "point2"),
            ("p1", "p2"),
            ("a", "b"),
        ]
        for start_key, end_key in point_alias_pairs:
            if start_key in data and end_key in data:
                p1 = AutoCADStandaloneApp._parse_point(data[start_key])
                p2 = AutoCADStandaloneApp._parse_point(data[end_key])
                if p1 and p2:
                    return float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]), layer

        # Text fallback: parse "from (x,y) to (x,y)" from strings in data/action.
        coord_pairs = AutoCADStandaloneApp._extract_coordinate_pairs_from_values(data)
        if len(coord_pairs) < 2 and isinstance(action, dict):
            coord_pairs.extend(AutoCADStandaloneApp._extract_coordinate_pairs_from_values(action))
        if len(coord_pairs) >= 2:
            (x1, y1), (x2, y2) = coord_pairs[0], coord_pairs[1]
            return float(x1), float(y1), float(x2), float(y2), layer

        raise ValueError(
            "create_line requires coordinates. Supported formats: "
            "{x1,y1,x2,y2}, {start:[x,y],end:[x,y]}, {from:[x,y],to:[x,y]}, "
            "{points:[[x1,y1],[x2,y2]]}, or {start_x,start_y,end_x,end_y}."
        )

    @staticmethod
    def _extract_rectangle_args(data: dict, action: dict | None = None) -> tuple[float, float, float, float, str | None]:
        """Extract rectangle corners from common payload variants."""
        layer = data.get("layer")

        if all(k in data for k in ("x1", "y1", "x2", "y2")):
            return (
                float(data["x1"]),
                float(data["y1"]),
                float(data["x2"]),
                float(data["y2"]),
                layer,
            )

        point_pairs = [
            ("start", "end"),
            ("from", "to"),
            ("corner1", "corner2"),
            ("p1", "p2"),
            ("point1", "point2"),
            ("lower_left", "upper_right"),
            ("min", "max"),
        ]
        for first_key, second_key in point_pairs:
            if first_key in data and second_key in data:
                p1 = AutoCADStandaloneApp._parse_point(data[first_key])
                p2 = AutoCADStandaloneApp._parse_point(data[second_key])
                if p1 and p2:
                    return float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]), layer

        insertion_point = None
        for key in ("insertion_point", "insert_point", "origin", "base_point"):
            if key in data:
                insertion_point = AutoCADStandaloneApp._parse_point(data[key])
                if insertion_point:
                    break
        width = data.get("width", data.get("w"))
        height = data.get("height", data.get("h"))
        if insertion_point and width is not None and height is not None:
            x0, y0 = insertion_point
            return (
                float(x0),
                float(y0),
                float(x0 + float(width)),
                float(y0 + float(height)),
                layer,
            )

        center = AutoCADStandaloneApp._parse_point(data.get("center"))
        if center and width is not None and height is not None:
            cx, cy = center
            half_w = float(width) / 2.0
            half_h = float(height) / 2.0
            return (
                float(cx - half_w),
                float(cy - half_h),
                float(cx + half_w),
                float(cy + half_h),
                layer,
            )

        points = data.get("points")
        if isinstance(points, list) and len(points) >= 2:
            p1 = AutoCADStandaloneApp._parse_point(points[0])
            p2 = AutoCADStandaloneApp._parse_point(points[1])
            if p1 and p2:
                return float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]), layer

        coord_pairs = AutoCADStandaloneApp._extract_coordinate_pairs_from_values(data)
        if len(coord_pairs) < 2 and isinstance(action, dict):
            coord_pairs.extend(AutoCADStandaloneApp._extract_coordinate_pairs_from_values(action))
        if len(coord_pairs) >= 2:
            (x1, y1), (x2, y2) = coord_pairs[0], coord_pairs[1]
            return float(x1), float(y1), float(x2), float(y2), layer

        raise ValueError(
            "create_rectangle requires corner coordinates. Supported formats: "
            "{x1,y1,x2,y2}, {start:[x,y],end:[x,y]}, {insertion_point:[x,y],width,height}, "
            "or {center:[x,y],width,height}."
        )

    @staticmethod
    def _extract_circle_args(data: dict, action: dict | None = None) -> tuple[float, float, float, str | None]:
        """Extract circle arguments from common payload variants."""
        layer = data.get("layer")

        radius: float | None = None
        if "radius" in data:
            radius = float(data["radius"])
        elif "r" in data:
            radius = float(data["r"])
        elif "diameter" in data:
            radius = float(data["diameter"]) / 2.0
        elif "d" in data:
            radius = float(data["d"]) / 2.0

        if "cx" in data and "cy" in data and radius is not None:
            return float(data["cx"]), float(data["cy"]), float(radius), layer

        if "x" in data and "y" in data and radius is not None:
            return float(data["x"]), float(data["y"]), float(radius), layer

        if "center" in data and radius is not None:
            center = AutoCADStandaloneApp._parse_point(data["center"])
            if center:
                return float(center[0]), float(center[1]), float(radius), layer

        if "center_x" in data and "center_y" in data and radius is not None:
            return float(data["center_x"]), float(data["center_y"]), float(radius), layer

        for nested_key in ("coordinates", "coords", "geometry", "circle"):
            nested = data.get(nested_key)
            if isinstance(nested, dict):
                try:
                    return AutoCADStandaloneApp._extract_circle_args(nested, action=None)
                except ValueError:
                    pass

        coord_pairs = AutoCADStandaloneApp._extract_coordinate_pairs_from_values(data)
        if len(coord_pairs) < 1 and isinstance(action, dict):
            coord_pairs.extend(AutoCADStandaloneApp._extract_coordinate_pairs_from_values(action))
        if len(coord_pairs) >= 1 and radius is not None:
            x, y = coord_pairs[0]
            return float(x), float(y), float(radius), layer

        raise ValueError(
            "create_circle requires center and radius. Supported formats: "
            "{cx,cy,radius}, {center:[x,y],radius}, {x,y,radius}, or {center_x,center_y,radius}. "
            "You can also provide diameter as {diameter}."
        )

    @staticmethod
    def _parse_point(value: Any) -> tuple[float, float] | None:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return float(value[0]), float(value[1])
        if isinstance(value, dict):
            if "x" in value and "y" in value:
                return float(value["x"]), float(value["y"])
            if "x1" in value and "y1" in value:
                return float(value["x1"]), float(value["y1"])
            if "cx" in value and "cy" in value:
                return float(value["cx"]), float(value["cy"])
            if "center_x" in value and "center_y" in value:
                return float(value["center_x"]), float(value["center_y"])
        if isinstance(value, str):
            pattern = re.compile(r"^\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?\s*$")
            match = pattern.match(value)
            if match:
                return float(match.group(1)), float(match.group(2))
        return None

    @staticmethod
    def _extract_coordinate_pairs_from_values(container: Any) -> list[tuple[float, float]]:
        pairs: list[tuple[float, float]] = []
        pattern = re.compile(r"\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?")

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for v in value.values():
                    walk(v)
                return
            if isinstance(value, list):
                for v in value:
                    walk(v)
                return
            if isinstance(value, str):
                for match in pattern.findall(value):
                    pairs.append((float(match[0]), float(match[1])))

        walk(container)
        return pairs

    @staticmethod
    def _extract_entity_count(payload: dict[str, Any]) -> int | None:
        count = payload.get("entity_count")
        return AutoCADStandaloneApp._extract_count_value(count)

    @staticmethod
    def _extract_count_value(count: Any) -> int | None:
        if isinstance(count, int):
            return count
        if isinstance(count, float):
            return int(count)
        if isinstance(count, str):
            try:
                return int(float(count))
            except ValueError:
                return None
        return None


def run_app() -> None:
    root = tk.Tk()
    AutoCADStandaloneApp(root)
    root.mainloop()
