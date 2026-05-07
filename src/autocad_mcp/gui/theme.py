"""Centralized visual theme for the standalone Tkinter GUI."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


@dataclass(frozen=True)
class GUITheme:
    bg: str = "#101521"
    panel: str = "#151C2C"
    panel_alt: str = "#1A2236"
    border: str = "#2A3450"
    text: str = "#E7ECF7"
    text_muted: str = "#AAB6CC"
    accent: str = "#5EA1FF"
    accent_hover: str = "#75B0FF"
    success: str = "#42BA96"
    warning: str = "#E9B949"
    danger: str = "#E66C6C"
    input_bg: str = "#0E1422"
    input_fg: str = "#EAF0FF"
    output_bg: str = "#0C1220"
    output_fg: str = "#D9E4FA"


def default_gui_theme() -> GUITheme:
    return GUITheme()


def apply_ttk_theme(root: tk.Tk, theme: GUITheme) -> ttk.Style:
    style = ttk.Style(root)
    available = set(style.theme_names())
    if "clam" in available:
        style.theme_use("clam")
    root.configure(bg=theme.bg)

    style.configure("App.TFrame", background=theme.bg)
    style.configure("Surface.TFrame", background=theme.panel)
    style.configure("Card.TFrame", background=theme.panel_alt)

    style.configure(
        "Header.TLabel",
        background=theme.panel_alt,
        foreground=theme.text,
        font=("Segoe UI Semibold", 13),
    )
    style.configure(
        "Subtle.TLabel",
        background=theme.panel_alt,
        foreground=theme.text_muted,
        font=("Segoe UI", 9),
    )
    style.configure(
        "Body.TLabel",
        background=theme.panel,
        foreground=theme.text,
        font=("Segoe UI", 10),
    )
    style.configure(
        "Status.TLabel",
        background=theme.panel_alt,
        foreground=theme.text,
        font=("Segoe UI Semibold", 9),
        padding=(8, 3),
    )
    style.configure(
        "Caption.TLabel",
        background=theme.panel_alt,
        foreground=theme.text_muted,
        font=("Segoe UI", 9),
    )

    style.configure(
        "Primary.TButton",
        background=theme.accent,
        foreground="#FFFFFF",
        bordercolor=theme.accent,
        focusthickness=2,
        focuscolor=theme.accent,
        padding=(10, 6),
        font=("Segoe UI Semibold", 9),
    )
    style.map(
        "Primary.TButton",
        background=[("active", theme.accent_hover), ("pressed", theme.accent_hover)],
        foreground=[("disabled", "#9CB4DB")],
    )

    style.configure(
        "Secondary.TButton",
        background=theme.panel_alt,
        foreground=theme.text,
        bordercolor=theme.border,
        padding=(10, 6),
        font=("Segoe UI", 9),
    )
    style.map(
        "Secondary.TButton",
        background=[("active", "#202A42"), ("pressed", "#202A42")],
        foreground=[("disabled", "#8D99B0")],
    )

    style.configure(
        "Danger.TButton",
        background=theme.danger,
        foreground="#FFFFFF",
        bordercolor=theme.danger,
        padding=(10, 6),
        font=("Segoe UI Semibold", 9),
    )
    style.map(
        "Danger.TButton",
        background=[("active", "#EC7C7C"), ("pressed", "#EC7C7C")],
    )

    style.configure(
        "App.TCheckbutton",
        background=theme.panel,
        foreground=theme.text_muted,
        font=("Segoe UI", 9),
    )
    style.map(
        "App.TCheckbutton",
        background=[("active", theme.panel)],
        foreground=[("active", theme.text)],
    )
    style.configure(
        "App.TEntry",
        fieldbackground=theme.input_bg,
        foreground=theme.input_fg,
        bordercolor=theme.border,
        insertcolor=theme.text,
        padding=(8, 5),
    )
    style.map(
        "App.TEntry",
        fieldbackground=[("readonly", theme.output_bg)],
        foreground=[("disabled", theme.text_muted)],
    )
    style.configure(
        "App.TRadiobutton",
        background=theme.panel,
        foreground=theme.text_muted,
        font=("Segoe UI", 9),
    )
    style.map(
        "App.TRadiobutton",
        background=[("active", theme.panel)],
        foreground=[("active", theme.text)],
    )

    style.configure(
        "App.TLabelframe",
        background=theme.panel,
        foreground=theme.text,
        bordercolor=theme.border,
        borderwidth=1,
        relief="solid",
    )
    style.configure(
        "App.TLabelframe.Label",
        background=theme.panel,
        foreground=theme.text,
        font=("Segoe UI Semibold", 10),
    )

    style.configure(
        "App.TNotebook",
        background=theme.panel,
        bordercolor=theme.border,
        tabmargins=(6, 6, 6, 0),
    )
    style.configure(
        "App.TNotebook.Tab",
        background=theme.panel_alt,
        foreground=theme.text_muted,
        bordercolor=theme.border,
        padding=(14, 6),
        font=("Segoe UI", 9),
    )
    style.map(
        "App.TNotebook.Tab",
        background=[("selected", theme.bg), ("active", "#222D47")],
        foreground=[("selected", theme.text), ("active", theme.text)],
    )
    style.configure(
        "Vertical.TScrollbar",
        background=theme.panel_alt,
        troughcolor=theme.bg,
        bordercolor=theme.border,
        arrowcolor=theme.text_muted,
    )
    style.configure(
        "Horizontal.TScrollbar",
        background=theme.panel_alt,
        troughcolor=theme.bg,
        bordercolor=theme.border,
        arrowcolor=theme.text_muted,
    )

    return style


def style_scrolled_text(widget: ScrolledText, theme: GUITheme) -> None:
    widget.configure(
        bg=theme.output_bg,
        fg=theme.output_fg,
        insertbackground=theme.text,
        selectbackground="#35568A",
        selectforeground="#FFFFFF",
        relief="flat",
        bd=0,
        padx=8,
        pady=8,
        highlightthickness=1,
        highlightbackground=theme.border,
        highlightcolor=theme.accent,
    )


def style_canvas(canvas: tk.Canvas, theme: GUITheme, *, background: str | None = None) -> None:
    canvas.configure(
        background=background or theme.panel_alt,
        bd=0,
        highlightthickness=1,
        highlightbackground=theme.border,
        highlightcolor=theme.accent,
    )
