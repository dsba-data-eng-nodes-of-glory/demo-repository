"""Shared presentation for monitoring time-series charts."""
from __future__ import annotations

from html import escape

import streamlit as st

ACTUAL_COLOR = "#1976D2"
FORECAST_COLOR = "#F57C00"
TEXT_COLOR = "#d5e4fa"


def apply_monitoring_axes(fig, start, end, history_hours, ylabel, unit, revision):
    fig.update_layout(
        height=480, template="plotly_dark", hovermode="x unified",
        font=dict(color="#e8f1ff", size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,27,48,.50)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=25, r=25, t=65, b=35), uirevision=revision,
    )
    tick_hours = 6 if history_hours >= 48 else 3
    fig.update_xaxes(
        range=[start, end], dtick=tick_hours * 60 * 60 * 1000,
        tickformat="%d %b<br>%H:%M", tickangle=0, ticklabeloverflow="allow",
        tickfont=dict(size=11, color=TEXT_COLOR), title_text="Time (UTC)",
        title_font=dict(color=TEXT_COLOR), showgrid=True, gridcolor="#284365",
        zeroline=False, rangeslider_visible=False, automargin=True,
    )
    fig.update_yaxes(
        title_text=ylabel, uirevision=unit, tickformat="~f",
        tickfont=dict(color=TEXT_COLOR), title_font=dict(color=TEXT_COLOR),
        showgrid=True, gridcolor="#284365", zeroline=False, automargin=True,
    )


def mark_now(fig, origin):
    fig.add_vline(
        x=origin, line_width=2, line_dash="dash", line_color="#888888",
        annotation_text="Now", annotation_position="top", annotation_font_color="#dceafa",
    )


def render_initializing(label: str, detail: str, progress: str = ""):
    st.markdown(
        f"""
        <style>
          .forecast-init-card {{
            min-height: 290px; display:flex; flex-direction:column; gap:12px;
            align-items:center; justify-content:center; text-align:center;
            border:1px solid rgba(128,128,128,.28); border-radius:18px;
            padding:24px; margin:12px 0 22px;
            background:linear-gradient(135deg, rgba(30,64,110,.10), rgba(14,165,160,.08));
          }}
          .forecast-init-spinner {{
            width:52px; height:52px; border-radius:50%;
            border:5px solid rgba(100,116,139,.22); border-top-color:#0ea5a4;
            border-right-color:#3b82f6; animation:forecast-spin 1.05s linear infinite;
          }}
          @keyframes forecast-spin {{ to {{ transform:rotate(360deg); }} }}
          @media (prefers-reduced-motion: reduce) {{ .forecast-init-spinner {{ animation:none; }} }}
          .forecast-init-title {{ font-size:22px; font-weight:700; }}
          .forecast-init-copy {{ max-width:630px; opacity:.83; }}
        </style>
        <div class="forecast-init-card" role="status">
          <div class="forecast-init-spinner" aria-hidden="true"></div>
          <div class="forecast-init-title">Initializing {escape(label)}</div>
          <div class="forecast-init-copy">{escape(detail)}</div>
          <div class="forecast-init-copy">{escape(progress)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
