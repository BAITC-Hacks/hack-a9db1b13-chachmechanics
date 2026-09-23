"""Streamlit host for the frontend component; all actions use one controller."""
import base64
from pathlib import Path


def main():
    import streamlit as st
    import streamlit.components.v2 as components
    from frontend.controller import Controller

    st.set_page_config(page_title="TwinTurbo.ai · Wind Intelligence", page_icon="🌿", layout="wide", initial_sidebar_state="collapsed")
    st.markdown("""<style>
      .stApp {background:#101513;} header[data-testid="stHeader"] {display:none;}
      .block-container {max-width:none;padding:0!important;}
      [data-testid="stMainBlockContainer"] {padding:0!important;}
      [data-testid="stVerticalBlock"] {gap:0;} footer {visibility:hidden;}
    </style>""", unsafe_allow_html=True)
    assets = Path(__file__).resolve().parents[1] / "frontend" / "assets"

    @st.cache_resource
    def register_component():
        return components.component("twinturbo_dashboard", html=(assets / "dashboard.html").read_text(encoding="utf-8"),
            css=(assets / "dashboard.css").read_text(encoding="utf-8"), js=(assets / "dashboard.js").read_text(encoding="utf-8"))

    if "tt_controller" not in st.session_state:
        st.session_state.tt_controller = Controller()
    def handle_action():
        # Callbacks run before rendering: the component receives the new view
        # in the same rerun, including errors and export download payloads.
        action = st.session_state["twinturbo"].get("action")
        if action and action.get("nonce") != st.session_state.get("tt_last_action"):
            st.session_state.tt_last_action = action["nonce"]
            st.session_state.tt_controller.dispatch(action)

    controller = st.session_state.tt_controller
    view = controller.view()
    download = view["download"]
    # Streamlit owns file delivery; local preview keeps its own download dialog.
    register_component()(data={**view, "download": None}, key="twinturbo", on_action_change=handle_action)
    if download:
        @st.dialog("Экспорт готов")
        def export_dialog():
            st.write(download["filename"])
            if controller.mode == "fixture":
                st.caption("Синтетические данные. Не использовать для конкурсной сдачи.")
            st.download_button("Сохранить CSV", data=base64.b64decode(download["base64"]),
                file_name=download["filename"], mime=download["mime"], on_click="ignore")
        export_dialog()
