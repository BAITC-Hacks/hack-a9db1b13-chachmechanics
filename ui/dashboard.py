"""Streamlit host for the frontend component; all actions use one controller."""
import base64
import os
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
        default_mode = os.environ.get("TWINTURBO_DEFAULT_MODE", "fixture")
        if default_mode not in {"fixture", "replay", "submission"}:
            default_mode = "fixture"
        st.session_state.tt_controller = Controller(mode=default_mode)
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
    # Acknowledge even actions that leave the view identical (export, refresh,
    # selecting the same release), so the frontend clears its loading state.
    register_component()(data={**view, "download": None, "action_ack": st.session_state.get("tt_last_action")},
                         key="twinturbo", on_action_change=handle_action)
    result, selection = view.get("result"), view.get("selection")
    if result and selection:
        from ui.operations import operational_summary, wind_map_rows
        import pandas as pd

        turbine_id = selection["turbine_id"]
        operations = operational_summary(
            result, turbine_id, int(selection["horizon_hours"])
        )
        st.markdown("### Оперативная аналитика")
        metric_columns = st.columns(4)
        metric_columns[0].metric(
            "Прогнозный КИУМ",
            f"{operations['forecast_kium']['capacity_factor']:.1%}",
            help="Средняя нормализованная мощность за выбранный горизонт.",
        )
        actual_kium = operations["actual_kium"]
        metric_columns[1].metric(
            "Фактический КИУМ",
            f"{actual_kium['capacity_factor']:.1%}" if actual_kium else "нет факта",
        )
        accuracy = operations["accuracy"]
        metric_columns[2].metric(
            "MAE",
            f"{accuracy['mae']:.4f}" if accuracy else "ещё не доступна",
        )
        coverage = operations["uncertainty"]["coverage_q10_q90"]
        metric_columns[3].metric(
            "Покрытие Q10–Q90",
            f"{coverage:.1%}" if coverage is not None else "не проверено",
        )

        left, right = st.columns((1.15, .85))
        with left:
            st.markdown("#### Ветровая карта")
            map_rows = wind_map_rows(
                result, view.get("catalog", {}).get("turbines", []), turbine_id
            )
            if map_rows:
                frame = pd.DataFrame(map_rows)
                frame["marker_size"] = frame["wind_ms"].fillna(1).clip(lower=1) * 90
                st.map(
                    frame,
                    latitude="latitude",
                    longitude="longitude",
                    size="marker_size",
                    color="#55d6a3",
                    zoom=13,
                    use_container_width=True,
                )
                st.dataframe(
                    frame[["turbine_id", "wind_ms", "temperature_c", "direction"]],
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(
                    "Карта показывает прогноз ветра в точках турбин. Это не аэродинамическое поле; обе турбины могут попадать в одну ячейку GFS 0,25°.")
            else:
                st.info("Координаты или погодные точки для карты отсутствуют.")
        with right:
            st.markdown("#### Рекомендации оператору и обслуживанию")
            for recommendation in operations["recommendations"]:
                message = f"**{recommendation['title']}**  \n{recommendation['text']}"
                if recommendation["severity"] == "warning":
                    st.warning(message)
                else:
                    st.info(message)
            if accuracy:
                st.caption(
                    "Отклонения: "
                    f"недопрогноз {accuracy['underforecast_rate']:.1%}, "
                    f"перепрогноз {accuracy['overforecast_rate']:.1%}, "
                    f"в пределах ±5 п.п. {accuracy['within_deadband_rate']:.1%}."
                )
    if download:
        @st.dialog("Экспорт готов")
        def export_dialog():
            st.write(download["filename"])
            if controller.mode == "fixture":
                st.caption("Синтетические данные. Не использовать для конкурсной сдачи.")
            st.download_button("Сохранить CSV", data=base64.b64decode(download["base64"]),
                file_name=download["filename"], mime=download["mime"], on_click="ignore")
        export_dialog()
