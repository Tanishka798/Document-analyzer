"""
frontend/app.py
----------------
Why this module exists:
    The Streamlit UI on top of the backend API (Phase 7). Every backend
    call goes through api_client.py; this file is only responsible for
    widgets, layout, and session state.

Voice input/output, and why they're built the way they are:
    - Input: st.audio_input (built into Streamlit 1.38+) records a clip
      in the browser with no extra JS component or network dependency.
      We don't auto-transcribe on every rerun (Streamlit reruns the whole
      script on every interaction, and the recorded clip persists in the
      widget until cleared) — instead there's an explicit "Use this
      recording" button, so transcription only happens once per clip.
        - Output: browser speech synthesis reads chat, summary, and comparison
            results aloud with pause/resume and stop controls. No server-side TTS
            package is required for the Streamlit UI.

Run with: streamlit run frontend/app.py
(from the project root; Streamlit calls the FastAPI app in-process)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make sure the project root (parent of this "frontend" folder) is on
# sys.path so `backend.config` resolves no matter how Streamlit was
# launched (streamlit's own sys.path handling only guarantees this
# script's own directory, not the project root above it).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import api_client as api

st.set_page_config(page_title="Document Assistant", page_icon="📄", layout="wide")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []  # list of {role, content, sources?}
if "selected_doc_ids" not in st.session_state:
    st.session_state.selected_doc_ids = set()
if "chat_query_input" not in st.session_state:
    st.session_state.chat_query_input = ""
if st.session_state.pop("clear_chat_query", False):
    st.session_state.chat_query_input = ""


# ---------------------------------------------------------------------------
# Sidebar: health, upload, document list
# ---------------------------------------------------------------------------

def fetch_health() -> dict:
    try:
        return api.health()
    except api.APIError as exc:
        st.sidebar.error(str(exc))
        return {}


def render_upload() -> None:
    st.sidebar.subheader("Upload a document")
    uploaded = st.sidebar.file_uploader(
        "PDF, DOCX, PPTX, XLSX, TXT, or MD",
        type=["pdf", "docx", "pptx", "xlsx", "txt", "md"],
    )
    if uploaded and st.sidebar.button("Upload", use_container_width=True):
        with st.sidebar:
            with st.spinner(f"Ingesting {uploaded.name}..."):
                try:
                    result = api.upload_document(uploaded.getvalue(), uploaded.name)
                except api.APIError as exc:
                    st.sidebar.error(str(exc))
                    return
        if result["status"] == "ready":
            st.sidebar.success(f"'{uploaded.name}' ready ({result['num_chunks']} chunks).")
        else:
            st.sidebar.error(f"'{uploaded.name}' failed: {result.get('error')}")
        st.rerun()


def render_document_list() -> None:
    st.sidebar.subheader("Documents")
    try:
        documents = api.list_documents()
    except api.APIError as exc:
        st.sidebar.error(str(exc))
        return

    if not documents:
        st.sidebar.caption("No documents uploaded yet.")
        return

    st.sidebar.caption("Check documents to scope chat/compare to just those.")
    valid_ids = {d["id"] for d in documents}
    st.session_state.selected_doc_ids &= valid_ids  # drop stale selections

    for doc in documents:
        status_icon = {"ready": "✅", "processing": "⏳", "failed": "❌"}.get(doc["status"], "❓")
        col_check, col_label, col_delete = st.sidebar.columns([0.15, 0.6, 0.25])
        with col_check:
            checked = st.checkbox(
                "Select",
                value=doc["id"] in st.session_state.selected_doc_ids,
                key=f"chk_{doc['id']}",
                label_visibility="collapsed",
            )
            if checked:
                st.session_state.selected_doc_ids.add(doc["id"])
            else:
                st.session_state.selected_doc_ids.discard(doc["id"])
        with col_label:
            st.markdown(f"{status_icon} **{doc['filename']}**")
            if doc["status"] == "failed" and doc.get("error"):
                st.caption(f"⚠️ {doc['error']}")
            elif doc["status"] == "ready":
                st.caption(f"{doc['num_chunks']} chunks")
        with col_delete:
            if st.button("🗑️", key=f"del_{doc['id']}"):
                try:
                    api.delete_document(doc["id"])
                except api.APIError as exc:
                    st.sidebar.error(str(exc))
                st.rerun()


health_info = fetch_health()
render_upload()
render_document_list()

ready_docs = [d for d in api.list_documents() if d["status"] == "ready"] if health_info else []


def read_aloud_button(text: str, auto_play: bool = False) -> None:
    safe_text = json.dumps(text, ensure_ascii=True).replace("<", "\\u003c")
    auto_play_script = "window.setTimeout(speak, 300);" if auto_play else ""
    st.components.v1.html(
        f"""
        <button id="speak" type="button" aria-label="Read AI response aloud">
            &#128266; Read aloud
        </button>
        <button id="pause" type="button" aria-label="Pause speech" disabled>
            &#9208; Pause
        </button>
        <button id="stop" type="button" aria-label="Stop speech" disabled>
            &#9209; Stop
        </button>
        <script>
            const responseText = {safe_text};
            const button = document.getElementById("speak");
            const pauseButton = document.getElementById("pause");
            const stopButton = document.getElementById("stop");
            function speak() {{
                if (!("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) {{
                    button.textContent = "Speech is unavailable in this browser";
                    return;
                }}
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(responseText);
                utterance.onstart = () => {{
                    button.textContent = "Speaking...";
                    pauseButton.disabled = false;
                    stopButton.disabled = false;
                }};
                utterance.onend = resetControls;
                utterance.onerror = resetControls;
                window.speechSynthesis.speak(utterance);
            }}
            function resetControls() {{
                button.innerHTML = "&#128266; Read aloud";
                pauseButton.innerHTML = "&#9208; Pause";
                pauseButton.setAttribute("aria-label", "Pause speech");
                pauseButton.disabled = true;
                stopButton.disabled = true;
            }}
            function togglePause() {{
                if (window.speechSynthesis.paused) {{
                    window.speechSynthesis.resume();
                    pauseButton.innerHTML = "&#9208; Pause";
                    pauseButton.setAttribute("aria-label", "Pause speech");
                }} else {{
                    window.speechSynthesis.pause();
                    pauseButton.innerHTML = "&#9654; Resume";
                    pauseButton.setAttribute("aria-label", "Resume speech");
                }}
            }}
            function stop() {{
                window.speechSynthesis.cancel();
                resetControls();
            }}
            button.addEventListener("click", speak);
            pauseButton.addEventListener("click", togglePause);
            stopButton.addEventListener("click", stop);
            {auto_play_script}
        </script>
        <style>
            body {{ margin: 0; display: flex; gap: 6px; }}
            button {{
                min-height: 34px;
                padding: 5px 8px;
                border: 1px solid #9aa0a6;
                border-radius: 4px;
                background: transparent;
                color: #c62828;
                font: 14px sans-serif;
                cursor: pointer;
            }}
            button:disabled {{ opacity: 0.5; cursor: default; }}
        </style>
        """,
        height=42,
        scrolling=False,
    )


def ask_assistant(query: str, document_ids: list[str] | None, speak_reply: bool) -> None:
    st.session_state.chat_messages.append({"role": "user", "content": query})
    try:
        result = api.chat(query, document_ids=document_ids)
        assistant_message = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    except api.APIError as exc:
        assistant_message = {"role": "assistant", "content": f"⚠️ {exc}", "sources": []}
    assistant_message["auto_speak"] = speak_reply
    st.session_state.chat_messages.append(assistant_message)


# ---------------------------------------------------------------------------
# Main area: Chat / Summarize / Compare tabs
# ---------------------------------------------------------------------------

tab_chat, tab_summarize, tab_compare = st.tabs(["💬 Chat", "📝 Summarize", "🔀 Compare"])

with tab_chat:
    scope = st.session_state.selected_doc_ids
    if scope:
        st.caption(f"Scoped to {len(scope)} selected document(s).")
    else:
        st.caption("Searching across all uploaded documents.")

    speak_replies = st.checkbox("Speak AI replies", value=False, key="speak_ai_replies")
    audio_clip = st.audio_input("Ask by voice")
    voice_col, text_col = st.columns(2)
    with voice_col:
        ask_by_voice = st.button(
            "Ask with recording",
            type="primary",
            disabled=audio_clip is None,
            use_container_width=True,
        )
    with text_col:
        transcribe_only = st.button(
            "Transcribe to text",
            disabled=audio_clip is None,
            use_container_width=True,
        )

    if ask_by_voice or transcribe_only:
        try:
            with st.spinner("Transcribing your recording..."):
                spoken_query = api.transcribe(audio_clip.getvalue())
            if ask_by_voice:
                with st.spinner("Waiting for the AI response..."):
                    ask_assistant(
                        spoken_query,
                        list(scope) if scope else None,
                        speak_replies,
                    )
                st.session_state.chat_query_input = ""
            else:
                st.session_state.chat_query_input = spoken_query
        except api.APIError as exc:
            st.error(str(exc))
        else:
            st.rerun()

    query = st.text_input("Ask a question about your documents", key="chat_query_input")
    if st.button("Send", type="primary") and query.strip():
        with st.spinner("Waiting for the AI response..."):
            ask_assistant(query, list(scope) if scope else None, speak_replies)
        st.session_state.clear_chat_query = True
        st.rerun()

    for i, message in enumerate(st.session_state.chat_messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                if message.get("sources"):
                    with st.expander(f"{len(message['sources'])} source(s)"):
                        for source in message["sources"]:
                            st.markdown(
                                f"**{source['filename']}** (chunk {source['chunk_index']}, "
                                f"distance {source['distance']:.3f})"
                            )
                            st.caption(source["text"][:300] + ("..." if len(source["text"]) > 300 else ""))
                read_aloud_button(
                    message["content"],
                    auto_play=message.pop("auto_speak", False),
                )

    if st.session_state.chat_messages and st.button("Clear conversation"):
        st.session_state.chat_messages = []
        st.rerun()

with tab_summarize:
    if not ready_docs:
        st.caption("Upload a document first.")
    else:
        options = {d["filename"]: d["id"] for d in ready_docs}
        choice = st.selectbox("Choose a document", options.keys())
        if st.button("Summarize", type="primary"):
            with st.spinner("Summarizing..."):
                try:
                    result = api.summarize(options[choice])
                    st.session_state["last_summary"] = result["summary"]
                except api.APIError as exc:
                    st.error(str(exc))
        if st.session_state.get("last_summary"):
            st.markdown(st.session_state["last_summary"])
            read_aloud_button(st.session_state["last_summary"])

with tab_compare:
    if len(ready_docs) < 2:
        st.caption("Upload at least two documents to compare.")
    else:
        options = {d["filename"]: d["id"] for d in ready_docs}
        chosen = st.multiselect("Choose 2-5 documents", options.keys(), max_selections=5)
        if st.button("Compare", type="primary", disabled=len(chosen) < 2):
            with st.spinner("Comparing..."):
                try:
                    result = api.compare([options[name] for name in chosen])
                    st.session_state["last_comparison"] = result["comparison"]
                except api.APIError as exc:
                    st.error(str(exc))
        if st.session_state.get("last_comparison"):
            st.markdown(st.session_state["last_comparison"])
            read_aloud_button(st.session_state["last_comparison"])