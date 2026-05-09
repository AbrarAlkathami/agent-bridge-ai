import json

import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Agent Bridge AI", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 0; }
.stChatMessage { padding: 0.25rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("Agent Bridge AI")

files_tab, chat_tab, tools_tab, graph_tab = st.tabs(["Files", "RAG Chat", "Tools Agent", "Graph Agent"])

# ── Files tab ─────────────────────────────────────────────────────────────────

with files_tab:
    st.subheader("Upload Documents")
    uploaded = st.file_uploader(
        "Choose a file (PDF, TXT, or Markdown)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=False,
    )

    if uploaded and st.button("Upload", use_container_width=False):
        with st.spinner("Uploading and indexing…"):
            resp = requests.post(
                f"{API_BASE}/documents/uploadfile/",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            )
        if resp.ok:
            st.success(resp.json().get("message", "Uploaded successfully."))
            st.rerun()
        else:
            st.error(f"Upload failed ({resp.status_code}): {resp.text}")

    st.divider()
    st.subheader("Indexed Documents")

    if st.button("Refresh list"):
        st.rerun()

    try:
        resp = requests.get(f"{API_BASE}/documents/", timeout=5)
        if resp.ok:
            files = resp.json().get("files", [])
            if files:
                for f in files:
                    col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
                    col1.write(f["filename"])
                    col2.write(f"{f['chunk_count']} chunks")
                    col3.link_button(
                        "Preview",
                        f"{API_BASE}/documents/{f['document_id']}/preview",
                        use_container_width=True,
                    )
                    if col4.button("Delete", key=f"delete_{f['document_id']}", use_container_width=True):
                        del_resp = requests.delete(f"{API_BASE}/documents/{f['document_id']}")
                        if del_resp.ok:
                            st.success(f"Deleted {f['filename']}")
                            st.rerun()
                        else:
                            st.error(f"Delete failed ({del_resp.status_code}): {del_resp.text}")
            else:
                st.info("No documents indexed yet. Upload a file above.")
        else:
            st.error(f"API error ({resp.status_code}): {resp.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not reach the API. Make sure the server is running: `uvicorn main:app --reload`")

# ── RAG Chat tab ──────────────────────────────────────────────────────────────

def _render_chunks(chunks: list):
    with st.expander(f"Retrieved context ({len(chunks)} chunks)"):
        for i, chunk in enumerate(chunks):
            source = chunk.get("source", "unknown")
            content = chunk.get("content", "")
            st.caption(f"Chunk {i + 1} — `{source}`")
            st.markdown(
                f'<div style="background:#f6f8fa;border-left:3px solid #4a9eff;'
                f'padding:10px 14px;border-radius:4px;font-size:0.85rem;'
                f'white-space:pre-wrap;word-break:break-word;">{content}</div>',
                unsafe_allow_html=True,
            )
            if i < len(chunks) - 1:
                st.divider()


with chat_tab:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_box = st.container(height=560)

    with chat_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("chunks"):
                    _render_chunks(msg["chunks"])

    query = st.chat_input("Ask a question about your documents…")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with chat_box:
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                answer = ""
                chunks = []
                try:
                    with requests.post(
                        f"{API_BASE}/rag/query",
                        json={"query": query},
                        stream=True,
                        timeout=60,
                    ) as resp:
                        if resp.ok:
                            placeholder = st.empty()
                            for raw_line in resp.iter_lines():
                                if not raw_line:
                                    continue
                                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                                if not line.startswith("data: "):
                                    continue
                                event = json.loads(line[len("data: "):])
                                if event.get("type") == "chunks":
                                    chunks = event.get("content", [])
                                elif event.get("type") == "token":
                                    answer += event.get("content", "")
                                    placeholder.markdown(answer)
                            placeholder.markdown(answer)
                            if chunks:
                                _render_chunks(chunks)
                            st.session_state.messages.append({"role": "assistant", "content": answer, "chunks": chunks})
                        else:
                            err = f"Error {resp.status_code}: {resp.text}"
                            st.error(err)
                            st.session_state.messages.append({"role": "assistant", "content": err})
                except requests.exceptions.ConnectionError:
                    err = "Could not reach the API. Make sure the server is running: `uvicorn main:app --reload`"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

# ── Tools Agent tab ───────────────────────────────────────────────────────────

with tools_tab:
    if "tools_messages" not in st.session_state:
        st.session_state.tools_messages = []

    tools_box = st.container(height=560)

    with tools_box:
        for msg in st.session_state.tools_messages:
            with st.chat_message(msg["role"]):
                for tool_name in msg.get("tools", []):
                    st.badge(f"Tool: {tool_name}", color="orange")
                st.markdown(msg["content"])

    question = st.chat_input("Ask the tools agent…", key="tools_input")
    if question:
        st.session_state.tools_messages.append({"role": "user", "content": question})

        with tools_box:
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                answer = ""
                tools_used = []
                try:
                    with requests.post(
                        f"{API_BASE}/agent/run",
                        json={"query": question},
                        stream=True,
                        timeout=60,
                    ) as resp:
                        if resp.ok:
                            tools_placeholder = st.empty()
                            placeholder = st.empty()
                            for raw_line in resp.iter_lines():
                                if not raw_line:
                                    continue
                                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                                if not line.startswith("data: "):
                                    continue
                                event = json.loads(line[len("data: "):])
                                if event.get("type") == "tool_call":
                                    tools_used.append(event.get("content", ""))
                                    with tools_placeholder.container():
                                        for t in tools_used:
                                            st.badge(f"Tool: {t}", color="orange")
                                elif event.get("type") == "token":
                                    answer += event.get("content", "")
                                    placeholder.markdown(answer)
                            placeholder.markdown(answer)
                            st.session_state.tools_messages.append({"role": "assistant", "content": answer, "tools": tools_used})
                        else:
                            err = f"Error {resp.status_code}: {resp.text}"
                            st.error(err)
                            st.session_state.tools_messages.append({"role": "assistant", "content": err})
                except requests.exceptions.ConnectionError:
                    err = "Could not reach the API. Make sure the server is running: `uvicorn main:app --reload`"
                    st.error(err)
                    st.session_state.tools_messages.append({"role": "assistant", "content": err})

# ── Graph Agent tab ───────────────────────────────────────────────────────────

with graph_tab:
    st.caption("The supervisor graph routes your message to either the RAG agent or the Tools agent automatically.")

    if "graph_messages" not in st.session_state:
        st.session_state.graph_messages = []

    graph_box = st.container(height=520)

    with graph_box:
        for msg in st.session_state.graph_messages:
            with st.chat_message(msg["role"]):
                if msg.get("route"):
                    route_label = "RAG" if msg["route"] == "rag" else "Tools"
                    st.badge(f"Routed to: {route_label}", color="blue" if msg["route"] == "rag" else "green")
                st.markdown(msg["content"])
                if msg.get("chunks"):
                    _render_chunks(msg["chunks"])

    graph_query = st.chat_input("Ask anything — the graph will route it…", key="graph_input")
    if graph_query:
        st.session_state.graph_messages.append({"role": "user", "content": graph_query})

        with graph_box:
            with st.chat_message("user"):
                st.markdown(graph_query)

            with st.chat_message("assistant"):
                answer = ""
                chunks = []
                route = None
                try:
                    with requests.post(
                        f"{API_BASE}/chat/",
                        json={"message": graph_query},
                        stream=True,
                        timeout=60,
                    ) as resp:
                        if resp.ok:
                            route_placeholder = st.empty()
                            placeholder = st.empty()
                            for raw_line in resp.iter_lines():
                                if not raw_line:
                                    continue
                                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                                if not line.startswith("data: "):
                                    continue
                                payload = line[len("data: "):]
                                if payload == "[DONE]":
                                    break
                                event = json.loads(payload)
                                if event.get("type") == "route":
                                    route = event.get("content")
                                    route_label = "RAG" if route == "rag" else "Tools"
                                    route_placeholder.badge(
                                        f"Routed to: {route_label}",
                                        color="blue" if route == "rag" else "green",
                                    )
                                elif event.get("type") == "chunks":
                                    chunks = event.get("content", [])
                                elif event.get("type") == "token":
                                    answer += event.get("content", "")
                                    placeholder.markdown(answer)
                            placeholder.markdown(answer)
                            if chunks:
                                _render_chunks(chunks)
                            st.session_state.graph_messages.append({
                                "role": "assistant",
                                "content": answer,
                                "chunks": chunks,
                                "route": route,
                            })
                        else:
                            err = f"Error {resp.status_code}: {resp.text}"
                            st.error(err)
                            st.session_state.graph_messages.append({"role": "assistant", "content": err})
                except requests.exceptions.ConnectionError:
                    err = "Could not reach the API. Make sure the server is running: `uvicorn main:app --reload`"
                    st.error(err)
                    st.session_state.graph_messages.append({"role": "assistant", "content": err})
