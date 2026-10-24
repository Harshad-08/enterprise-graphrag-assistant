"""Streamlit UI for the Enterprise Knowledge Assistant.

Talks to the FastAPI backend over HTTP (API_BASE_URL, default http://localhost:8000).
Run: streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import json
import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = int(os.getenv("API_TIMEOUT_SECONDS", "300"))

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="📚", layout="wide")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def api(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, f"{API_BASE_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Cannot reach the API at {API_BASE_URL}: {exc}")
        return None
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"API error {response.status_code}: {detail}")
        return None
    return response.json()


def fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


# ---------------------------------------------------------------------------
# Sidebar: system status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 Knowledge Assistant")
    st.caption("Reliability-focused GraphRAG")
    health = api("GET", "/health")
    if health:
        st.metric("Indexed documents", health["documents_indexed"])
        st.write(f"**Graph:** {health['graph_backend']} ({'connected' if health['graph_connected'] else 'DISCONNECTED'})")
        st.write(f"**LLM:** {health['llm_provider']}")
        st.write(f"**Embeddings:** {health['embedding_backend']}")
        st.write(f"**Reranker:** {health['reranker']}")
        st.write(f"**Verifier:** {health['verifier']}")
        if health["mock_mode"]:
            st.warning(
                "MOCK LLM mode: answers are extractive (copied from evidence) and no language model is called. "
                "Set GROQ_API_KEY (or GEMINI_API_KEY) for real generation."
            )
        with st.expander("Graph statistics"):
            st.json(health.get("details", {}))

chat_tab, docs_tab, eval_tab = st.tabs(["💬 Chat", "📄 Documents", "📊 Evaluation"])

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
with chat_tab:
    st.subheader("Ask the enterprise knowledge base")
    col_mode, col_k = st.columns([3, 1])
    mode = col_mode.radio(
        "Pipeline", ["improved", "baseline"], horizontal=True,
        help="improved = GraphRAG + recovery + conflict handling + verification; baseline = plain vector RAG",
    )
    top_k = col_k.number_input("Top-k passages", min_value=1, max_value=20, value=6)

    examples = [
        "What is the company's work-from-home policy?",
        "Which security policies govern production database access?",
        "Which procedure depends on the API authentication policy?",
        "What are the access requirements for the Atlas cluster?",
        "What is the inactivity timeout for VPN sessions?",
        "How long must application logs be retained?",
        "What is the company's maternity leave policy?",
    ]
    example = st.selectbox("Example questions", ["—"] + examples)
    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.chat_input("Ask a question about company policies and procedures")
    if example != "—" and st.button("Ask example"):
        question = example

    if question:
        with st.spinner("Retrieving, verifying and checking for conflicts..."):
            result = api("POST", "/chat", json={"question": question, "mode": mode, "top_k": int(top_k)})
        if result:
            st.session_state.history.insert(0, (question, result))

    for asked, result in st.session_state.history:
        with st.chat_message("user"):
            st.write(asked)
        with st.chat_message("assistant"):
            badges = []
            if result["abstained"]:
                badges.append("🛑 Abstained")
            if result["verified"]:
                badges.append("✅ Claims verified")
            if result["recovery_performed"]:
                badges.append("🔁 Graph recovery")
            if result["conflict_detected"]:
                badges.append("⚠️ Conflict detected")
            if result["mock_mode"]:
                badges.append("🧪 Mock LLM")
            st.caption(" · ".join(badges + [f"mode: {result['mode']}", f"{result.get('latency_ms') or 0:.0f} ms"]))

            if result["abstained"]:
                st.warning(result["answer"])
            else:
                st.markdown(result["answer"].replace("\n", "  \n"))

            for conflict in result.get("conflicts", []):
                box = st.info if conflict["resolved"] else st.error
                box(
                    f"**{'Resolved' if conflict['resolved'] else 'Unresolved'} conflict** on "
                    f"{conflict.get('subject') or 'a rule'} ({conflict.get('property') or 'n/a'}):\n\n"
                    f"- {conflict['claim_a']} — *{conflict['claim_a_document']}*\n"
                    f"- {conflict['claim_b']} — *{conflict['claim_b_document']}*\n\n"
                    f"{conflict['explanation']}"
                )

            if result["citations"]:
                st.markdown("**Sources**")
                for citation in result["citations"]:
                    location = " · ".join(
                        x for x in [
                            f"page {citation['page']}" if citation.get("page") else None,
                            citation.get("section"),
                            f"v{citation['version']}" if citation.get("version") else None,
                            citation.get("document_status"),
                        ] if x
                    )
                    with st.expander(f"[{citation['citation_id']}] {citation['document_title']} ({citation['document_id']}) {location}"):
                        st.write(f"> {citation['excerpt']}")
                        st.caption(f"Chunk: {citation['chunk_id']}")

            with st.expander("Reliability details"):
                st.write(f"**Retrieval outcome:** {result.get('retrieval_outcome') or 'n/a'}")
                for note in result.get("reliability_notes", []):
                    st.write(f"- {note}")
                if result.get("graph_paths"):
                    st.write("**Graph paths used:**")
                    for path in result["graph_paths"]:
                        st.code(path, language=None)
                if result.get("claim_verifications"):
                    st.dataframe(
                        pd.DataFrame(result["claim_verifications"])[["status", "claim", "cited_ids", "supporting_ids", "support_score", "method", "note"]],
                        width="stretch",
                    )

# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
with docs_tab:
    st.subheader("Upload a document")
    with st.form("upload", clear_on_submit=True):
        uploaded = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
        c1, c2, c3 = st.columns(3)
        title = c1.text_input("Title (optional)")
        doc_id = c2.text_input("Document ID (optional)")
        doc_type = c3.selectbox("Type", ["", "policy", "standard", "procedure", "guideline", "technical", "register"])
        c4, c5, c6 = st.columns(3)
        status = c4.selectbox("Status", ["", "active", "draft", "superseded", "archived"])
        authority = c5.selectbox("Authority", ["", "board", "executive", "department", "team", "individual"])
        version = c6.text_input("Version")
        c7, c8, c9 = st.columns(3)
        department = c7.text_input("Owning department")
        effective = c8.text_input("Effective date (YYYY-MM-DD)")
        supersedes = c9.text_input("Supersedes (document IDs; semicolon-separated)")
        submitted = st.form_submit_button("Upload and index")
    if submitted:
        if uploaded is None:
            st.error("Choose a file first.")
        else:
            form = {k: v for k, v in {
                "title": title, "document_id": doc_id, "doc_type": doc_type, "status": status,
                "authority": authority, "version": version, "department": department,
                "effective_date": effective, "supersedes": supersedes,
            }.items() if v}
            with st.spinner("Parsing, extracting entities and claims, indexing..."):
                result = api("POST", "/documents/upload", files={"file": (uploaded.name, uploaded.getvalue())}, data=form)
            if result:
                st.success(
                    f"Indexed {result['document_id']}: {result['chunk_count']} chunks, {result['claim_count']} claims, "
                    f"{result['relation_count']} relationships, {result['conflicts_detected']} new contradiction(s)."
                )
                for warning in result.get("warnings", []):
                    st.caption(f"⚠ {warning}")

    st.subheader("Indexed documents")
    documents = api("GET", "/documents") or []
    if documents:
        rows = [{
            "document_id": d["metadata"]["document_id"],
            "title": d["metadata"]["title"],
            "type": d["metadata"]["doc_type"],
            "version": d["metadata"].get("version"),
            "status": d["metadata"]["status"],
            "authority": d["metadata"].get("authority"),
            "effective": d["metadata"].get("effective_date"),
            "chunks": d["chunk_count"],
            "claims": d["claim_count"],
            "indexing": d["indexing_status"],
        } for d in documents]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        selected = st.selectbox("Select a document", [r["document_id"] for r in rows])
        detail_col, delete_col = st.columns([4, 1])
        with detail_col.expander("Metadata"):
            st.json(next(d for d in documents if d["metadata"]["document_id"] == selected)["metadata"])
        confirm = delete_col.checkbox("Confirm delete")
        if delete_col.button("Delete", disabled=not confirm, type="primary"):
            if api("DELETE", f"/documents/{selected}"):
                st.success(f"Deleted {selected} (graph nodes, vectors and keyword index entries removed).")
                st.rerun()
    else:
        st.info("No documents indexed yet. Upload files or run scripts/seed_synthetic_data.py.")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
with eval_tab:
    st.subheader("Evaluate baseline vs improved pipeline")
    st.caption(
        "Runs the labelled synthetic dataset through both configurations. Ragas metrics need GROQ_API_KEY or GEMINI_API_KEY; "
        "without it they are reported as skipped. Nothing here is hard-coded."
    )
    c1, c2 = st.columns(2)
    include_ragas = c1.checkbox("Include Ragas metrics (LLM-judged, slower, uses API quota)")
    limit = c2.number_input("Limit items (0 = all)", min_value=0, value=0)
    if st.button("Run evaluation", type="primary"):
        payload = {"configurations": ["baseline", "improved"], "include_ragas": include_ragas}
        if limit:
            payload["limit"] = int(limit)
        with st.spinner("Running evaluation — this can take a few minutes..."):
            st.session_state.eval_report = api("POST", "/evaluation/run", json=payload)
    if "eval_report" not in st.session_state:
        latest = requests.get(f"{API_BASE_URL}/evaluation/latest", timeout=TIMEOUT) if health else None
        if latest is not None and latest.status_code == 200:
            st.session_state.eval_report = latest.json()
            st.caption("Showing the most recent saved evaluation run.")

    report = st.session_state.get("eval_report")
    if report:
        env = report.get("environment", {})
        st.write(f"Generated: {report.get('generated_at')} · items: {report.get('item_count')} · LLM: {env.get('llm_provider')}")
        if env.get("mock_mode"):
            st.warning("These results were produced in MOCK LLM mode (extractive answers).")
        configs = report["configurations"]
        rows = [{"metric": m, **{c: report["custom_metrics"][c].get(m) for c in configs}} for m in report["custom_metrics"][configs[0]]]
        st.markdown("**Custom reliability metrics**")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        corpus = report.get("corpus_conflict_metrics", {})
        if corpus:
            st.markdown("**Corpus-level contradiction detection**")
            det = corpus["detection"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision", fmt(det.get("precision")))
            m2.metric("Recall", fmt(det.get("recall")))
            m3.metric("Resolution accuracy", fmt(corpus.get("resolution_accuracy")))
            m4.metric("Compatible pairs flagged", corpus.get("compatible_pairs_flagged_as_contradiction"))

        ragas = report.get("ragas", {})
        st.markdown("**Ragas metrics**")
        if ragas.get("status") in ("completed", "partial"):
            st.caption(f"Judges: {ragas.get('judge_models')}")
            if ragas.get("status") == "partial":
                st.warning(ragas.get("reason"))
            matched_rows = [
                {"metric": name, "matched items": m.get("items"), **{c: m.get(c) for c in configs}}
                for name, m in ragas.get("matched", {}).items()
            ]
            st.markdown("Matched comparison (questions scored for every configuration):")
            st.dataframe(pd.DataFrame(matched_rows), width="stretch", hide_index=True)
        else:
            st.info(f"Ragas {ragas.get('status', 'not run')}: {ragas.get('reason') or ''}")

        with st.expander("Per-item results"):
            if report.get("items"):
                items = pd.DataFrame(report["items"])
                st.dataframe(items[["configuration", "item_id", "category", "abstained", "should_abstain", "retrieval_outcome",
                                    "recovery_triggered", "claims_supported", "claims_total", "answer"]], width="stretch")
        for note in report.get("notes", []):
            st.caption(f"Note: {note}")
        st.download_button("Download report (JSON)", json.dumps(report, indent=2, default=str), "evaluation_report.json", "application/json")
        csv_df = pd.DataFrame(rows)
        st.download_button("Download comparison (CSV)", csv_df.to_csv(index=False), "evaluation_comparison.csv", "text/csv")
