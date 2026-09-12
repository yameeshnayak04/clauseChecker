import os
import sys
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import streamlit as st
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import retrieve
from classifier import classify_and_answer
from config import CONFIDENCE_THRESHOLD, TOP_K

st.set_page_config(page_title="ClauseCheck", page_icon="⚖️", layout="wide")

# Custom CSS styling for visually distinct states and taste
st.markdown("""
<style>
.state-badge {
    display: inline-block;
    padding: 6px 14px;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    border-radius: 20px;
    margin-bottom: 12px;
}
.badge-answered {
    background-color: #d1e7dd;
    color: #0f5132;
    border: 1px solid #badbcc;
}
.badge-not-in-corpus {
    background-color: #e2e3e5;
    color: #41464b;
    border: 1px solid #d3d6d8;
}
.badge-contradiction {
    background-color: #f8d7da;
    color: #842029;
    border: 1px solid #f5c2c7;
}
.box-answered {
    background-color: #f0fdf4;
    border-left: 6px solid #22c55e;
    padding: 18px 20px;
    border-radius: 8px;
    margin-bottom: 15px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.box-not-in-corpus {
    background-color: #f8fafc;
    border-left: 6px solid #94a3b8;
    padding: 18px 20px;
    border-radius: 8px;
    margin-bottom: 15px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.box-contradiction {
    background-color: #fef2f2;
    border-left: 6px solid #ef4444;
    padding: 18px 20px;
    border-radius: 8px;
    margin-bottom: 15px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.citation-box {
    background-color: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 12px 16px;
    margin-top: 10px;
    font-size: 0.92rem;
}
.highlight-text {
    background-color: #fef08a;
    padding: 2px 4px;
    border-radius: 3px;
    font-weight: 500;
}
.conflict-card-a {
    background-color: #fff1f2;
    border: 1px solid #fecdd3;
    border-radius: 8px;
    padding: 14px;
}
.conflict-card-b {
    background-color: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 14px;
}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ ClauseCheck: Contradiction-Aware Academic Assistant")
st.caption("A reliable regulation assistant that answers from documents only, classifies its certainty into 3 explicit states, and surfaces planted policy contradictions.")

tab_query, tab_calibration, tab_contradictions = st.tabs([
    "🔍 Regulation Query & Assistant",
    "📈 Calibration & The Answer/Refuse Boundary",
    "📋 Planted Contradictions Ledger"
])

with tab_query:
    col_input, col_meta = st.columns([3, 1])
    
    with col_meta:
        st.markdown("### Controls")
        conf_thresh = st.slider(
            "Confidence Threshold",
            0.0, 1.0, float(CONFIDENCE_THRESHOLD), 0.05,
            help="Threshold below which an ANSWERED state is safely demoted to NOT_IN_CORPUS."
        )
        top_k_val = st.slider("Top-K Chunks", 1, 10, int(TOP_K), 1)
        category_filter = st.selectbox("Document Filter", ["All", "attendance", "fees", "scholarship", "hostel"])
        if category_filter == "All":
            category_filter = None

    with col_input:
        st.markdown("### Test Queries")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🟢 Ex: Normal Query (Answered)", width="stretch"):
                st.session_state.query_input = "What are the scholarship GPA requirements?"
        with c2:
            if st.button("⚪ Ex: Near-Miss (Not in Corpus)", width="stretch"):
                st.session_state.query_input = "What happens if I miss the exam because of a family wedding?"
        with c3:
            if st.button("🔴 Ex: Planted Contradiction", width="stretch"):
                st.session_state.query_input = "Can overnight guests stay at the hostel?"
                
        user_query = st.text_input(
            "Enter question about academic regulations:",
            value=st.session_state.get("query_input", ""),
            placeholder="e.g., What is the minimum attendance required to sit for an exam?"
        )
        submit_btn = st.button("Analyze Question", type="primary")

    if submit_btn or (user_query and "prev_query" in st.session_state and st.session_state.prev_query != user_query):
        st.session_state.prev_query = user_query
        
        with st.spinner("Retrieving clauses and analyzing regulations..."):
            chunks = retrieve(user_query, category=category_filter, top_k=top_k_val)
            res = classify_and_answer(user_query, chunks, conf_thresh)
            
        st.markdown("---")
        
        # State Presentation
        if res.state == "ANSWERED":
            st.markdown(f'<span class="state-badge badge-answered">STATE: ANSWERED</span>', unsafe_allow_html=True)
            st.markdown(f'''
            <div class="box-answered">
                <h4 style="margin-top:0; color:#15803d;">Direct Answer from Regulations</h4>
                <p style="font-size:1.05rem; line-height:1.6;">{res.answer}</p>
            </div>
            ''', unsafe_allow_html=True)
            
            if res.citations:
                st.markdown("#### 📄 Inline Passage Citations")
                for cit in res.citations:
                    # Find chunk text to show exact passage inline
                    matching_c = next((c for c in chunks if str(c.id) == str(cit.chunk_id)), None)
                    passage_text = matching_c.text if matching_c else "Text extracted from verified passage."
                    st.markdown(f'''
                    <div class="citation-box">
                        <strong>Source:</strong> <code>{cit.source_file}</code> &nbsp;|&nbsp; 
                        <strong>Section:</strong> <code>{cit.section_title}</code> &nbsp;|&nbsp; 
                        <strong>Chunk ID:</strong> <code>{cit.chunk_id}</code>
                        <div style="margin-top:8px; padding:10px; background:#ffffff; border-left:3px solid #3b82f6; border-radius:4px;">
                            <em>"{passage_text}"</em>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)

        elif res.state == "NOT_IN_CORPUS":
            st.markdown(f'<span class="state-badge badge-not-in-corpus">STATE: NOT_IN_CORPUS</span>', unsafe_allow_html=True)
            st.markdown(f'''
            <div class="box-not-in-corpus">
                <h4 style="margin-top:0; color:#475569;">Question Not Covered in Rulebook</h4>
                <p style="font-size:1.05rem; line-height:1.6;">{res.answer}</p>
                <p style="font-size:0.9rem; color:#64748b; margin-bottom:0;">
                    <strong>Integrity Note:</strong> Outside knowledge is strictly forbidden. The system correctly refuses rather than guessing.
                </p>
            </div>
            ''', unsafe_allow_html=True)

        elif res.state == "CONTRADICTION":
            st.markdown(f'<span class="state-badge badge-contradiction">STATE: CONTRADICTION DETECTED</span>', unsafe_allow_html=True)
            st.markdown(f'''
            <div class="box-contradiction">
                <h4 style="margin-top:0; color:#b91c1c;">⚠️ Conflicting Academic Regulations Found</h4>
                <p style="font-size:1.05rem; line-height:1.6;">{res.answer}</p>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown("#### ⚖️ Conflicting Clauses (Side-by-Side Comparison)")
            # Display conflicting chunks side by side
            chunk_map = {str(c.id): c for c in chunks}
            
            if res.conflicting_chunks:
                for pair in res.conflicting_chunks:
                    col_a, col_b = st.columns(2)
                    cid_a = str(pair[0]) if len(pair) > 0 else None
                    cid_b = str(pair[1]) if len(pair) > 1 else None
                    
                    with col_a:
                        ca = chunk_map.get(cid_a)
                        st.markdown(f"**Position A:** `{ca.source_file if ca else 'Clause A'}`")
                        if ca:
                            st.info(f"**{ca.section_title}**\n\n{ca.text}")
                        else:
                            st.info("Referenced clause from corpus.")
                            
                    with col_b:
                        cb = chunk_map.get(cid_b)
                        st.markdown(f"**Position B:** `{cb.source_file if cb else 'Clause B'}`")
                        if cb:
                            st.warning(f"**{cb.section_title}**\n\n{cb.text}")
                        else:
                            st.warning("Referenced contradicting clause from corpus.")
            else:
                # Fallback if specific pair wasn't formatted
                cols = st.columns(min(len(res.citations), 3) or 2)
                for i, cit in enumerate(res.citations[:3]):
                    with cols[i]:
                        c = chunk_map.get(str(cit.chunk_id))
                        st.markdown(f"**Conflict Source {i+1}:** `{cit.source_file}`")
                        st.error(f"**{cit.section_title}**\n\n{c.text if c else ''}")

        # Metrics bar
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns([1, 1, 2])
        with m1:
            st.metric("Model Confidence", f"{res.confidence*100:.0f}%")
        with m2:
            st.metric("State Status", res.state)
        with m3:
            st.caption(f"**Classifier Reasoning:** {res.reasoning_note}")

        # Raw chunks inspection
        with st.expander("🔍 Inspect All Retrieved Chunks (Similarity Scores & Metadata)"):
            for i, c in enumerate(chunks, 1):
                st.markdown(f"**Chunk #{i}** | ID: `chunk_{c.id}` | Similarity Score: `Cosine {c.score:.4f}` | Source: `{c.source_file}` ({c.section_title})")
                st.code(c.text, language="markdown")

with tab_calibration:
    st.markdown("### The Calibration Experiment: Measuring the Answer vs. Refuse Boundary")
    st.markdown("""
    The central challenge in building a contradiction-aware, high-certainty assistant is navigating the boundary 
    between being overly confident (hallucinating answers for unanswerable questions) and overly cautious 
    (refusing questions whose answers are in the text).
    
    Rather than guessing or tuning by vibe, **ClauseCheck systematically sweeps confidence thresholds** 
    to measure empirical accuracy curves for both **ANSWERED** questions and **NOT_IN_CORPUS** questions.
    """)
    
    chart_paths = [
        Path("frontend/calibration_chart.png"),
        Path("eval/calibration_chart.png")
    ]
    chart_found = False
    for p in chart_paths:
        if p.exists():
            st.image(str(p), caption="Empirical Calibration Curve: ANSWERED vs NOT_IN_CORPUS Trade-off", width="stretch")
            chart_found = True
            break
            
    if not chart_found:
        st.info("Run `uv run python eval/calibrate.py` to generate the measured calibration chart.")
        
    eval_results_file = Path("eval/eval_results.json")
    if eval_results_file.exists():
        import json
        with open(eval_results_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            
        st.markdown("#### Latest Evaluation Run Metrics")
        em1, em2, em3, em4 = st.columns(4)
        em1.metric("Overall Accuracy", f"{eval_data.get('overall_accuracy', 0)*100:.1f}%")
        state_acc = eval_data.get('state_accuracy', {})
        em2.metric("ANSWERED Accuracy", f"{state_acc.get('ANSWERED', 0)*100:.1f}%")
        em3.metric("NOT_IN_CORPUS Accuracy", f"{state_acc.get('NOT_IN_CORPUS', 0)*100:.1f}%")
        em4.metric("CONTRADICTION Accuracy", f"{state_acc.get('CONTRADICTION', 0)*100:.1f}%")

with tab_contradictions:
    st.markdown("### Planted Contradictions Ledger")
    st.markdown("Documented real policy desynchronizations planted across different regulation documents:")
    
    contradictions_md = Path("contradictions.md")
    if contradictions_md.exists():
        st.markdown(contradictions_md.read_text(encoding="utf-8"))
    else:
        st.info("`contradictions.md` file not found.")
