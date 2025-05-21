import streamlit as st
import os
from io import StringIO
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from backend.logic import generate_answer

# ================= ENV ===================
load_dotenv()
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    st.error("❌ OPENAI_API_KEY belum ditemukan di environment!")
    st.stop()

# ================= CONFIG ===================
st.set_page_config(page_title="Chatbot - Info INSTIKI", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "references" not in st.session_state:
    st.session_state.references = []
if "contexts" not in st.session_state:
    st.session_state.contexts = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# ================= STYLE ===================
st.markdown("""
<style>
.bubble-user {
    background-color: #1e40af;
    color: white;
    padding: 0.8rem 1.2rem;
    border-radius: 16px;
    max-width: 70%;
    margin-left: auto;
    margin-bottom: 1rem;
    font-size: 0.95rem;
    font-family: 'Segoe UI', sans-serif;
}
.bubble-assistant {
    background-color: transparent;
    color: #0f172a;
    padding: 0;
    max-width: 100%;
    margin-right: auto;
    margin-bottom: 1rem;
    font-size: 0.95rem;
    font-family: 'Segoe UI', sans-serif;
}
.result-card {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    font-size: 1rem;
    line-height: 1.6;
}
.stDownloadButton > button {
    background-color: #10b981;
    color: white;
    padding: 0.6rem 1rem;
    border-radius: 0.5rem;
    font-weight: 600;
    margin-top: 1rem;
}
.stDownloadButton > button:hover {
    background-color: #059669;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER ===================
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between;">
    <h2>🤖 <span style='color:#38bdf8'>Info INSTIKI</span></h2>
</div>
""", unsafe_allow_html=True)

st.info("📁 Menggunakan database: `matkul_db_faiss` dari folder `vectorstores/`")

# ================= FORMAT CARD ===================
def format_result_card(text):
    return f"<div class='result-card'>{text}</div>"

# ================= CHAT HISTORY ===================
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(f'<div class="bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            formatted = format_result_card(msg["content"])
            st.markdown(f'<div class="bubble-assistant">{formatted}</div>', unsafe_allow_html=True)

            context_text = st.session_state.contexts[i // 2] if i // 2 < len(st.session_state.contexts) else "Tidak ada context."
            referensi_text = st.session_state.references[i // 2] if i // 2 < len(st.session_state.references) else "Tidak ada referensi."

            with st.expander("📚 Referensi Jawaban", expanded=False):
                st.markdown("#### 📘 Context Digunakan:")
                st.markdown(context_text)
                st.markdown("---")
                st.markdown("#### 📄 Referensi Dokumen:")
                st.markdown(f"**{referensi_text}**")

# ================= INPUT CHAT ===================
input_prompt = st.chat_input("Ketik pertanyaanmu...")
if input_prompt:
    # Tampilkan bubble user
    with st.chat_message("user"):
        st.markdown(f'<div class="bubble-user">{input_prompt}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": input_prompt})

    # Jawaban dari model
    with st.spinner("✅ Info INSTIKI sedang berpikir..."):
        result = generate_answer(input_prompt)

    st.session_state.messages.append({"role": "assistant", "content": result["rag"]})
    st.session_state.references.append(result["referensi"])
    st.session_state.contexts.append(result["context"])

    # Tampilkan hasil assistant + referensi
    with st.chat_message("assistant"):
        formatted = format_result_card(result["rag"])
        st.markdown(f"<div class='bubble-assistant'>{formatted}</div>", unsafe_allow_html=True)

        with st.expander("📚 Referensi Jawaban", expanded=False):
            st.markdown("#### 📘 Context Digunakan:")
            st.markdown(result["context"])
            st.markdown("---")
            st.markdown("#### 📄 Referensi Dokumen:")
            st.markdown(f"**{result['referensi']}**")

# ================= EXPORT CSV ===================
export_data = []
ref_idx = 0
for msg_idx in range(0, len(st.session_state.messages), 2):
    try:
        question = st.session_state.messages[msg_idx]["content"]
        answer = st.session_state.messages[msg_idx + 1]["content"]
        reference = st.session_state.references[ref_idx] if ref_idx < len(st.session_state.references) else ""
        context = st.session_state.contexts[ref_idx] if ref_idx < len(st.session_state.contexts) else ""
        ref_idx += 1
        export_data.append({
            "Pertanyaan": question,
            "Jawaban": answer,
            "Context": context,
            "Referensi": reference
        })
    except IndexError:
        continue

if export_data:
    df_export = pd.DataFrame(export_data)
    csv_buffer = StringIO()
    df_export.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Export ke CSV",
        data=csv_buffer.getvalue(),
        file_name=f"riwayat_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="export_csv"
    )

# ================= RESET ===================
st.button("Reset Percakapan", on_click=lambda: st.session_state.clear())
