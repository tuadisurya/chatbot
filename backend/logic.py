import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# ======================= API KEY =======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("❌ OPENAI_API_KEY belum diatur di environment variable.")

# ======================= CHAT MODEL =======================
llm = ChatOpenAI(
    model_name="gpt-4-turbo",
    temperature=0.2,
    openai_api_key=OPENAI_API_KEY
)

# ======================= PROMPT TEMPLATE =======================
rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Anda adalah asisten AI kampus INSTIKI yang cerdas, ramah, dan faktual. Jawaban Anda **hanya boleh berdasarkan informasi yang ada di konteks** berikut ini, tanpa mengarang.

Tugas Anda:
- Jawab dengan **bahasa natural**, jelas, dan ringkas.
- Jika pertanyaan berkaitan dengan **dosen**, sertakan (jika tersedia):
  • Nama lengkap
  • NIDN
  • Daftar mata kuliah yang diajar, berikut:
    - Nama matkul
    - Kelas
    - Hari & jam
    - Ruangan

- Jika pertanyaan tentang **mata kuliah**, berikan:
  • Jadwal (hari dan jam)
  • Ruangan
  • Dosen pengampu

- Gunakan format markdown agar mudah dibaca:
  - Gunakan **bold** untuk nama penting
  - Gunakan bullet `•` dan penomoran `1. 2. 3.` jika daftar
  - Tampilkan sebagai daftar jika lebih dari satu hasil

Jika informasi tidak ditemukan di konteks, katakan dengan sopan: "Maaf, informasi tersebut tidak tersedia dalam data yang saya miliki."

---

📥 Pertanyaan:
{question}

📚 Konteks Dokumen:
{context}
"""
)

# ======================= LOAD VECTORSTORE =======================
def load_vectorstore():
    db_path = "vectorstores/matkul_db_faiss"
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"❌ Database FAISS tidak ditemukan di: {db_path}")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=OPENAI_API_KEY
    )
    faiss = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    return faiss.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# ======================= CONTEXT RETRIEVAL =======================
def get_relevant_context(question, retriever, max_chars=3000):
    docs = retriever.get_relevant_documents(question)

    unique_texts = []
    seen = set()
    total_chars = 0

    for doc in docs:
        text = doc.page_content.strip()
        if text in seen:
            continue
        if total_chars + len(text) > max_chars:
            break
        unique_texts.append(text)
        seen.add(text)
        total_chars += len(text)

    combined_context = "\n\n".join(unique_texts)
    return combined_context, "Data diambil dari: vectorstores/matkul_db_faiss"

# ======================= FINAL ANSWER EXTRACTION =======================
def extract_final_answer(full_output: str) -> str:
    if "Final Answer:" in full_output:
        return full_output.split("Final Answer:")[-1].strip()
    return full_output.strip()

# ======================= MAIN FUNCTION =======================
def generate_answer(question: str, _=None):
    try:
        retriever = load_vectorstore()
        rag_context, referensi = get_relevant_context(question, retriever)

        if not rag_context:
            return {
                "rag": "Maaf, saya tidak menemukan informasi yang relevan dalam dokumen.",
                "context": "Tidak ada context ditemukan.",
                "referensi": "Tidak ada referensi."
            }

        prompt = rag_prompt.format(context=rag_context, question=question)

        response = llm.predict(prompt)
        final_answer = extract_final_answer(response)

        return {
            "rag": final_answer,
            "context": rag_context,
            "referensi": referensi
        }

    except Exception as e:
        return {
            "rag": f"❌ Terjadi kesalahan saat memproses pertanyaan: {str(e)}",
            "context": "Gagal memuat konteks.",
            "referensi": "Tidak ada"
        }
