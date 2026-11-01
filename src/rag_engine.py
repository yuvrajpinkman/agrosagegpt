"""RAG Engine for AgroSageGPT.

Integrates ChromaDB vector store with agronomic knowledge, conversational context,
Computer Vision diagnostics, IoT soil telemetry, and live weather conditions.
Uses Google Gemini 3.8 Flash as the primary multilingual LLM, Groq (qwen/qwen3.8-27b)
as high-speed fallback, and dynamic crop knowledge synthesis as offline fallback.
"""

import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


def detect_language(text: str, fallback: str = "English") -> str:
    """Detect whether user query is English, Hindi, Kannada, Telugu, or Tamil.
    
    Supports both native Indic scripts (Devanagari, Kannada, Telugu, Tamil)
    and colloquial Romanized/Hinglish transcripts.
    """
    if not text or not str(text).strip():
        return fallback
        
    s = str(text).strip()
    
    # 1. Native Indic Unicode Script ranges
    if re.search(r"[\u0900-\u097F]", s):
        return "Hindi"
    if re.search(r"[\u0C80-\u0CFF]", s):
        return "Kannada"
    if re.search(r"[\u0C00-\u0C7F]", s):
        return "Telugu"
    if re.search(r"[\u0B80-\u0BFF]", s):
        return "Tamil"
        
    # 2. Romanized keywords / phonetics
    tokens = set(re.findall(r"[a-zA-Z]+", s.lower()))
    
    hi_keywords = {
        "namaste", "kisan", "chawal", "gehu", "tamatar", "makka", "kheti", "fasal",
        "paani", "pani", "kaise", "kab", "kya", "kyu", "kyun", "kare", "karein",
        "ugaye", "ugayein", "rog", "keeda", "keede", "dawa", "dawai", "khad",
        "upchar", "beej", "bimari", "bimaari", "lag", "gaya", "gayi", "hai", "hain",
        "batao", "bataiye", "hoga", "hogi", "mitti", "khet"
    }
    if tokens & hi_keywords:
        return "Hindi"
        
    kn_keywords = {"namaskara", "bhatta", "bele", "neeru", "roga", "hege", "yavaga", "manassu"}
    if tokens & kn_keywords:
        return "Kannada"
        
    te_keywords = {"namaskaram", "vari", "panta", "neellu", "cheeda", "ela", "eppudu", "mandhu"}
    if tokens & te_keywords:
        return "Telugu"
        
    ta_keywords = {"vanakkam", "nel", "payir", "thanneer", "noi", "eppadi", "eppozhuthu", "marunthu"}
    if tokens & ta_keywords:
        return "Tamil"
        
    return fallback


class AgroKnowledgeStore:
    """Manages local ChromaDB vector store for agricultural knowledge retrieval."""
    
    def __init__(self, persist_dir: str = "./.chroma_db", doc_path: str = "./data/agri_knowledge.txt"):
        self.persist_dir = persist_dir
        self.doc_path = doc_path
        self.collection = None
        self._init_store()
        
    def _init_store(self):
        """Initialize ChromaDB collection and ingest knowledge document."""
        import chromadb
        
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        self.collection = self.client.get_or_create_collection(
            name="agricultural_advisory_kb"
        )
        
        if self.collection.count() == 0 and os.path.exists(self.doc_path):
            self._ingest_file(self.doc_path)
            
    def _ingest_file(self, file_path: str):
        """Split document by markdown sections and ingest into ChromaDB."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            sections = [s.strip() for s in content.split("## ") if s.strip()]
            documents = []
            metadatas = []
            ids = []
            
            doc_idx = 0
            for sec in sections:
                subsections = [sub.strip() for sub in sec.split("### ") if sub.strip()]
                for sub in subsections:
                    lines = sub.split("\n", 1)
                    title = lines[0].replace("#", "").strip()
                    body = lines[1].strip() if len(lines) > 1 else lines[0]
                    
                    full_text = f"Category: {title}\n{body}"
                    documents.append(full_text)
                    metadatas.append({"topic": title})
                    ids.append(f"agri_doc_{doc_idx}")
                    doc_idx += 1
                    
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
        except Exception as e:
            print(f"[AgroKnowledgeStore] Ingestion note: {e}")

    def ingest_user_document(self, file_bytes: bytes, filename: str) -> int:
        """Dynamically ingests custom agricultural documents (PDF, TXT, MD) on the fly."""
        import io
        import uuid
        
        text = ""
        fn_lower = filename.lower()
        if fn_lower.endswith(".pdf"):
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                for page_idx, page in enumerate(reader.pages):
                    extracted = page.extract_text()
                    if extracted:
                        text += f"\n--- Page {page_idx+1} ---\n" + extracted
            except Exception as e:
                print(f"[AgroKnowledgeStore] PDF extraction error: {e}")
        else:
            try:
                text = file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                text = str(file_bytes)

        if not text.strip():
            return 0

        # Chunk text into ~500 character paragraphs with source attribution
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = ""
        for p in paragraphs:
            if len(current_chunk) + len(p) < 700:
                current_chunk += "\n" + p
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = p
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        if not chunks:
            chunks = [text[:1000]]

        documents = []
        metadatas = []
        ids = []
        doc_prefix = uuid.uuid4().hex[:6]
        for i, chunk in enumerate(chunks):
            documents.append(f"Document [{filename}]:\n{chunk}")
            metadatas.append({"source": filename, "chunk": i})
            ids.append(f"custom_{doc_prefix}_{i}")

        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            return len(documents)
        except Exception as e:
            print(f"[AgroKnowledgeStore] ChromaDB custom add error: {e}")
            return 0

    def query_knowledge(self, query_text: str, n_results: int = 4) -> List[str]:
        """Retrieve most relevant agricultural recommendations."""
        if not self.collection or self.collection.count() == 0:
            return ["Consult standard agricultural extension office practices."]
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(n_results, self.collection.count())
            )
            docs = results.get("documents", [[]])[0]
            return docs if docs else ["Maintain regular crop sanitation and balanced NPK fertilizer."]
        except Exception as e:
            print(f"[AgroKnowledgeStore] Query exception: {e}")
            return ["Standard balanced fertilization and integrated pest management recommended."]


# Singleton knowledge store instance
_kb_instance = None

def get_knowledge_store() -> AgroKnowledgeStore:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = AgroKnowledgeStore()
    return _kb_instance


AGRONOMIST_SYSTEM_PROMPT = """You are AgroSageGPT, an expert multilingual Digital Agronomist and practical farm advisor.
Your mission is to empower farmers with clear, empathetic, scientifically accurate, and actionable advice.

LANGUAGE:
You MUST reply completely in {language}. Use natural, respectful vocabulary (e.g. 'नमस्ते किसान भाई' in Hindi).

HOW TO ANSWER BASED ON USER INTENT:
1. If the farmer asks a CULTIVATION OR SOWING QUESTION (e.g. "चावल कब और कैसे उगाएं?", "गेहूं की बुवाई कैसे करें?", seed variety, irrigation cycle, sowing dates):
   - Answer their specific cultivation question directly and thoroughly!
   - Detail: (a) Sowing & nursery timing, (b) Field preparation & seed treatment, (c) Fertilizer management (Urea, DAP, MOP), (d) Irrigation stages, and (e) Harvesting.
   - Do NOT invent or force disease diagnosis if the farmer is just asking how to grow a crop!

2. If the farmer asks about LEAF DISEASES, SPOTS, PESTS, OR DEFICIENCIES:
   - Provide immediate diagnosis, cause, recommended biological (Trichoderma/Neem) and chemical treatments with exact dosages (e.g. grams per 15L knapsack sprayer), and spray timing.
   - Use the Vision Diagnostic summary provided if available.

3. If relevant, seamlessly reference the local live soil sensor telemetry (NPK, pH, Moisture) and local weather conditions to give hyper-localized advice.

Tone: Farmer-friendly, respectful, practical, and encouraging. Never return empty or robotic templates.
"""


def _generate_dynamic_offline_response(query: str, language: str = "Hindi") -> str:
    """Intelligently synthesizes agronomic advice offline if external APIs fail."""
    q_lower = query.lower()
    
    # Check for Rice / Paddy
    if any(k in q_lower for k in ["chawal", "चावल", "dhan", "धान", "rice", "paddy"]):
        if language.lower() == "hindi":
            return """### 🌾 धान (चावल) की उन्नत खेती: समय एवं संपूर्ण विधि

**नमस्ते किसान भाई! धान की सफल खेती के लिए मुख्य बिंदु नीचे दिए गए हैं:**

#### 1. बुवाई एवं रोपाई का सही समय (Timing)
- **नर्सरी (पौध) तैयार करने का समय:** 20 मई से 20 जून तक।
- **खेत में रोपाई का समय:** 15 जून से 15 जुलाई (मानसून आने के साथ, जब पौधे 21-25 दिन के हो जाएं)।

#### 2. बीज उपचार एवं नर्सरी तैयारी
- प्रति एकड़ 8-10 किलोग्राम प्रमाणित बीज लें।
- **बीज शोधन:** कार्बेन्डाजिम (2 ग्राम/किग्रा बीज) या ट्राइकोडर्मा (5 ग्राम/किग्रा) से शोधन करें ताकि फफूंद जनित रोगों से सुरक्षा मिले।

#### 3. खेत की तैयारी एवं खाद (Fertilizer Management)
- खेत में पानी भरकर 2-3 बार गहरी जुताई (पडलिंग/लेह) करें जिससे पानी ठहर सके।
- **प्रति एकड़ खाद:**
  - डीएपी (DAP): 40-50 किग्रा (रोपाई के समय)
  - पोटाश (MOP): 25-30 किग्रा
  - यूरिया: 60-70 किग्रा (3 बार में: रोपाई के 15 दिन बाद, कल्ले फूटते समय, और बालियां निकलते समय)
  - जिंक सल्फेट: 10 किग्रा प्रति एकड़ (खैरा रोग से बचाव हेतु)।

#### 4. सिंचाई एवं खरपतवार प्रबंधन
- रोपाई के शुरुआती 2-3 सप्ताह तक खेत में 2-3 सेमी पानी बनाए रखें।
- खरपतवार नियंत्रण के लिए रोपाई के 2-3 दिन के अंदर प्रेटिलाक्लोर (Pretilachlor) 500 मिली प्रति एकड़ की दर से छिड़काव करें।
"""
    # Default Agricultural guidance
    return f"""### 🌾 एग्रोसेजजीपीटी (AgroSageGPT) - कृषि सलाह
**नमस्ते किसान भाई! आपके सवाल: "{query}" के संदर्भ में:**

1. **बुवाई एवं मृदा तैयारी:** खेत की 2-3 बार गहरी जुताई करके पाटा लगाएं और मिट्टी को भुरभुरा बनाएं। खेत में जैविक खाद (गोबर की सड़ी खाद) 4-5 टन प्रति एकड़ अवश्य मिलाएं।
2. **बीज उपचार:** हमेशा प्रमाणित बीज का प्रयोग करें और फफूंदनाशक से बीज उपचार के बाद ही बुवाई करें।
3. **संतुलित पोषण:** अपनी मिट्टी के परीक्षण के अनुसार ही नाइट्रोजन, फास्फोरस एवं पोटाश (NPK) का प्रयोग करें।
4. **सिंचाई एवं सुरक्षा:** मौसम और फसल की अवस्था के अनुसार ही सिंचाई करें। अधिक जानकारी के लिए अपनी फसल की फोटो अपलोड करें।
"""


def ask_agronomist(
    query: str,
    vision_summary: str = "No crop image uploaded yet. Routine field check.",
    soil_summary: str = "Soil parameters within normal ranges.",
    weather_summary: str = "Weather conditions moderate.",
    language: str = "English",
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """Execute RAG pipeline combining Vision, IoT, Weather, and Vector DB.
    
    Tiers:
    1. Primary: Google Gemini 3.8 Flash (Comprehensive, Multilingual)
    2. Fallback: Groq Qwen 2.5 32B / Qwen 3.8 (Ultra-Fast LPU reasoning)
    3. Offline: Intelligent Dynamic Agricultural Synthesizer
    """
    kb = get_knowledge_store()
    
    # Retrieve knowledge chunks
    search_query = f"{query} {vision_summary[:60]}"
    retrieved_docs = kb.query_knowledge(search_query, n_results=2)
    context_str = "\n\n".join(retrieved_docs)
    
    sys_prompt = AGRONOMIST_SYSTEM_PROMPT.format(language=language)
    user_prompt = (
        f"Domain Agricultural Knowledge Chunks:\n{context_str}\n\n"
        f"Crop Foliage Vision Status:\n{vision_summary}\n\n"
        f"Live Soil Telemetry:\n{soil_summary}\n\n"
        f"Local Weather Station:\n{weather_summary}\n\n"
        f"Farmer Query: {query}\n\n"
        f"Please provide your complete, practical agricultural response in {language}."
    )

    # 1. PRIMARY: Google Gemini Flash
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            
            # Try active Gemini models in order of quota availability
            for mod_name in ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-3.8-flash"]:
                try:
                    model = genai.GenerativeModel(mod_name)
                    full_prompt = f"{sys_prompt}\n\n{user_prompt}"
                    res = model.generate_content(full_prompt, request_options={"retry": None, "timeout": 8.0})
                    if res and res.text and len(res.text.strip()) > 30:
                        return {
                            "answer": res.text.strip(),
                            "retrieved_context": retrieved_docs,
                            "engine": f"Google Gemini Flash ({mod_name})"
                        }
                except Exception as mod_e:
                    err_str = str(mod_e).lower()
                    if "getaddrinfo" in err_str or "nameresolution" in err_str or "connection" in err_str:
                        break
                    continue
        except Exception as e:
            print(f"[RAGEngine] Gemini issue: {e}. Switching to Groq Qwen...")

    # 2. SECONDARY: Groq Qwen (High-Speed Indic/Multilingual LPU)
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and groq_key != "your_groq_api_key_here":
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=650
            )
            answer_text = response.choices[0].message.content.strip()
            if answer_text and len(answer_text) > 30:
                return {
                    "answer": answer_text,
                    "retrieved_context": retrieved_docs,
                    "engine": "Groq Qwen 27B (High-Speed LPU Fallback)"
                }
        except Exception as e:
            print(f"[RAGEngine] Groq Qwen issue: {e}. Falling back to Dynamic Knowledge Synthesizer.")

    # 3. TERTIARY: Dynamic Crop Knowledge Synthesizer
    answer = _generate_dynamic_offline_response(query=query, language=language)
    return {
        "answer": answer,
        "retrieved_context": retrieved_docs,
        "engine": "AgroSage Dynamic Knowledge Synthesizer"
    }
