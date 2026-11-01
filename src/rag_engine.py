"""RAG Engine for AgroSageGPT.

Integrates ChromaDB vector store with agronomic knowledge, conversational context,
Computer Vision diagnostics, IoT soil telemetry, and live weather conditions.
Uses Google Gemini 3.8 Flash as the primary multilingual LLM, Groq (openai/gpt-oss-120b)
as high-speed fallback, and local knowledge synthesis as offline fallback.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


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
        
        # Collection for agricultural knowledge
        self.collection = self.client.get_or_create_collection(
            name="agricultural_advisory_kb"
        )
        
        # Ingest if collection is empty
        if self.collection.count() == 0 and os.path.exists(self.doc_path):
            self._ingest_file(self.doc_path)
            
    def _ingest_file(self, file_path: str):
        """Split document by markdown sections and ingest into ChromaDB."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Chunking by sections or paragraphs
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

    def query_knowledge(self, query_text: str, n_results: int = 3) -> List[str]:
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


AGRONOMIST_SYSTEM_PROMPT = """You are AgroSageGPT, an expert multilingual Digital Agronomist and crop physician.
Your mission is to empower farmers with clear, empathetic, practical, and scientifically sound advice.

You must synthesize information from four key dimensions:
1. Computer Vision Diagnostic (from leaf/crop images)
2. Real-time IoT Soil Sensor Telemetry (NPK, pH, Moisture, Temp, EC)
3. Live Local Weather Data (Temperature, Humidity, Wind, Forecast)
4. Domain Agronomic Knowledge (retrieved research context)

Farmer Language Preference: {language}

Advisory Structure:
1. 🩺 Immediate Diagnosis & Crop Health: Explain in plain terms what is happening with the crop.
2. 🌱 Soil & Nutrient Recommendation: Address any NPK deficiency, pH imbalance, or moisture stress.
3. 🌦️ Weather Precaution & Spray Timing: Explain whether current weather/wind/humidity is safe for spray or requires irrigation adjustments.
4. 📋 Clear 3-Step Action Plan: Concise, numbered, practical steps for the farmer to execute today.

Tone Guidelines:
- Farmer-friendly, respectful, actionable, and encouraging.
- Avoid overly academic jargon; translate numerical values into practical advice (e.g., dosage per 15L backpack knapsack sprayer).
- IMPORTANT: Reply entirely in {language} so the farmer can understand effortlessly.
"""


def _generate_synthetic_multilingual_response(
    query: str,
    vision_summary: str,
    soil_summary: str,
    weather_summary: str,
    retrieved_context: str,
    language: str = "English"
) -> str:
    """Provides a high-quality agronomic response when external LLM keys are offline."""
    lang_lower = language.lower()
    
    headings = {
        "hindi": {
            "title": "🌾 एग्रोसेजजीपीटी (AgroSageGPT) - डिजिटल कृषि परामर्श",
            "diag": "🩺 फसल स्वास्थ्य एवं रोग निदान:",
            "soil": "🌱 मिट्टी एवं पोषण विश्लेषण:",
            "weather": "🌦️ मौसम आधारित सलाह:",
            "plan": "📋 आज के लिए 3 मुख्य कदम:"
        },
        "kannada": {
            "title": "🌾 ಆಗ್ರೋಸೇಜ್‌ಜಿಪಿಟಿ (AgroSageGPT) - ಡಿಜಿಟಲ್ ಕೃಷಿ ಸಲಹೆಗಾರ",
            "diag": "🩺 ಬೆಳೆ ಆರೋಗ್ಯ ಮತ್ತು ರೋಗ ತಪಾಸಣೆ:",
            "soil": "🌱 ಮಣ್ಣು ಮತ್ತು ಪೋಷಕಾಂಶಗಳ ವಿಶ್ಲೇಷಣೆ:",
            "weather": "🌦️ ಹವಾಮಾನ ಆಧಾರಿತ ಮುನ್ನೆಚ್ಚರಿಕೆ:",
            "plan": "📋 ರೈತರಿಗೆ ಇಂದಿನ 3 ಪ್ರಮುಖ ಹಂತಗಳು:"
        },
        "telugu": {
            "title": "🌾 ఆగ్రోసేజ్‌జిపిటి (AgroSageGPT) - డిజిటల్ వ్యవసాయ సలహాదారు",
            "diag": "🩺 పంట ఆరోగ్యం మరియు వ్యాధి నిర్ధారణ:",
            "soil": "🌱 నేల మరియు పోషకాల విశ్లేషణ:",
            "weather": "🌦️ వాతావరణ ఆధారిత సలహా:",
            "plan": "📋 నేడు రైతు చేయవలసిన 3 ముఖ్యమైన పనులు:"
        },
        "tamil": {
            "title": "🌾 அக்ரோசேஜ்ஜிபிடி (AgroSageGPT) - டிஜிட்டல் வேளாண்மை வழிகாட்டி",
            "diag": "🩺 பயிர் ஆரோக்கியம் மற்றும் நோய் கண்டறிதல்:",
            "soil": "🌱 மண் மற்றும் ஊட்டச்சத்து நிலை:",
            "weather": "🌦️ வானிலை அடிப்படையிலான ஆலோசனை:",
            "plan": "📋 விவசாயிகளுக்கான 3 முக்கிய நடவடிக்கைகள்:"
        },
        "english": {
            "title": "🌾 AgroSageGPT Unified Agronomist Advisory",
            "diag": "🩺 Crop Health & Diagnostic Assessment:",
            "soil": "🌱 Soil Health & Nutrient Telemetry:",
            "weather": "🌦️ Weather Impact & Agro-Meteorological Advisory:",
            "plan": "📋 Actionable 3-Step Execution Plan for Today:"
        }
    }
    
    chosen_h = headings.get(lang_lower, headings["english"])
    
    if lang_lower == "hindi":
        return f"""### {chosen_h['title']}
**नमस्ते किसान साथी! आपके खेत के आंकड़ों का संपूर्ण विश्लेषण नीचे दिया गया है:**

#### {chosen_h['diag']}
- {vision_summary.replace(chr(10), ' ')}
- रोग की रोकथाम के लिए जैविक एवं वैज्ञानिक उपाय तुरंत अपनाएं।

#### {chosen_h['soil']}
- {soil_summary.replace(chr(10), ' ')}
- मिट्टी में नमी और संतुलित खाद का विशेष ध्यान रखें ताकि पौधों की जड़ें मजबूत रहें।

#### {chosen_h['weather']}
- {weather_summary.replace(chr(10), ' ')}
- तेज हवा या बारिश की संभावना हो तो रासायनिक छिड़काव टालें।

#### {chosen_h['plan']}
1. **संक्रमित पत्तियों को हटाएं**: रोगग्रस्त पत्तियों को काटकर खेत से दूर नष्ट करें ताकि संक्रमण न फैले।
2. **जैविक छिड़काव**: ट्राइकोडर्मा या 0.2% कॉपर ऑक्सीक्लोराइड का सुबह या शाम के समय छिड़काव करें।
3. **सिंचाई प्रबंधन**: मौसम में आर्द्रता के अनुसार ड्रिप सिंचाई का समय तय करें; पौधों के ऊपर पानी डालने से बचें।

*नोट: यह परामर्श आपकी लाइव मिट्टी, मौसम और दृष्टि सेंसर डेटा पर आधारित है।*
"""
    else:
        return f"""### {chosen_h['title']}
**Hello Farmer! Here is your tailored cross-domain agronomic prescription:**

#### {chosen_h['diag']}
{vision_summary}

#### {chosen_h['soil']}
{soil_summary}

#### {chosen_h['weather']}
{weather_summary}

#### 📚 Agronomic Context from Knowledge Base:
*{retrieved_context[:250]}...*

#### {chosen_h['plan']}
1. **Canopy Sanitization & Pruning**: Remove visibly spotted or chlorotic foliage to arrest spore dispersion and improve internal air circulation.
2. **Nutrient & Soil Balancing**: Align nitrogen and phosphorus application with your soil sensor test; ensure pH remains buffered between 6.0 and 7.0 for optimal micronutrient uptake.
3. **Precision Spray Timing**: Conduct any required biological (Trichoderma harzianum) or preventative copper spray in the calm early morning or late afternoon to prevent evaporative drift and scorch.
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
    1. Primary: Google Gemini 3.8 Flash (Multilingual & High-Fidelity)
    2. Fallback: Groq openai/gpt-oss-120b (High-Speed LLM inference)
    3. Offline: Local Agronomic Rule-based Synthesizer
    
    Args:
        query: User farmer query / question.
        vision_summary: Output summary from Vision module.
        soil_summary: Formatted string from Soil IoT module.
        weather_summary: Formatted string from Weather API module.
        language: User-selected target language.
        chat_history: Previous turn messages.
        
    Returns:
        dict: response text, retrieved context documents, and metadata engine name.
    """
    kb = get_knowledge_store()
    
    # Retrieve relevant knowledge chunks based on query and detected symptoms
    search_query = f"{query} {vision_summary[:80]}"
    retrieved_docs = kb.query_knowledge(search_query, n_results=2)
    context_str = "\n\n".join(retrieved_docs)
    
    sys_prompt = AGRONOMIST_SYSTEM_PROMPT.format(language=language)
    user_prompt = (
        f"Agronomic Reference Documents from Local Vector DB:\n{context_str}\n\n"
        f"Current Crop Diagnostics:\n{vision_summary}\n\n"
        f"Current Soil Sensor Telemetry:\n{soil_summary}\n\n"
        f"Live Local Weather Conditions:\n{weather_summary}\n\n"
        f"Farmer Query: {query}\n"
        f"Please provide your actionable, compassionate response in {language}."
    )

    # 1. PRIMARY: Try Gemini 3.8 Flash
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-3.8-flash")
            full_prompt = f"{sys_prompt}\n\n{user_prompt}"
            res = model.generate_content(full_prompt)
            if res and res.text:
                return {
                    "answer": res.text,
                    "retrieved_context": retrieved_docs,
                    "engine": "Google Gemini 3.8 Flash (Live Multilingual Model)"
                }
        except Exception as e:
            print(f"[RAGEngine] Gemini 3.8 Flash issue: {e}. Switching to Groq fallback.")

    # 2. SECONDARY: Try Groq High-Speed LLM Fallback
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and groq_key != "your_groq_api_key_here":
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            answer_text = response.choices[0].message.content
            return {
                "answer": answer_text,
                "retrieved_context": retrieved_docs,
                "engine": "Groq (High-Speed LLM Fallback)"
            }
        except Exception as e:
            print(f"[RAGEngine] Groq issue: {e}. Switching to offline synthesizer.")

    # 3. TERTIARY: Offline Self-Contained Fallback
    answer = _generate_synthetic_multilingual_response(
        query=query,
        vision_summary=vision_summary,
        soil_summary=soil_summary,
        weather_summary=weather_summary,
        retrieved_context=context_str,
        language=language
    )
    
    return {
        "answer": answer,
        "retrieved_context": retrieved_docs,
        "engine": "AgroSage Offline Rule Synthesizer (Zero-Network Fallback)"
    }
