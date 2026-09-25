"""Verification and Unit Testing Suite for AgroSageGPT.

Tests:
1. Soil Mock Module (synthetic IoT generation and summary format)
2. Weather API Module (fallback / live API handling)
3. Vision Module (Ultralytics YOLO inference & disease heuristic)
4. RAG Engine (ChromaDB knowledge retrieval & prompt synthesis)
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

def test_soil_module():
    print("[1/4] Testing Soil Mock Module...")
    from src.soil_mock import generate_soil_history, get_latest_soil_reading
    
    df = generate_soil_history(hours=12)
    assert not df.empty, "Soil history DataFrame is empty"
    assert "nitrogen_mg_kg" in df.columns, "Missing nitrogen column"
    assert "ph" in df.columns, "Missing pH column"
    assert "moisture_pct" in df.columns, "Missing moisture column"
    
    summary_str = get_latest_soil_reading(df, as_dict=False)
    assert isinstance(summary_str, str) and len(summary_str) > 20, "Soil summary string invalid"
    summary_dict = get_latest_soil_reading(df, as_dict=True)
    assert isinstance(summary_dict, dict) and "nitrogen" in summary_dict, "Soil dict invalid"
    print("  -> Soil Module passed successfully.")


def test_weather_module():
    print("[2/4] Testing Weather API Module...")
    from src.weather_api import get_weather_data, format_weather_summary
    
    w_data = get_weather_data("Bengaluru")
    assert isinstance(w_data, dict), "Weather data is not a dict"
    assert "temperature_c" in w_data, "Missing temperature"
    assert "humidity_pct" in w_data, "Missing humidity"
    
    summary = format_weather_summary(w_data)
    assert "Live Weather at" in summary or "Weather at" in summary, "Weather summary malformed"
    print(f"  -> Weather Module passed successfully (Source: {w_data['source']}).")


def test_vision_module():
    print("[3/4] Testing Vision Module with sample image...")
    from src.vision_module import analyze_crop_image
    
    img_path = "sample_data/sample_diseased_leaf.jpg"
    assert os.path.exists(img_path), f"Sample image {img_path} not found"
    
    res = analyze_crop_image(img_path, crop_type="Tomato")
    assert "detections" in res, "Missing detections in result"
    assert "annotated_image" in res, "Missing annotated image"
    assert "disease_diagnosis" in res, "Missing disease diagnosis"
    assert "summary" in res, "Missing summary string"
    assert "Early Blight" in res["disease_diagnosis"]["condition"], "Disease classification mismatch"
    print(f"  -> Vision Module passed successfully (Diagnosed: {res['disease_diagnosis']['condition']}).")


def test_rag_engine():
    print("[4/4] Testing RAG Engine & ChromaDB Vector Store...")
    from src.rag_engine import ask_agronomist, get_knowledge_store
    
    # 1. Test vector store retrieval
    kb = get_knowledge_store()
    retrieved = kb.query_knowledge("What is the remedy for leaf blight?", n_results=2)
    assert len(retrieved) > 0, "No documents retrieved from ChromaDB"
    
    # 2. Test Multilingual English advisory
    en_res = ask_agronomist(
        query="Should I spray fungicide today?",
        vision_summary="Early Blight detected on 40% leaves.",
        soil_summary="Moisture: 42%, N: 45 mg/kg, pH: 6.4",
        weather_summary="Temp: 28C, Humidity: 65%, Wind: 3 m/s",
        language="English"
    )
    assert "answer" in en_res and len(en_res["answer"]) > 100, "English response too short"
    assert len(en_res["retrieved_context"]) > 0, "Missing retrieved context in answer"
    
    # 3. Test Hindi advisory
    hi_res = ask_agronomist(
        query="मेरी फसल में रोग लगा है क्या करूं?",
        vision_summary="Early Blight detected on leaves.",
        soil_summary="Moisture: 42%",
        weather_summary="Temp: 28C",
        language="Hindi"
    )
    assert "answer" in hi_res and "फसल" in hi_res["answer"], "Hindi response generation failed"
    print("  -> RAG Engine & ChromaDB passed successfully.")


if __name__ == "__main__":
    print("=== Running AgroSageGPT System Verification ===")
    test_soil_module()
    test_weather_module()
    test_vision_module()
    test_rag_engine()
    print("=== ALL 4 CORE MODULE TESTS PASSED! ===")
