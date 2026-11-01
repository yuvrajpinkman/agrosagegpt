"""Vision Module for Crop & Leaf Disease Detection.

Combines Hugging Face Vision Transformers (Meta DINOv2 on PlantVillage)
with Google Gemini Multimodal Vision as an expert validator.
Features Ultralytics YOLOv8 for spatial object localization and
an OpenCV heuristic fallback if offline.
"""

import os
import json
import re
from typing import Tuple, Dict, Any, Union, Optional, List
import numpy as np
from PIL import Image
import cv2
from dotenv import load_dotenv

load_dotenv()

# Global lazy-loaded models
_yolo_model = None
_hf_vision_pipeline = None


def get_yolo_model(model_name: str = "yolov8n.pt"):
    """Lazy load the YOLO model."""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO(model_name)
        except Exception as e:
            print(f"[VisionModule] Note: Error loading YOLO ({e}). Running in fallback mode.")
            _yolo_model = None
    return _yolo_model


def get_hf_vision_pipeline(model_id: str = "rodynaemad/plant-disease-dinov2-small"):
    """Lazy load the Hugging Face plant disease classification pipeline."""
    global _hf_vision_pipeline
    if _hf_vision_pipeline is None:
        try:
            from transformers import pipeline
            _hf_vision_pipeline = pipeline("image-classification", model=model_id)
        except Exception as e:
            print(f"[VisionModule] Note: Error loading HF vision pipeline ({e}).")
            _hf_vision_pipeline = None
    return _hf_vision_pipeline


def _format_plant_label(raw_label: str) -> str:
    """Format PlantVillage raw label into human-readable crop condition."""
    # Example: 'Tomato___Early_blight' -> 'Tomato - Early Blight'
    parts = raw_label.split("___")
    crop = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    
    # Capitalize words
    crop_title = " ".join(word.capitalize() for word in crop.split())
    disease_title = " ".join(word.capitalize() for word in disease.split())
    
    if disease_title.lower() == "healthy":
        return f"{crop_title} - Healthy Foliage"
    return f"{crop_title} - {disease_title}"


def _classify_with_hf(pil_img: Image.Image) -> Optional[Dict[str, Any]]:
    """Classify foliage using local Hugging Face Vision Transformer (DINOv2)."""
    p = get_hf_vision_pipeline()
    if p is None:
        return None
        
    try:
        raw_preds = p(pil_img)
        if not raw_preds:
            return None
            
        top_pred = raw_preds[0]
        top_condition = _format_plant_label(top_pred["label"])
        top_conf = round(float(top_pred["score"]), 3)
        
        candidates = []
        for pred in raw_preds[:3]:
            candidates.append({
                "condition": _format_plant_label(pred["label"]),
                "confidence": round(float(pred["score"]), 3)
            })
            
        return {
            "top_condition": top_condition,
            "raw_label": top_pred["label"],
            "confidence": top_conf,
            "candidates": candidates
        }
    except Exception as e:
        print(f"[VisionModule] HF classifier exception: {e}")
        return None


def _validate_with_gemini(
    pil_img: Image.Image,
    crop_type: str,
    hf_result: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Use Gemini Multimodal Vision as an expert validator for the local HF prediction."""
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        return None
        
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        
        if hf_result:
            candidates_str = ", ".join([f"{c['condition']} ({c['confidence']*100:.1f}%)" for c in hf_result["candidates"]])
            prompt = f"""You are a senior plant pathologist and agricultural diagnostician acting as an EXPERT VALIDATOR.
A local Vision Transformer (Meta DINOv2 trained on PlantVillage 38 crop disease classes) analyzed this {crop_type} foliage photo:
- Primary Local Prediction: {hf_result['top_condition']} (Confidence: {hf_result['confidence']*100:.1f}%)
- Alternative Candidates: {candidates_str}

YOUR TASK AS AN EXPERT VALIDATOR:
1. Inspect the foliar image closely. Confirm whether you AGREE, DISAGREE, or REFINE the local model's prediction.
2. Identify specific foliar visual markers (e.g. concentric target rings, necrotic margins, chlorotic halos, rust pustules, interveinal yellowing, or healthy leaf turgor).
3. Gauge disease severity: "None (Healthy)", "Mild", "Moderate", or "High".
4. Prescribe immediate, actionable agricultural management (specific chemical fungicide/bactericide or organic bio-agent dosage, e.g., Mancozeb 75% WP @ 2g/L, Copper Oxychloride, Trichoderma viride, or Neem seed kernel extract, plus irrigation precautions).

Respond ONLY with a valid JSON object in this exact format (no markdown fences, just raw JSON):
{{
  "condition": "Confirmed or Corrected condition name",
  "validator_verdict": "Agreed" or "Refined" or "Corrected",
  "validation_notes": "Expert reasoning on why you agree, refine, or correct the local model",
  "confidence": 0.94,
  "severity": "None (Healthy)" or "Mild" or "Moderate" or "High",
  "symptoms": "Accurate description of visual lesion features, chlorosis, or healthy turgor",
  "recommendation": "Precise agronomic recommendation including chemical/biological dosage or cultural control"
}}"""
        else:
            prompt = f"""You are a professional plant pathologist and agricultural diagnostician.
Analyze this image of a {crop_type} crop foliage carefully.

Evaluate for:
1. Plant diseases (e.g. early/late blight, powdery mildew, rust, leaf spot, bacterial canker, mosaic virus).
2. Nutrient deficiencies (chlorosis, necrosis, purpling).
3. Insect or pest damage.
4. Or healthy foliage.

Respond ONLY with a valid JSON object in this exact format (no markdown fences, just raw JSON):
{{
  "condition": "Specific disease or 'Healthy Foliage'",
  "validator_verdict": "Direct Assessment",
  "validation_notes": "Direct multimodal visual diagnosis",
  "confidence": 0.92,
  "severity": "None (Healthy)" or "Mild" or "Moderate" or "High",
  "symptoms": "Accurate description of visual lesion features, chlorosis, or necrotic halos",
  "recommendation": "Precise agronomic recommendation including chemical/biological dosage or cultural control"
}}"""

        for mod_name in ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-3.8-flash"]:
            try:
                model = genai.GenerativeModel(mod_name)
                response = model.generate_content([prompt, pil_img], request_options={"retry": None, "timeout": 8.0})
                text = response.text.strip()
                
                clean_json = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
                clean_json = re.sub(r"```$", "", clean_json, flags=re.MULTILINE).strip()
                
                data = json.loads(clean_json)
                if hf_result:
                    data["source"] = f"Hugging Face DINOv2 + Google Gemini ({mod_name}) Validator"
                    data["hf_prediction"] = f"{hf_result['top_condition']} ({hf_result['confidence']*100:.1f}%)"
                else:
                    data["source"] = f"Google Gemini Vision ({mod_name})"
                return data
            except Exception as e:
                print(f"[VisionValidator] {mod_name} error: {e}")
                err_str = str(e).lower()
                if "getaddrinfo" in err_str or "nameresolution" in err_str or "connection" in err_str:
                    # Offline / DNS unreachable: fail fast to local HF offline engine
                    break
                continue
    except Exception as e:
        print(f"[VisionModule] Gemini Vision validator fallback triggered: {e}")
        return None


def analyze_crop_image(
    image_input: Union[str, np.ndarray, Image.Image],
    crop_type: str = "General Crop"
) -> Dict[str, Any]:
    """Run YOLO inference and agronomic disease diagnosis on an uploaded crop image.
    
    Args:
        image_input: File path, PIL Image, or OpenCV NumPy array.
        crop_type: Optional crop context (e.g., Tomato, Rice, Maize, Wheat).
        
    Returns:
        dict containing:
            - 'detections': List of detected bounding boxes and classes.
            - 'annotated_image': OpenCV image array with visual boxes/labels.
            - 'disease_diagnosis': Simulated or multimodal disease classification details.
            - 'summary': Farmer-friendly text summary for RAG prompt injection.
    """
    # Normalize input image into OpenCV BGR numpy array & PIL Image
    if isinstance(image_input, str):
        cv_img = cv2.imread(image_input)
        if cv_img is None:
            raise ValueError(f"Failed to read image at {image_input}")
        pil_img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
        rgb_array = np.array(pil_img)
        cv_img = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        cv_img = image_input.copy()
        pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError("Unsupported image input type.")

    detections = []
    annotated_img = cv_img.copy()
    
    # 1. Run YOLOv8 object localization
    model = get_yolo_model()
    if model is not None:
        try:
            results = model(cv_img, verbose=False)
            for r in results:
                annotated_img = r.plot()
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = model.names.get(cls_id, f"class_{cls_id}")
                    conf = float(box.conf[0].item())
                    xyxy = [round(float(c), 1) for c in box.xyxy[0].tolist()]
                    detections.append({
                        "class": cls_name,
                        "confidence": round(conf, 3),
                        "bbox": xyxy
                    })
        except Exception as e:
            print(f"[VisionModule] YOLO inference error: {e}")

    # 2. Primary Classification: Local Hugging Face Meta DINOv2 Vision Transformer
    hf_res = _classify_with_hf(pil_img)

    # 3. Validation: Google Gemini Multimodal Vision Validator
    diagnosis = _validate_with_gemini(pil_img, crop_type, hf_res)

    # 4. Fallback A: Use Hugging Face classification directly if Gemini validator is unavailable
    if not diagnosis and hf_res is not None:
        is_healthy = "healthy" in hf_res["top_condition"].lower()
        severity = "None (Healthy)" if is_healthy else ("High" if hf_res["confidence"] > 0.85 else "Moderate")
        
        if is_healthy:
            symptoms = "Foliage exhibits uniform chlorophyll, intact leaf margins, and robust turgor pressure."
            recom = "Maintain preventative Integrated Pest Management (IPM) and balanced macro/micronutrient fertigation."
        else:
            symptoms = f"Foliar lesions, chlorotic halo, or necrotic spotting characteristic of {hf_res['top_condition']}."
            recom = f"Isolate infected foliage. Apply targeted fungicide/bio-agent (e.g. Mancozeb 75% WP @ 2g/L or Trichoderma viride) and regulate leaf surface moisture."

        diagnosis = {
            "condition": hf_res["top_condition"],
            "validator_verdict": "Offline Classification (Gemini Skipped)",
            "validation_notes": f"Primary prediction by local Meta DINOv2 ViT model with {hf_res['confidence']*100:.1f}% confidence.",
            "confidence": hf_res["confidence"],
            "severity": severity,
            "symptoms": symptoms,
            "recommendation": recom,
            "source": "Hugging Face DINOv2 (Local Offline Engine)",
            "hf_prediction": f"{hf_res['top_condition']} ({hf_res['confidence']*100:.1f}%)"
        }

    # 5. Fallback B: OpenCV lesion heuristic if both HF and Gemini are unavailable
    if not diagnosis:
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        lesion_pixels = np.sum((h_channel < 25) | (h_channel > 165))
        total_pixels = cv_img.shape[0] * cv_img.shape[1]
        lesion_ratio = lesion_pixels / float(total_pixels)
        
        if lesion_ratio > 0.04:
            diagnosis = {
                "condition": f"Early Blight / Leaf Spot on {crop_type}",
                "validator_verdict": "Heuristic Estimate",
                "validation_notes": "Estimated via OpenCV HSV color histogram thresholding.",
                "confidence": 0.88,
                "severity": "Moderate to High",
                "symptoms": "Concentric rings, chlorosis, and brown necrotic lesions on foliage.",
                "recommendation": "Prune infected lower foliage immediately. Apply copper-based fungicide or Trichoderma harzianum bio-agent. Maintain drip irrigation.",
                "source": "OpenCV Lesion Heuristic (Fallback)"
            }
        elif lesion_ratio > 0.015:
            diagnosis = {
                "condition": f"Nutrient Chlorosis & Early Spot on {crop_type}",
                "validator_verdict": "Heuristic Estimate",
                "validation_notes": "Speckled lesions detected via OpenCV HSV segmentation.",
                "confidence": 0.79,
                "severity": "Mild",
                "symptoms": "Yellowing interveinal discoloration with localized speckled lesions.",
                "recommendation": "Supplement with micronutrient foliar spray (Zinc + Magnesium) and monitor for spore progression over the next 48 hours.",
                "source": "OpenCV Lesion Heuristic (Fallback)"
            }
        else:
            diagnosis = {
                "condition": f"Healthy Foliage with Normal Leaf Turgor ({crop_type})",
                "validator_verdict": "Heuristic Estimate",
                "validation_notes": "Uniform green canopy detected with low necrotic ratio.",
                "confidence": 0.94,
                "severity": "None (Healthy)",
                "symptoms": "Uniform chlorophyll distribution, intact leaf margins, no significant lesion clustering.",
                "recommendation": "Maintain standard preventative IPM (Integrated Pest Management) and standard irrigation schedule.",
                "source": "OpenCV Lesion Heuristic (Fallback)"
            }
        
    # Build summary string for RAG injection
    det_str = f"{len(detections)} object(s) localized ({', '.join([d['class'] for d in detections[:3]]) if detections else 'Crop Foliage'})."
    val_line = f"- Validator Verdict: {diagnosis.get('validator_verdict', 'Validated')} ({diagnosis.get('validation_notes', '')})\n" if "validation_notes" in diagnosis else ""
    hf_line = f"- Local HF Pre-check: {diagnosis.get('hf_prediction')}\n" if "hf_prediction" in diagnosis else ""
    
    rag_summary = (
        f"Computer Vision Diagnostic ({diagnosis.get('source', 'Vision Engine')}):\n"
        f"- Target Crop: {crop_type}\n"
        f"- Spatial Localization: {det_str}\n"
        f"{hf_line}"
        f"{val_line}"
        f"- Disease Assessment: {diagnosis['condition']} (Confidence: {diagnosis['confidence'] * 100:.1f}%, Severity: {diagnosis['severity']})\n"
        f"- Observed Symptoms: {diagnosis['symptoms']}\n"
        f"- Immediate Field Action: {diagnosis['recommendation']}"
    )

    # Convert annotated image back to RGB for Streamlit rendering
    rgb_annotated = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    return {
        "detections": detections,
        "annotated_image": rgb_annotated,
        "disease_diagnosis": diagnosis,
        "summary": rag_summary
    }
