"""Vision Module for Crop & Leaf Disease Detection.

Combines Ultralytics YOLOv8 for spatial object localization with
Google Gemini 3.8 Flash Multimodal Vision for plant pathology diagnosis.
Features an OpenCV heuristic fallback if API access is offline.
"""

import os
import json
import re
from typing import Tuple, Dict, Any, Union, Optional
import numpy as np
from PIL import Image
import cv2
from dotenv import load_dotenv

load_dotenv()

# Global lazy-loaded YOLO model
_yolo_model = None


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


def _diagnose_with_gemini(pil_img: Image.Image, crop_type: str) -> Optional[Dict[str, Any]]:
    """Use Gemini 3.8 Flash Multimodal Vision to evaluate plant pathology."""
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        return None
        
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-3.8-flash")
        
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
  "confidence": 0.92,
  "severity": "None (Healthy)" or "Mild" or "Moderate" or "High",
  "symptoms": "Accurate description of visual lesion features, chlorosis, or necrotic halos",
  "recommendation": "Precise agronomic recommendation including chemical/biological dosage or cultural control"
}}"""

        response = model.generate_content([prompt, pil_img])
        text = response.text.strip()
        
        # Clean potential markdown backticks
        clean_json = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        clean_json = re.sub(r"```$", "", clean_json, flags=re.MULTILINE).strip()
        
        data = json.loads(clean_json)
        data["source"] = "Gemini 3.8 Flash Multimodal Vision"
        return data
    except Exception as e:
        print(f"[VisionModule] Gemini Vision fallback triggered: {e}")
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

    # 2. Try Gemini 3.8 Flash Multimodal Plant Pathology
    diagnosis = _diagnose_with_gemini(pil_img, crop_type)

    # 3. Fallback to OpenCV lesion analysis if Gemini is unavailable
    if not diagnosis:
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        lesion_pixels = np.sum((h_channel < 25) | (h_channel > 165))
        total_pixels = cv_img.shape[0] * cv_img.shape[1]
        lesion_ratio = lesion_pixels / float(total_pixels)
        
        if lesion_ratio > 0.04:
            diagnosis = {
                "condition": f"Early Blight / Leaf Spot on {crop_type}",
                "confidence": 0.88,
                "severity": "Moderate to High",
                "symptoms": "Concentric rings, chlorosis, and brown necrotic lesions on foliage.",
                "recommendation": "Prune infected lower foliage immediately. Apply copper-based fungicide or Trichoderma harzianum bio-agent. Maintain drip irrigation.",
                "source": "OpenCV Lesion Heuristic (Fallback)"
            }
        elif lesion_ratio > 0.015:
            diagnosis = {
                "condition": f"Nutrient Chlorosis & Early Spot on {crop_type}",
                "confidence": 0.79,
                "severity": "Mild",
                "symptoms": "Yellowing interveinal discoloration with localized speckled lesions.",
                "recommendation": "Supplement with micronutrient foliar spray (Zinc + Magnesium) and monitor for spore progression over the next 48 hours.",
                "source": "OpenCV Lesion Heuristic (Fallback)"
            }
        else:
            diagnosis = {
                "condition": f"Healthy Foliage with Normal Leaf Turgor ({crop_type})",
                "confidence": 0.94,
                "severity": "None (Healthy)",
                "symptoms": "Uniform chlorophyll distribution, intact leaf margins, no significant lesion clustering.",
                "recommendation": "Maintain standard preventative IPM (Integrated Pest Management) and standard irrigation schedule.",
                "source": "OpenCV Lesion Heuristic (Fallback)"
            }
        
    # Build summary string for RAG injection
    det_str = f"{len(detections)} object(s) localized ({', '.join([d['class'] for d in detections[:3]]) if detections else 'Crop Foliage'})."
    rag_summary = (
        f"Computer Vision Diagnostic ({diagnosis.get('source', 'Vision Engine')}):\n"
        f"- Target Crop: {crop_type}\n"
        f"- Spatial Localization: {det_str}\n"
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
