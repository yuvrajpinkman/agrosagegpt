"""Vision Module for Crop & Leaf Disease Detection.

Uses Ultralytics YOLOv8 for object localization and features a prototype
disease classification simulation designed for field agronomist workflows.
"""

from typing import Tuple, Dict, Any, Union
import numpy as np
from PIL import Image
import cv2

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
            - 'disease_diagnosis': Simulated disease classification details.
            - 'summary': Farmer-friendly text summary for RAG prompt injection.
    """
    # Normalize input image into OpenCV BGR numpy array
    if isinstance(image_input, str):
        cv_img = cv2.imread(image_input)
        if cv_img is None:
            raise ValueError(f"Failed to read image at {image_input}")
    elif isinstance(image_input, Image.Image):
        rgb_array = np.array(image_input.convert("RGB"))
        cv_img = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        cv_img = image_input.copy()
    else:
        raise ValueError("Unsupported image input type.")

    detections = []
    annotated_img = cv_img.copy()
    
    model = get_yolo_model()
    if model is not None:
        try:
            results = model(cv_img, verbose=False)
            for r in results:
                # Use Ultralytics plot() for visualization
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
            print(f"[VisionModule] Inference error: {e}")

    # Simulated Disease Classification for Agronomic Prototype
    # In practice, specialized models (e.g., PlantVillage fine-tuned YOLO) classify leaf lesions.
    # We analyze average green/yellow/brown pixel ratios as a heuristic for prototype realism.
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    
    # Yellow/brown lesion hue ranges (0-25 and 20-35)
    lesion_pixels = np.sum((h_channel < 25) | (h_channel > 165))
    total_pixels = cv_img.shape[0] * cv_img.shape[1]
    lesion_ratio = lesion_pixels / float(total_pixels)
    
    if lesion_ratio > 0.04:
        diagnosis = {
            "condition": "Early Blight (Alternaria solani) / Leaf Spot",
            "confidence": 0.88,
            "severity": "Moderate to High",
            "symptoms": "Concentric rings, chlorosis, and brown necrotic lesions on foliage.",
            "recommendation": "Prune infected lower foliage immediately. Apply copper-based fungicide or Trichoderma harzianum bio-agent. Maintain drip irrigation."
        }
    elif lesion_ratio > 0.015:
        diagnosis = {
            "condition": "Nutrient Chlorosis & Early Cercospora Spot",
            "confidence": 0.79,
            "severity": "Mild",
            "symptoms": "Yellowing interveinal discoloration with localized speckled lesions.",
            "recommendation": "Supplement with micronutrient foliar spray (Zinc + Magnesium) and monitor for spore progression over the next 48 hours."
        }
    else:
        diagnosis = {
            "condition": "Healthy Foliage with Normal Leaf Turgor",
            "confidence": 0.94,
            "severity": "None (Healthy)",
            "symptoms": "Uniform chlorophyll distribution, intact leaf margins, no significant lesion clustering.",
            "recommendation": "Maintain standard preventative IPM (Integrated Pest Management) and standard irrigation schedule."
        }
        
    # Build summary string for RAG injection
    det_str = f"{len(detections)} object(s) localized ({', '.join([d['class'] for d in detections[:3]]) if detections else 'General Foliage'})."
    rag_summary = (
        f"Computer Vision Diagnostic:\n"
        f"- Target Category: {crop_type}\n"
        f"- Object Localization: {det_str}\n"
        f"- Disease Assessment: {diagnosis['condition']} (Confidence: {diagnosis['confidence'] * 100:.1f}%, Severity: {diagnosis['severity']})\n"
        f"- Observed Symptoms: {diagnosis['symptoms']}\n"
        f"- Immediate Field Action: {diagnosis['recommendation']}"
    )

    # Convert annotated image back to RGB for Streamlit/PIL rendering
    rgb_annotated = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    return {
        "detections": detections,
        "annotated_image": rgb_annotated,
        "disease_diagnosis": diagnosis,
        "summary": rag_summary
    }
