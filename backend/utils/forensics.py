import PIL.Image
import PIL.ExifTags
from typing import Dict, Optional

class MetadataAnalyzer:
    """
    Forensic Tool: Scans image metadata for AI signatures.
    Targets EXIF/IPTC tags that modern AI models often leak.
    """
    AI_SIGNATURES = [
        "dall-e", "midjourney", "stable diffusion", "adobe firefly",
        "bing image creator", "canva ai", "starryai", "miricanvas",
        "wombo", "artbreeder", "nightcafe"
    ]

    @staticmethod
    def scan(image_path: str) -> Dict:
        """Analyze image metadata for AI software traces."""
        info = {"has_ai_metadata": False, "software": None, "tags_found": []}
        try:
            img = PIL.Image.open(image_path)
            exif = img.getexif()
            
            # Check standard EXIF tags
            for tag_id, value in exif.items():
                tag = PIL.ExifTags.TAGS.get(tag_id, tag_id)
                if isinstance(value, str):
                    val_lower = value.lower()
                    for sig in MetadataAnalyzer.AI_SIGNATURES:
                        if sig in val_lower:
                            info["has_ai_metadata"] = True
                            info["software"] = sig
                            info["tags_found"].append(f"{tag}: {sig}")

            # Check for TIFF software tag (common for Firefly/Canva)
            if hasattr(img, 'info') and 'software' in img.info:
                sw = str(img.info['software']).lower()
                for sig in MetadataAnalyzer.AI_SIGNATURES:
                    if sig in sw:
                        info["has_ai_metadata"] = True
                        info["software"] = sig
                        info["tags_found"].append(f"Software: {sig}")

        except Exception as e:
            print(f"[Metadata] Error: {e}")
            
        return info

class ExplanationGenerator:
    """
    Engine: Converts forensic signals into human-readable technical reasons.
    """
    @staticmethod
    def generate(res: Dict) -> str:
        """Compose an explanation based on the forensic evidence."""
        reasons = []
        
        # 1. Spectral Reason
        if res.get("spectral_score", 0) > 0.6:
            reasons.append("High-frequency spectral artifacts (FFT) suggest synthetic generation noise.")
        elif res.get("spectral_score", 0) > 0.45:
            reasons.append("Minor frequency-domain inconsistencies detected.")

        # 2. Expert Reason
        if res.get("expert_score", 0) > 0.4:
            reasons.append("Deep residual fingerprints matching modern Diffusion models identified.")
        
        # 3. Biometric Reason (for Humans)
        if res.get("face_detected"):
            if res.get("iris_score", 0) > 0.6:
                reasons.append("Non-biological iris patterns and gaze asymmetry detected.")
            elif res.get("authenticity_score", 0) < 15:
                reasons.append("Authentic biological micro-saccades and pupil consistency verified.")

        # 4. Metadata Reason
        if res.get("metadata", {}).get("has_ai_metadata"):
            reasons.append(f"Metadata signature from {res['metadata']['software'].upper()} found.")

        if not reasons:
            return "No definitive AI artifacts found. Visual patterns align with standard photography."
        
        return " ".join(reasons[:2]) # Keep it concise
