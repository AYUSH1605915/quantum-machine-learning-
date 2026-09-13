import re
import os
import io
import tempfile
import subprocess
from PIL import Image

class MedicalReportParser:
    """
    Intelligent Medical Text and Lab Report Parser.
    Extracts numerical biomarkers from unformatted clinical text, doctor notes,
    or OCR recognition results into structured clinical features.
    """

    # Comprehensive regular expression dictionary for multi-alias matching
    BIOMARKER_PATTERNS = {
        "Age": [
            r"(?:age|patient\s*age|years?\s*old)[^\d\n\r]*?([0-9]{1,3})",
            r"\b([0-9]{1,3})\s*(?:years|yrs|yo|y\/o)\b"
        ],
        "Gender": [
            r"(?:gender|sex)[^\w\n\r]*?(male|female|m|f|1|0)\b",
            r"\b(male|female)\b"
        ],
        "BMI": [
            r"(?:bmi|body\s*mass\s*index)[^\d\n\r]*?([0-9]{1,2}(?:\.[0-9]{1,2})?)",
        ],
        "Total_Bilirubin": [
            r"(?:total\s*bilirubin|t\.?\s*bili(?:rubin)?|bilirubin\s*total|t-bili)[^\d\n\r]*?([0-9]{1,2}(?:\.[0-9]{1,2})?)",
            r"(?:bilirubin)[^\d\n\r]*?([0-9]{1,2}(?:\.[0-9]{1,2})?)"
        ],
        "Direct_Bilirubin": [
            r"(?:direct\s*bilirubin|conjugated\s*bilirubin|d\.?\s*bili(?:rubin)?|d-bili)[^\d\n\r]*?([0-9]{1,2}(?:\.[0-9]{1,2})?)"
        ],
        "Alkaline_Phosphatase": [
            r"(?:alkaline\s*phosphatase|alk\s*phos|alp)[^\d\n\r]*?([0-9]{2,4}(?:\.[0-9]{1,2})?)"
        ],
        "Alamine_Aminotransferase": [
            r"(?:alamine\s*aminotransferase|alanine\s*aminotransferase|alt\s*[\/\-]?\s*sgpt|sgpt\s*[\/\-]?\s*alt|\balt\b|\bsgpt\b)[^\d\n\r]*?([0-9]{1,4}(?:\.[0-9]{1,2})?)"
        ],
        "Aspartate_Aminotransferase": [
            r"(?:aspartate\s*aminotransferase|ast\s*[\/\-]?\s*sgot|sgot\s*[\/\-]?\s*ast|\bast\b|\bsgot\b)[^\d\n\r]*?([0-9]{1,4}(?:\.[0-9]{1,2})?)"
        ],
        "Total_Proteins": [
            r"(?:total\s*protein[s]?|t\.?\s*protein[s]?|serum\s*protein)[^\d\n\r]*?([0-9]{1,2}(?:\.[0-9]{1,2})?)"
        ],
        "Albumin": [
            r"(?:serum\s*albumin|\balbumin\b)[^\d\n\r]*?([0-9]{1,2}(?:\.[0-9]{1,2})?)"
        ],
        "Albumin_and_Globulin_Ratio": [
            r"(?:albumin[\s\/\-]and[\s\/\-]?globulin\s*ratio|a[\s\/\:]g\s*ratio|a\/g)[^\d\n\r]*?([0-9]{0,2}(?:\.[0-9]{1,2})?)"
        ],
        "Fasting_Glucose": [
            r"(?:fasting\s*glucose|fbs|fasting\s*blood\s*sugar|blood\s*sugar|\bglucose\b)[^\d\n\r]*?([0-9]{2,4}(?:\.[0-9]{1,2})?)"
        ],
        "Serum_Cholesterol": [
            r"(?:total\s*cholesterol|serum\s*cholesterol|\bcholesterol\b)[^\d\n\r]*?([0-9]{2,4}(?:\.[0-9]{1,2})?)"
        ],
        "Platelet_Count": [
            r"(?:platelet[s]?\s*count|\bplatelet[s]?\b)[^\d\n\r]*?([0-9]{2,6}(?:\.[0-9]{1,2})?)"
        ]
    }

    # Reference clinical defaults (median dataset values)
    DEFAULT_MEDIANS = {
        "Age": 45.0,
        "Gender": 1.0,
        "BMI": 26.5,
        "Total_Bilirubin": 1.0,
        "Direct_Bilirubin": 0.3,
        "Alkaline_Phosphatase": 205.0,
        "Alamine_Aminotransferase": 35.0,
        "Aspartate_Aminotransferase": 38.0,
        "Total_Proteins": 6.7,
        "Albumin": 3.3,
        "Albumin_and_Globulin_Ratio": 0.95,
        "Fasting_Glucose": 110.0,
        "Serum_Cholesterol": 195.0,
        "Platelet_Count": 235.0
    }

    @classmethod
    def parse_text(cls, text: str) -> dict:
        """
        Parses unformatted clinical text and extracts all detected biomarkers.
        Returns a dictionary with extracted features, confidence, and detection status.
        """
        if not text:
            return {"extracted": {}, "missing": list(cls.DEFAULT_MEDIANS.keys()), "status": "empty"}

        extracted = {}
        text_lower = text.lower()

        # Check for Height and Weight to auto-calculate BMI if explicit BMI isn't found
        height_m = None
        weight_kg = None

        h_match = re.search(r"(?:height|ht)[\s:]*([0-9]{2,3}(?:\.[0-9]+)?)\s*(cm|m)?", text_lower)
        if h_match:
            val = float(h_match.group(1))
            unit = h_match.group(2)
            height_m = (val / 100.0) if (val > 3.0 or unit == "cm") else val

        w_match = re.search(r"(?:weight|wt)[\s:]*([0-9]{2,3}(?:\.[0-9]+)?)\s*(kg|lbs)?", text_lower)
        if w_match:
            val = float(w_match.group(1))
            unit = w_match.group(2)
            weight_kg = (val * 0.453592) if unit == "lbs" else val

        if height_m and weight_kg and height_m > 0.5:
            calculated_bmi = round(weight_kg / (height_m ** 2), 1)
            extracted["BMI"] = calculated_bmi

        # Run regex for all 14 biomarkers
        for biomarker, patterns in cls.BIOMARKER_PATTERNS.items():
            if biomarker == "BMI" and "BMI" in extracted:
                continue

            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    raw_val = match.group(1)
                    if biomarker == "Gender":
                        val = 1.0 if raw_val in ["male", "m", "1"] else 0.0
                        extracted["Gender"] = val
                    else:
                        try:
                            val = float(raw_val)
                            # Handle Platelets expressed in thousands or full number (e.g. 240000 -> 240)
                            if biomarker == "Platelet_Count" and val > 1000:
                                val = round(val / 1000.0, 1)
                            extracted[biomarker] = val
                        except ValueError:
                            pass
                    break

        # Derived: if A/G ratio is missing but Total Proteins and Albumin are available
        if "Albumin_and_Globulin_Ratio" not in extracted:
            if "Total_Proteins" in extracted and "Albumin" in extracted:
                globulin = extracted["Total_Proteins"] - extracted["Albumin"]
                if globulin > 0.1:
                    extracted["Albumin_and_Globulin_Ratio"] = round(extracted["Albumin"] / globulin, 2)

        # Separate identified vs missing features
        found_keys = set(extracted.keys())
        all_keys = set(cls.DEFAULT_MEDIANS.keys())
        missing_keys = list(all_keys - found_keys)

        # Create complete filled record using dataset medians for any missing
        complete_vitals = {}
        for k, median_val in cls.DEFAULT_MEDIANS.items():
            complete_vitals[k] = extracted.get(k, median_val)

        return {
            "extracted": extracted,
            "complete_vitals": complete_vitals,
            "detected_count": len(extracted),
            "total_count": len(cls.DEFAULT_MEDIANS),
            "missing_keys": missing_keys,
            "status": "success" if extracted else "no_features_found"
        }

    @classmethod
    def parse_image_bytes(cls, image_bytes: bytes) -> dict:
        """
        Performs OCR on the provided image bytes using Windows Native OCR or local fallback,
        then extracts biomarkers from the recognized text.
        """
        # Save temp image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Normalize with PIL
            img = Image.open(io.BytesIO(image_bytes))
            # Convert to grayscale and enhance size if small
            img = img.convert("L")
            if img.width < 1000:
                scale = 1000 / img.width
                img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
            img.save(tmp_path, format="PNG")

            # Execute Windows Native OCR via PowerShell
            ps_script = f"""
            [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
            [Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
            [Windows.Storage.StorageFile, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null

            $asyncOp = [Windows.Storage.StorageFile]::GetFileFromPathAsync('{tmp_path.replace(chr(92), '/')}')
            $file = $asyncOp.GetAwaiter().GetResult()
            $streamOp = $file.OpenAsync([Windows.Storage.FileAccessMode]::Read)
            $stream = $streamOp.GetAwaiter().GetResult()
            $decoderOp = [Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)
            $decoder = $decoderOp.GetAwaiter().GetResult()
            $bitmapOp = $decoder.GetSoftwareBitmapAsync()
            $bitmap = $bitmapOp.GetAwaiter().GetResult()

            $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new('en-US'))
            if (-not $engine) {{
                $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
            }}
            $ocrOp = $engine.RecognizeAsync($bitmap)
            $result = $ocrOp.GetAwaiter().GetResult()
            $result.Text
            """
            proc = subprocess.run(["powershell", "-NoProfile", "-Command", ps_script],
                                  capture_output=True, text=True, timeout=15)
            ocr_text = proc.stdout.strip()
            
            parsed = cls.parse_text(ocr_text)
            parsed["raw_ocr_text"] = ocr_text
            return parsed

        except Exception as e:
            return {
                "extracted": {},
                "complete_vitals": cls.DEFAULT_MEDIANS.copy(),
                "detected_count": 0,
                "total_count": len(cls.DEFAULT_MEDIANS),
                "missing_keys": list(cls.DEFAULT_MEDIANS.keys()),
                "raw_ocr_text": "",
                "status": f"OCR extraction error: {str(e)}"
            }
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
