import math
import numpy as np
from fastapi.responses import JSONResponse

def sanitize_for_json(data):
    """Sanitize data to ensure all values are JSON compliant."""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, np.ndarray):
        return sanitize_for_json(data.tolist())
    elif isinstance(data, (float, np.float64, np.float32)):
        # Replace invalid values with None
        if math.isnan(data) or math.isinf(data):
            return None
        return float(data)  # Ensure native Python float
    else:
        return data

class CustomJSONResponse(JSONResponse):
    """Custom JSONResponse that handles NumPy types and NaN/Inf values."""
    def render(self, content) -> bytes:
        sanitized_content = sanitize_for_json(content)
        return super().render(sanitized_content)