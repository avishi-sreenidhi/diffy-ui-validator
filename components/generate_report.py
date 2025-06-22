import json

def generate_json_report(report_dict):
    """
    Converts the final report dictionary into a downloadable JSON string.
    """
    try:
        json_string = json.dumps(report_dict, indent=4, ensure_ascii=False)
        return json_string.encode("utf-8")  # Return bytes for download
    except Exception as e:
        raise RuntimeError(f"Failed to serialize report: {e}")
