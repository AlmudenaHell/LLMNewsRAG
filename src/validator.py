def validate_output(output, text):
    if "error" in output:
        return False

    company = output.get("company", "").lower()

    # Simple validation: company must appear in text
    if company and company in text.lower():
        return True

    return False

def confidence_score(output, text):
    if "error" in output:
        return 0.0

    score = 0.5

    if output.get("company", "").lower() in text.lower():
        score += 0.25

    if output.get("event"):
        score += 0.25

    return min(score, 1.0)
