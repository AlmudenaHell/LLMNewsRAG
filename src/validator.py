def validate_output(output, text):
    if "error" in output:
        return False

    company = output.get("company", "").lower()

    # Simple validation: company must appear in text
    if company and company in text.lower():
        return True

    return False
