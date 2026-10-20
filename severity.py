
def calculate_severity(boxes, image_width, image_height):
    image_area = image_width * image_height

    if image_area <= 0:
        return 0.0, "NONE"

    total_area = 0.0

    for x1, y1, x2, y2 in boxes:
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        total_area += width * height

    coverage = (total_area / image_area) * 100

    if not boxes:
        severity = "NONE"
    elif coverage < 3:
        severity = "LOW"
    elif coverage < 10:
        severity = "MEDIUM"
    elif coverage < 20:
        severity = "HIGH"
    else:
        severity = "CRITICAL"

    return round(coverage, 2), severity
