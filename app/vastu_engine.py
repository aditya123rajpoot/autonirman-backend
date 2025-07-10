def vastu_compliance_score(layout: dict) -> dict:
    score = 0
    suggestions = []

    if layout.get("kitchen") == "North-East":
        score += 20
    else:
        suggestions.append("Place kitchen in North-East")

    if layout.get("entrance") == "East":
        score += 20
    else:
        suggestions.append("Place entrance in East")

    if layout.get("master_bedroom") == "South-West":
        score += 20
    else:
        suggestions.append("Master bedroom in South-West")

    if layout.get("toilet") == "South":
        score += 20
    else:
        suggestions.append("Toilet should be in South")

    if layout.get("pooja_room") == "North-East":
        score += 20
    else:
        suggestions.append("Pooja room in North-East")

    return {"score": score, "suggestions": suggestions}
