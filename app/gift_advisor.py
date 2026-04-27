import re

def extract_price(price_str):
    try:
        return float(price_str.replace("$", "").strip())
    except:
        return None


def gift_recommend(df, query, top_k=5):
    query = query.lower()

    # --- extract age ---
    age_match = re.search(r'(\d+)', query)
    target_age = int(age_match.group(1)) if age_match else None

    # --- extract budget ---
    price_match = re.search(r'\$(\d+)', query)
    budget = float(price_match.group(1)) if price_match else None

    scored = []

    for _, row in df.iterrows():

        score = 0

        text = (row["name"] + " " + row["description"]).lower()

        # interest match
        keywords = query.split()
        for kw in keywords:
            if kw in text:
                score += 2

        # age match
        if target_age and row.get("age"):
            if str(target_age) in str(row["age"]):
                score += 3

        # price filter
        price = extract_price(row["price"])
        if budget and price:
            if price <= budget:
                score += 2
            else:
                continue  # too expensive → drop

        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [r for _, r in scored[:top_k]]