import json

def generate_category_response(category):
    return f"This product belongs to the \"{category}\" category."

def generate_matching_response(display_name, description):
    return (
        f"Based on the visual and textual details, this product closely matches \"{display_name}\". "
        f"Key features include: {description[:150].strip()}..."
    )

def generate_original_response(display_name, description, category):
    return (
        f"The product in the image appears to be a \"{display_name}\", which falls under the \"{category}\" category. "
        f"It features characteristics such as: {description[:200].strip()}... Based on the visual elements, "
        f"this matches the given product details."
    )

def update_json_with_questions(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated_data = []
    for item in data:
        image_id = item.get("id")
        image_file = item.get("image")
        conversations = item.get("conversations", [])

        # Parse metadata from the original human message
        human_messages = [c["value"] for c in conversations if c["from"] == "human"]
        if not human_messages:
            continue

        value = human_messages[0]
        try:
            description = [line for line in value.split('\n') if "Description:" in line][0].split("Description:")[1].strip()
            display_name = [line for line in value.split('\n') if "Display Name:" in line][0].split("Display Name:")[1].strip()
            category = [line for line in value.split('\n') if "Category:" in line][0].split("Category:")[1].strip()
        except IndexError:
            print(f"Skipping malformed item: {image_id}")
            continue

        # Build new conversations
        new_conversations = []

        # Q1: Original
        new_conversations.append({
            "from": "human",
            "value": value
        })
        new_conversations.append({
            "from": "gpt",
            "value": generate_original_response(display_name, description, category)
        })

        # Q2: What is the category?
        new_conversations.append({
            "from": "human",
            "value": "What is the category of this product?"
        })
        new_conversations.append({
            "from": "gpt",
            "value": generate_category_response(category)
        })

        # Q3: Any known product match?
        new_conversations.append({
            "from": "human",
            "value": "Does this product match any known items?"
        })
        new_conversations.append({
            "from": "gpt",
            "value": generate_matching_response(display_name, description)
        })

        updated_data.append({
            "id": image_id,
            "image": image_file,
            "conversations": new_conversations
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, indent=4, ensure_ascii=False)

# Example usage
update_json_with_questions("dataset.json", "updated_with_qna.json")
