from openai import OpenAI
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
import requests
import os
import re
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def validate_output(input_text, output_text, threshold=0.7):
    similarity = SequenceMatcher(None, input_text, output_text).ratio()
    print(f"Text similarity: {similarity}")
    return similarity >= threshold

def validate_caption(original_caption, processed_caption, threshold=0.8):
    similarity = SequenceMatcher(None, original_caption, processed_caption).ratio()
    print(f"Caption similarity score: {similarity}")
    return similarity >= threshold

def search_google_with_serper(query):
    headers = {
        "X-API-KEY": os.getenv("X-API-KEY"),
        "Content-Type": "application/json"
    }
    payload = {"q": query}
    try:
        response = requests.post("https://google.serper.dev/search", json=payload, headers=headers)
        response.raise_for_status()
        results = response.json()
        return results["organic"][0].get("link") if results.get("organic") else None
    except Exception as e:
        print("Serper error:", e)
        return None

def search_for_image_vietnamplus(headline, article_text, primary_model="gpt-4o", fallback_model="gpt-4o-mini-2024-07-18"):
    query = f"site:en.vietnamplus.vn {headline}"
    article_url = search_google_with_serper(query)
    if not article_url:
        return None, "No image found"

    try:
        res = requests.get(article_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        img = soup.find("img", {"class": "cms-photo"})
        if not img:
            return None, "No image found"
        url = img["src"]
        original_caption = img.get("alt", "").replace(" (Photo: VNA)", " VNA/VNS Photo")

        for model in [primary_model, fallback_model]:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": (
                            "Add Vietnamese diacritics to proper nouns, places, and terms only. "
                            "Do NOT translate or modify English terms. Keep format and structure intact."
                        )},
                        {"role": "user", "content": original_caption}
                    ],
                    max_tokens=512,
                    temperature=0.0
                )
                processed = response.choices[0].message.content
                if validate_caption(original_caption, processed):
                    return url, processed
            except Exception as e:
                print(f"Caption model error: {e}")

        return url, original_caption
    except Exception as e:
        print("Error fetching image:", e)
        return None, "No image found"

def format_article(text):
    lines = text.split("\n")
    if len(lines) >= 3:
        third = lines[2]
        match = re.match(r"(.*?),.*?\\(VNA\\)\\s*–\\s*(.*)", third)
        if match:
            location, content = match.groups()
            lines[2] = f"{location.strip().upper()} — {content.strip()}"
    lines = [l.strip() for l in lines if l.strip()]
    if len(lines) > 1:
        lines[-2] = lines[-2].replace("./.", ".") + " — VNS"
        lines = lines[:-1]
    return "\n".join(lines)

def add_diacritics_to_text(original_text, primary_model="gpt-4o", fallback_model="gpt-4o-mini-2024-07-18"):
    system_message = (
        "You are a Vietnamese language assistant. Your ONLY task is to add diacritics to Vietnamese proper nouns, "
        "places, and terms in the given text.\n"
        "- Do NOT translate any text.\n"
        "- Do NOT modify English or foreign terms.\n"
        "- Keep structure, language, and punctuation exactly the same.\n"
        "- Foreign country names like 'Egypt', 'Finland', 'Japan', 'France', etc., must remain unchanged.\n"
        "- Words like 'Duc' in names (e.g. 'Ho Duc Phoc') are personal names, not countries, and should be diacritised properly.\n"
        "- Currency should be formatted as: VNĐamount (US$amount). Do not convert USD, just format as US$amount.\n"
        "- Use 'per cent' instead of '%'.\n"
        "- Use the format 'July 5' for dates.\n"
        "- Use '5.30pm' or '5am' for time.\n"
        "- Numerals 0-9 should be written as words (zero through nine) except in dates or decimals.\n"
        "- Ordinals 1st-9th should be written as: first, second, ..., ninth.\n"
        "Examples:\n"
        "- 'Vietnam' → 'Việt Nam'\n"
        "- 'Ha Noi' → 'Hà Nội'\n"
        "- 'Nguyen Van A' → 'Nguyễn Văn A'\n"
        "- 'Ho Chi Minh City' → 'Hồ Chí Minh City'\n"
        "- 'National Assembly' remains unchanged\n"
    )

    excluded_terms = {
        "Ai Cập": "Egypt",
        "Phần Lan": "Finland",
        "Pháp": "France",
        "Đức": "Germany",
        "Nhật Bản": "Japan",
        "Lào": "Laos",
        "Trung Quốc": "China",
    }

    for attempt, model in enumerate([primary_model, fallback_model]):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": original_text}
                ],
                max_tokens=2048,
                temperature=0.0
            )
            processed = response.choices[0].message.content

            # Revert mistranslated foreign terms if necessary
            for term_vi, term_en in excluded_terms.items():
                processed = processed.replace(term_vi, term_en)

            if validate_output(original_text, processed):
                return processed
        except Exception as e:
            print(f"Model {model} failed:", e)

    raise ValueError("Failed to add diacritics after all attempts.")
