from openai import OpenAI
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
import requests
import os
import re
from dotenv import load_dotenv

load_dotenv()

# Create a reusable client object
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def validate_output(input_text, output_text, threshold=0.7):
    similarity = SequenceMatcher(None, input_text, output_text).ratio()
    print(f"Text similarity: {similarity}")
    return similarity >= threshold

def validate_caption(original_caption, processed_caption, threshold=0.8):
    similarity = SequenceMatcher(None, original_caption, processed_caption).ratio()
    print(f"Caption similarity: {similarity}")
    return similarity >= threshold

def search_google_with_serper(query):
    headers = {
        "X-API-KEY": os.getenv("X-API-KEY"),
        "Content-Type": "application/json"
    }
    payload = { "q": query }
    try:
        response = requests.post("https://google.serper.dev/search", json=payload, headers=headers)
        response.raise_for_status()
        results = response.json()
        return results["organic"][0].get("link") if results.get("organic") else None
    except Exception as e:
        print("Serper error:", e)
        return None

def search_for_image_vietnamplus(headline, article_text):
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

        for model in ["gpt-4o"]:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": (
                            "Add Vietnamese diacritics to proper nouns only. Don't translate anything. Keep English terms untouched.")},
                        {"role": "user", "content": original_caption}
                    ],
                    max_tokens=512,
                    temperature=0
                )
                processed = response.choices[0].message.content
                if validate_caption(original_caption, processed):
                    return url, processed
            except Exception as e:
                print("Caption model error:", e)

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
    prompt = (
        "You are a Vietnamese language assistant. Add diacritics to Vietnamese proper nouns, places, and terms only. "
        "Do NOT translate or modify English terms. Keep structure intact.\n"
        "- 'Vietnam' → 'Việt Nam'\n"
        "- 'Ha Noi' → 'Hà Nội'\n"
        "- Leave 'National Assembly' and other English phrases unchanged."
    )

    for attempt, model in enumerate([primary_model, fallback_model]):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": original_text}
                ],
                max_tokens=2048,
                temperature=0.0
            )
            output = response.choices[0].message.content
            if validate_output(original_text, output):
                return output
        except Exception as e:
            print(f"Diacritics model {model} failed:", e)

    raise ValueError("Failed to add diacritics after all attempts.")
