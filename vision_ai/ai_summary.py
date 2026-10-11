import requests
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

CHAT_SYSTEM_PROMPT = """
You are 'BookNest AI', an empathetic, highly knowledgeable literary companion and book storyteller.
Your mission:
1. Explain books vividly — set the stage with the dramatic scenario, explain the genre/type of book, and narrate the core premise like a captivating short story.
2. Keep responses punchy, engaging, and clear.
3. If the user asks in Tamil or asks for Tamil explanation, explain gracefully in conversational Tamil (எளிய தமிழில் கதை போல சுருக்கமாக).
4. Be interactive: highlight key conflicts, themes, characters, and practical life takeaways.
"""


def generate_book_scenario(book: dict):
    """
    Generates a vivid scenario, book genre classification, and short story overview.
    """
    title = book.get("title", "Unknown Book")
    author = book.get("authors", "Unknown Author")
    desc = book.get("description", "")
    genre = book.get("genre", "General Literature")

    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            prompt = f"""
Book: {title} by {author}
Genre: {genre}
Description: {desc[:1000]}

Please generate a JSON object with:
1. "scenario": A dramatic 2-sentence setting/world description setting the atmosphere.
2. "genre_type": The specific book style/genre (e.g. Dystopian Sci-Fi, Psychological Drama, Self-Development).
3. "story_narrative": A thrilling 3-4 sentence short story summary of the plot and core dilemma.
4. "key_takeaway": 1 memorable life lesson or thought-provoking takeaway.
5. "vibe": 3 punchy keywords separated by bullets (e.g. "Dark • Intriguing • Masterpiece").

Output ONLY raw valid JSON without markdown fences.
"""
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You are a master literary critic and book storyteller. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 400
            }
            res = requests.post(url, headers=headers, json=payload, timeout=20)
            if res.status_code == 200:
                text = res.json()["choices"][0]["message"]["content"].strip()
                # Clean any markdown code blocks
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                data = json.loads(text.strip())
                return {
                    "scenario": data.get("scenario", f"Set in the world of {title}, exploring deep human emotions and turning points."),
                    "genre_type": data.get("genre_type", genre),
                    "story_narrative": data.get("story_narrative", desc[:250] + "..."),
                    "key_takeaway": data.get("key_takeaway", "Every choice defines the path ahead."),
                    "vibe": data.get("vibe", "Engaging • Inspiring • Classic")
                }
        except Exception as e:
            print("⚠️ Groq scenario generation error:", e)

    # High quality fallback scenario builder
    clean_desc = desc if len(desc) > 50 else f"A celebrated narrative by {author} diving into human nature, choices, and resilience."
    sentences = [s.strip() for s in clean_desc.split(". ") if len(s.strip()) > 10]

    scenario_text = sentences[0] + "." if sentences else f"Set in an immersive universe crafted by {author}."
    narrative_text = " ".join(sentences[1:4]) if len(sentences) > 1 else clean_desc[:250] + "..."

    return {
        "scenario": f"🌍 Setting: {scenario_text}",
        "genre_type": f"📖 {genre}",
        "story_narrative": narrative_text,
        "key_takeaway": f"💡 Discover how {author} challenges our worldview through compelling storytelling.",
        "vibe": "Thought-provoking • Captivating • Must-Read"
    }


def chat_with_book_ai(book: dict, user_message: str, history: list = None):
    """
    Interactive conversational response from the Book AI mini chatbot.
    """
    if history is None:
        history = []

    title = book.get("title", "Selected Book")
    author = book.get("authors", "")
    desc = book.get("description", "")
    genre = book.get("genre", "Literature")

    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            messages = [
                {
                    "role": "system",
                    "content": f"{CHAT_SYSTEM_PROMPT}\n\nCurrent Book Context: Title: '{title}', Author: '{author}', Genre: '{genre}'. Details: {desc[:1200]}"
                }
            ]

            # Add recent history (up to last 6 messages)
            for h in history[-6:]:
                role = "user" if h.get("sender") == "user" else "assistant"
                messages.append({"role": role, "content": h.get("text", "")})

            messages.append({"role": "user", "content": user_message})

            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "temperature": 0.6,
                "max_tokens": 350
            }

            res = requests.post(url, headers=headers, json=payload, timeout=20)
            if res.status_code == 200:
                reply = res.json()["choices"][0]["message"]["content"].strip()
                return reply
        except Exception as e:
            print("⚠️ Groq chat error:", e)

    # Smart contextual fallback chatbot response
    msg_lower = user_message.lower()

    # Detect any Tamil script characters (\u0B80-\u0BFF) or keyword
    has_tamil_script = bool(re.search(r'[\u0B80-\u0BFF]', user_message))
    is_tamil = has_tamil_script or any(w in msg_lower for w in ["tamil", "tanglish", "thamil"])

    if is_tamil:
        scenario = generate_book_scenario(book)
        return (
            f"📖 **'{title}' — கதை மற்றும் பின்னணி (Story & Scenario)**:\n\n"
            f"இப்புத்தகம் **{author}** அவர்களால் எழுதப்பட்ட ஒரு சிறந்த **{genre}** படைப்பாகும்.\n\n"
            f"🔹 **கதைக் களம் (Scenario)**: மனித வாழ்க்கை, சவால்கள் மற்றும் முக்கியமான திருப்பங்களை மையமாகக் கொண்டது.\n\n"
            f"🔹 **கதைச் சுருக்கம்**: {scenario.get('story_narrative', desc[:200])}\n\n"
            f"💡 **முக்கிய பாடம்**: {scenario.get('key_takeaway', 'வாழ்க்கையின் முக்கிய முடிவுகளை ஆராய்கிறது.')}\n\n"
            f"வேறு ஏதேனும் தகவல்கள் அல்லது குறிப்பிட்ட கதாபாத்திரங்கள் பற்றி தெரிந்து கொள்ள வேண்டுமா?"
        )

    if any(w in msg_lower for w in ["character", "who", "protagonist", "hero"]):
        return f"In '{title}', the central characters navigate pivotal dilemmas. The protagonist represents human struggle and courage against overwhelming odds in {genre}."

    if any(w in msg_lower for w in ["takeaway", "moral", "lesson", "why read"]):
        return f"The strongest takeaway from '{title}' is how our choices shape our destiny. It gives students and readers a clear perspective on resilience and critical thinking."

    if any(w in msg_lower for w in ["scenario", "story", "summary", "explain", "plot"]):
        scenario = generate_book_scenario(book)
        return f"📚 **Scenario & Story**:\n\n{scenario['scenario']}\n\n{scenario['story_narrative']}\n\n💡 **Core Insight**: {scenario['key_takeaway']}"

    return f"Regarding '{title}': This masterpiece by {author} blends rich storytelling with deep thematic questions. Would you like to know about the main plot twists, key quotes, or similar book recommendations?"


def summarize_book(book: dict):
    """
    Backwards-compatible one-shot summary function
    """
    scenario = generate_book_scenario(book)
    return f"{scenario['scenario']}\n\n{scenario['story_narrative']}\n\nKey Takeaway: {scenario['key_takeaway']}"