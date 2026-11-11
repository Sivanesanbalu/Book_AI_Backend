# 📚 BookNest AI — Visual Book Scanner & Story Assistant

An intelligent, mobile-first book scanner and personal library manager with 1,000+ preloaded books, Groq Multimodal Vision AI, and a bilingual AI Storyteller companion.

---

## 🔗 Live Access & Downloads

- 💻 **Desktop Web App**: [http://localhost:8000](http://localhost:8000)
- 📱 **Mobile Phone Web (HTTPS)**: [https://fun-equality-becomes-typical.trycloudflare.com](https://fun-equality-becomes-typical.trycloudflare.com)
- 📥 **Android Release APK**: [Download BookNest AI APK (v1.0.0)](https://github.com/Sivanesanbalu/Book_AI_Backend/releases/download/v1.0.0/BookNestAI-release.apk)

---

## 🌟 What BookNest AI Does

### 1. 📸 Visual Book Cover Scanner
- Point your mobile or webcam at any physical book cover.
- Animated HUD viewfinder with neon laser sweep and 3D hologram loading steps.
- Recognizes book title and author using Groq's high-speed neural vision (`qwen/qwen3.8-27b`).
- Instantly checks whether the book is **Already In Your Database** or a **New Book**.

### 2. 📚 1,005 Preloaded Books across 9 Categories
Explore and filter a curated master catalog of **1,005 books**:
- 💥 **Comics**: Spider-Man, Batman, Watchmen, Sandman, Tintin, Asterix
- 🎌 **Manga**: One Piece, Naruto, Bleach, Attack on Titan, Death Note, Demon Slayer
- 📚 **Fiction**: Harry Potter, Lord of the Rings, 1984, To Kill a Mockingbird
- 🚀 **Sci-Fi & Fantasy**: Dune, Foundation, Neuromancer, Three-Body Problem
- 💡 **Self-Help**: Atomic Habits, Deep Work, Think & Grow Rich, Ikigai
- 💻 **Technology & Code**: Clean Code, Pragmatic Programmer, Designing Data-Intensive Apps
- 🕵️ **Mystery & Thriller**: Sherlock Holmes, Da Vinci Code, Gone Girl, Agatha Christie
- 📜 **History & Bio**: Sapiens, Steve Jobs, Wings of Fire, Ponniyin Selvan
- 💼 **Business & Finance**: Zero to One, Psychology of Money, The Intelligent Investor

### 3. ➕ Custom Topic & Category Creator
- Click the **`+ Add Topic`** button in the category bar to type any personalized genre or study topic.
- Tap the **`✓` (Save)** button or hit Enter to immediately save and filter books by your custom topic.
- Custom categories persist across sessions and automatically populate all book assignment dropdowns.

### 3. ⚪ Clean White 3D Luxury Design
- Modern porcelain white glassmorphic aesthetic with 3D card depth.
- Physical 3D book cover tilt with spine depth and lighting highlights.
- **Theme Switcher**: 1-click toggle between **Clean White** and **Dark Mode**.

### 4. 🔐 Google Login & Avatar Customization
- One-tap Google Sign-In syncing your library and profile.
- Tap your avatar to upload a custom profile picture or choose from 3D avatars.

### 5. 📖 3D Reading Tracker & Confetti Celebration
- Tap any book to open its interactive 3D Reading Screen.
- Manage status: `📖 Reading`, `🎉 Completed`, or `⭐ Wishlist`.
- Give 1 to 5 star ratings.
- **Confetti Cannon Animation**: Marking a book as "Completed" triggers a full-screen celebratory confetti burst!
- Live metrics: Total Books, Completed Books, Reading Now, and 2026 Goal Progress.

### 6. 💬 Intelligent AI Story Companion
- Automatically generates atmospheric world setting, narrative hook, and key takeaways.
- Interactive AI chat to discuss characters, plot twists, and life lessons from any book in your collection.

---

## 🚀 Quick Start (Run Locally)

```bash
# 1. Clone the repository
git clone https://github.com/Sivanesanbalu/Book_AI_Backend.git
cd Book_AI_Backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key in .env
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Run server
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 📁 Project Structure

```text
Book_AI_Backend/
├── main.py                  # FastAPI server & API routes
├── firebase_service.py      # SQLite mirror & user persistence engine
├── vision_ai/
│   ├── vision.py            # Groq multimodal vision cover recognition
│   ├── ai_summary.py        # Bilingual AI storyteller & scenario generator
│   └── book_fetcher.py      # OpenLibrary & Google Books metadata fetcher
├── data/
│   └── books.json           # 1,005 books catalog database
├── static/
│   ├── index.html           # Single Page Web App (PWA)
│   ├── css/app.css          # Clean White 3D luxury stylesheet
│   ├── js/app.js            # Frontend reactive controller & confetti
│   ├── manifest.json        # PWA manifest
│   └── BookNestAI-release.apk # Production signed Android release APK
├── android/                 # Native Android app source code
└── scripts/
    └── generate_1000_books.py # Catalog generator & SQLite seeder
```

---

## 📡 API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Serves the 3D Web PWA |
| `/scan` | `POST` | Upload book cover photo → returns title, author, and scenario |
| `/api/library` | `GET` | Fetches user's saved books |
| `/api/library/add` | `POST` | Saves book with category and reading status |
| `/api/library/update` | `POST` | Updates reading status (e.g. Completed), rating, or category |
| `/api/library/remove` | `DELETE` | Removes book from library |
| `/api/chat` | `POST` | Bilingual AI Storyteller chatbot |
| `/api/profile` | `GET` / `POST` | Fetches / updates user profile, goal, and avatar |
| `/api/auth/google` | `POST` | Authenticates Google user |
| `/download` | `GET` | Downloads production signed Android APK |
