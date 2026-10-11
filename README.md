# 📚 BookNest AI — Visual Book Scanner & Story Assistant

An intelligent, mobile-first AI book companion and personal library manager. Built with FastAPI, ConvNeXt / Perceptual Visual Matching, Hybrid Cloud Firestore / SQLite Engine, and an interactive AI Storyteller Chatbot.

---

## 🌟 Core Features

### 1. 📸 Visual Book Cover Scanner & Database Identifier
- **Camera Scanning**: Point your mobile camera at any book cover with a real-time HUD viewfinder, laser sweep animation, and haptic feedback.
- **Database Ownership Check**: Instantly verifies whether the scanned book already exists in your personal library database:
  - 🟢 **"Already in Your Database" / "உங்களிடம் உள்ளது"**
  - 🔵 **"New Book — Not in Database" / "புதிய புத்தகம்"** with 1-tap **"Add to My Library"**.
- **AI Match Confidence**: Visual embedding & feature comparison providing accurate match scoring.

### 2. 💬 Side Mini Chatbot & AI Storyteller
- **Atmospheric Book Scenario**: Automatically extracts the setting, era, and core conflict.
- **Genre & Style Classification**: Badges indicating genre (e.g., Dystopian Sci-Fi, Psychological Drama, Self-Improvement).
- **Story Synopsis & Narrative Hook**: Tells the book's premise like an engaging short story.
- **Conversational Chat**: Ask questions in **English** or **Tamil (தமிழ்)**:
  - *"இந்த கதையின் சுருக்கம் தமிழில் சொல்"*
  - *"Who is the main protagonist?"*
  - *"What is the core moral/life takeaway?"*

### 3. 📚 Personal Library (My Database)
- Real-time search by title or author.
- Filter by reading progress: *All Books*, *Reading*, *Finished*, *Wishlist*.
- One-tap deletion and instant chatbot integration for any saved book.

### 4. 👤 User Profile & Reading Analytics
- Customized profile for personal/brother's reading goals.
- Reading metrics: Books Owned, Total Scans, 2026 Reading Goal progress.
- Live backend engine diagnostics: Vector Engine, Database mode, and Vision AI status.

### 5. 📱 Phone Connect & Instant PWA Installation
- Dynamic on-screen QR Code linking to `http://<local-ip>:8000`.
- Open camera on your iPhone or Android, scan the QR code, and tap **"Add to Home Screen"** to install it as a standalone mobile application with PWA offline support!

---

## 🛠️ Architecture & Upgrades

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Backend** | FastAPI + Uvicorn | Async Python API with backwards compatibility |
| **Vision & Matching** | ConvNeXt / 768-D Perceptual Vectors | Hybrid deep learning & perceptual visual indexing |
| **Book Metadata** | Google Books + OpenLibrary + Wikipedia | High-res covers, author details, synopsis |
| **Database** | Cloud Firestore + Local SQLite Fallback | Resilient zero-config persistence |
| **AI Storyteller** | LLaMA 3.3 / LLaMA 3.2 Vision + Context Engine | Bilingual scenario narration & conversational chat |
| **Frontend** | Mobile-first Responsive PWA | Dark luxury theme, WebRTC camera feed, glassmorphic UI |

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/Sivanesanbalu/Book_AI_Backend.git
cd Book_AI_Backend
pip install -r requirements.txt
```

### 2. Run Live Server
```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access on Desktop or Phone
- **Desktop Browser**: Open [http://localhost:8000](http://localhost:8000)
- **Mobile Phone**: Connect to the same Wi-Fi, scan the QR Code on the **"To Phone"** tab, or open `http://<your-mac-ip>:8000`.

---

## 📡 API Reference

- `GET /` — Serves the mobile-first Web PWA application.
- `POST /scan` — Upload image file → returns book identity, database status (`owned` vs `found`), confidence, and AI scenario.
- `POST /add` — Registers book into vector index and saves to user library.
- `POST /api/chat` — Conversational chatbot endpoint for book scenarios and questions.
- `GET /api/library` — Fetches user's complete saved books.
- `POST /api/library/add` — Adds a book with status and rating.
- `DELETE /api/library/remove?title=...` — Removes a book from library.
- `GET /api/profile` — Retrieves user reading statistics and profile.
- `GET /api/network-info` — Returns local network IP and mobile access URL.
