# 📚 BookNest AI — Visual Book Scanner, 1,000+ Library & AI Story Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-indigo.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20v2.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Groq Vision](https://img.shields.io/badge/Vision%20AI-Groq%20Qwen%203.8%20(27B)-f97316.svg)](https://groq.com/)
[![Android](https://img.shields.io/badge/Android-APK%20Release%20Ready-34d399.svg)](https://developer.android.com/)
[![Design](https://img.shields.io/badge/UI-Clean%20White%203D%20Luxury-6366f1.svg)](https://github.com/Sivanesanbalu/Book_AI_Backend)

An intelligent, mobile-first AI book companion, personal library manager, and visual scanner. Built with **FastAPI**, **Groq Neural Multimodal Vision**, **Hybrid Cloud Firestore / SQLite Mirror**, **1,005 Preloaded Curated Books**, and a **Bilingual Tamil & English AI Storyteller Chatbot**.

---

## 🌟 Core Features

### 1. 📸 3D Holographic Visual Book Cover Scanner
- **Live Camera Scanning**: Point your camera at any book cover with a real-time HUD viewfinder, laser sweep animation, and simulated 3D holographic book loading steps.
- **Groq Neural Vision**: Ultra-fast cover recognition and typography extraction powered by Groq's high-speed multimodal inference.
- **Database Ownership Check**: Instantly verifies whether the scanned book already exists in your library:
  - 🟢 **"Already in Your Database" / "உங்களிடம் உள்ளது"**
  - 🔵 **"New Book — Not in Database" / "புதிய புத்தகம்"** with 1-tap **"Save to My Library"**.
- **Category Auto-Tagging**: Detects and suggests appropriate categories (Comics, Manga, Fiction, Self-Help, etc.) upon scanning.

### 2. 📚 1,000+ Curated Books across 9 Categories
Preloaded master catalog with **1,005 books** spanning 9 rich genres:
- 💥 **Comics** (Spider-Man, Batman, Watchmen, Sandman, Avengers, Tintin, Asterix)
- 🎌 **Manga** (One Piece, Naruto, Bleach, Attack on Titan, Death Note, Demon Slayer, Jujutsu Kaisen)
- 📚 **Fiction** (Harry Potter, Lord of the Rings, 1984, To Kill a Mockingbird, The Great Gatsby)
- 🚀 **Sci-Fi & Fantasy** (Dune, Foundation, Neuromancer, The Three-Body Problem, The Martian)
- 💡 **Self-Help** (Atomic Habits, Deep Work, Think and Grow Rich, Ikigai, Can't Hurt Me)
- 💻 **Technology & Code** (Clean Code, Pragmatic Programmer, Designing Data-Intensive Applications)
- 🕵️ **Mystery & Thriller** (Sherlock Holmes, Da Vinci Code, Gone Girl, And Then There Were None)
- 📜 **History & Bio** (Sapiens, Steve Jobs, Wings of Fire, Ponniyin Selvan, Alexander Hamilton)
- 💼 **Business & Finance** (Zero to One, The Psychology of Money, The Intelligent Investor)

### 3. ⚪ Clean White 3D Luxury Real App UI
- **Porcelain Light Theme**: Minimalist, high-end Apple / Linear design with soft layered shadows and glassy acrylic cards.
- **3D Book Cover Perspective**: Physical book depth with spine lighting gradients and hover/tilt micro-animations.
- **Theme Switcher**: 1-tap toggle between **Clean White** and **Dark Luxury Mode**.

### 4. 🔐 Google Login & Avatar Customization
- One-tap **Google Sign-In** with profile picture, name, and email sync.
- **Custom Photo Upload**: Tap the avatar to upload any image directly from phone/Mac, or pick from stylish 3D avatar presets.

### 5. 📖 3D Reading Screen & Celebration Counter
- Interactive `<dialog>` Reading Modal for every book:
  - 3D physical cover presentation.
  - Category tag selector.
  - 5-Star interactive rating system.
  - Atmospheric AI scenario and narrative premise.
- **Confetti Particle Celebration**: Marking a book as **"Completed (படிச்சுட்டேன்! 🎉)"** launches a confetti particle cannon across the screen with congratulations!
- **Real-Time Analytics Dashboard**: Tracks Total Books, Completed Books, Reading Now, and 2026 Goal Progress Bar.

### 6. 💬 Bilingual Tamil & English AI Storyteller Chatbot
- Atmospheric narrative scenario synthesis (world, conflict, core theme).
- Chat interactively in **English** or **conversational Tamil (எளிய தமிழில் கதை போல)**:
  - *"இந்த கதையின் சுருக்கம் தமிழில் சொல்"*
  - *"Who is the main protagonist?"*
  - *"What is the core moral/life takeaway?"*

---

## 🛠️ Architecture & Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Backend API** | FastAPI + Uvicorn | Async Python API with high-throughput endpoints |
| **Vision AI** | Groq Qwen 3.8 (27B Vision) | Multimodal neural book cover identification |
| **Storyteller LLM** | Groq Qwen / LLaMA | Bilingual scenario narration & chat |
| **Book Metadata** | OpenLibrary + Google Books | Covers, author details, synopsis |
| **Database** | SQLite Mirror + Cloud Firestore | Zero-config persistent local storage |
| **Frontend** | Modern Vanilla PWA | Clean White 3D CSS, WebRTC Camera, Canvas Confetti |
| **Android Packaging** | Java 8 + Android SDK (aapt/d8) | Native signed APK (`BookNestAI-release.apk`) |

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/Sivanesanbalu/Book_AI_Backend.git
cd Book_AI_Backend
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Create a `.env` file with your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run Server
```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access App
- **Desktop Browser**: Open [http://localhost:8000](http://localhost:8000)
- **Mobile Phone (HTTPS)**: Use Cloudflare tunnel or local network IP to scan books with camera permissions.

---

## 📱 Google Play Store Deployment Guide

The native Android app is packaged under:
- **Package Name**: `com.booknest.ai`
- **Application Name**: `BookNest AI`
- **Version Code**: `1`
- **Version Name**: `1.0.0`
- **Release APK**: [`static/BookNestAI-release.apk`](file:///Users/sivanesanbalu/.gemini/antigravity/scratch/Book_AI_Backend/static/BookNestAI-release.apk)
- **Release Keystore**: `booknest-release.keystore` (Alias: `booknest`)

### Steps to Submit to Google Play Console:
1. **Open Google Play Console**: Log in to [play.google.com/console](https://play.google.com/console).
2. **Create New App**:
   - App Name: `BookNest AI - Visual Book Scanner & Story Assistant`
   - Default language: `English (United States)`
   - App or Game: `App` | Free or Paid: `Free`
3. **App Listing & Store Assets**:
   - **Short Description**: Smart visual book scanner, personal library manager & AI story companion.
   - **Full Description**: BookNest AI transforms how you discover and track books. Point your camera at any book cover to instantly identify titles, check your personal library, explore 1,000+ curated books across Comics, Manga, and Fiction, and discuss books with an AI Storyteller in English and Tamil.
   - **App Icon**: 512x512 PNG.
   - **Feature Graphic**: 1024x500 PNG.
   - **Screenshots**: Upload 4-8 screenshots of the Scanner, Library, and 3D Reading Screen.
4. **Upload Release Build**:
   - Go to **Production** > **Create new release**.
   - Upload `BookNestAI-release.apk` (or build `.aab` using Android Studio / bundletool).
   - Enter Release Notes: *"Initial release of BookNest AI with 1,000+ book catalog, 3D scanning, and reading tracker."*
5. **Review & Rollout**: Complete Content Rating, Target Audience, and Privacy Policy, then submit for review!

---

## 📡 API Endpoints

- `GET /` — Serves the Clean White 3D Web PWA.
- `POST /scan` — Upload cover photo → returns book identity, database status, and AI scenario.
- `POST /api/library/add` — Adds a book with category and reading status.
- `POST /api/library/update` — Updates reading status (e.g. Completed), rating, or category.
- `DELETE /api/library/remove?title=...` — Deletes book from library.
- `POST /api/chat` — Conversational bilingual AI Storyteller chatbot.
- `GET /api/profile` — Returns user profile, completed count, and reading goal progress.
- `POST /api/auth/google` — Authenticates user and syncs Google profile.
- `POST /api/profile/avatar` — Uploads custom profile photo.
- `GET /BookNestAI-release.apk` — Downloads production-signed Android APK.
- `GET /api/network-info` — Returns live network and tunnel URLs.
