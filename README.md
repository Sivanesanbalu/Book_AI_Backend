# 📚 BookNest AI — Visual Book Scanner & Story Assistant

<p align="center">
  <img src="https://raw.githubusercontent.com/Sivanesanbalu/Book_AI_Backend/main/static/icons/icon-512.png" width="110" height="110" alt="BookNest AI Logo" style="border-radius: 28px; box-shadow: 0 10px 30px rgba(99,102,241,0.25);" />
</p>

<p align="center">
  <strong>An Intelligent, Mobile-First Vision Book Scanner, Personal Reading Tracker & Groq AI Storyteller</strong>
</p>

<p align="center">
  <a href="https://github.com/Sivanesanbalu/Book_AI_Backend/releases"><img src="https://img.shields.io/badge/Release-v2.7.0%20Production-blue.svg?style=for-the-badge&logo=android&logoColor=white" alt="Release v2.7.0"></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://groq.com"><img src="https://img.shields.io/badge/Groq%20Vision-Qwen%20%26%20Llama-F55036.svg?style=for-the-badge" alt="Groq Vision"></a>
  <a href="https://sqlite.org"><img src="https://img.shields.io/badge/SQLite-WAL%20Engine-003B57.svg?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite WAL"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="MIT License"></a>
</p>

---

## 📖 Overview

**BookNest AI** turns physical book covers into an interactive digital library. Simply snap a photo of any book cover to automatically identify it, detect duplicates, and add it to your shelf. With an embedded focus reading stopwatch, habit trackers, 3D porcelain cards, and a Groq Vision AI Storyteller companion, BookNest AI bridges physical reading with intelligent digital workflows.

---

## 📱 App Preview & Mobile Interface

| 🏠 Home Dashboard & Metrics | 📚 My Library Catalog | 📖 3D Bookshelf Grid |
| :---: | :---: | :---: |
| <img src="docs/screenshots/01_home_dashboard.png" width="260" alt="Home Dashboard" /> | <img src="docs/screenshots/02_library_catalog.png" width="260" alt="Library Catalog" /> | <img src="docs/screenshots/03_library_shelf.png" width="260" alt="Bookshelf Grid" /> |

| ⏱️ Focus Reading Timer | 📸 Google Lens Scanner HUD | 🤖 Groq AI Storyteller |
| :---: | :---: | :---: |
| <img src="docs/screenshots/04_reading_timer.png" width="260" alt="Reading Timer" /> | <img src="docs/screenshots/08_camera_scanner.png" width="260" alt="Camera Scanner" /> | <img src="docs/screenshots/07_ai_storyteller.png" width="260" alt="AI Storyteller" /> |

| 👤 3D Luxury Porcelain Profile | 🔑 1-Tap Google Account Chooser |
| :---: | :---: |
| <img src="docs/screenshots/05_user_profile.png" width="260" alt="User Profile" /> | <img src="docs/screenshots/06_google_account_picker.png" width="260" alt="Google Account Picker" /> |

---

## 📥 Production Mobile APK Download

Get the standalone native Android release build directly:

[![Download APK](https://img.shields.io/badge/⬇️_Download_BookNest_AI_APK-v2.7.0_Release-4F46E5?style=for-the-badge&logo=android&logoColor=white)](https://github.com/Sivanesanbalu/Book_AI_Backend/raw/main/static/BookNestAI-release.apk)

- **Target Architecture**: Android 8.0+ (API 26 to API 35+)
- **Security**: Signed with 2048-bit RSA Certificate via `apksigner`
- **Native OS Bridge**: Direct Android `AccountManager` Google account detection, WebRTC camera hardware access, offline SQLite sync, and smooth 120Hz WebView rendering.

---

## 🌟 Architecture & System Design

```mermaid
flowchart TD
    subgraph Client["📱 Mobile Android App / Web PWA"]
        Cam["📸 Live Camera Stream (WebRTC)"]
        UI["🎨 Clean White 3D Porcelain UI"]
        Bridge["⚡ AndroidAuthBridge (JavaScriptInterface)"]
        AccMgr["🔑 Android OS AccountManager"]
        Bot["🤖 Movable Vector AI Bot (Pointer Capture)"]
    end

    subgraph Edge["☁️ Cloud Edge / Secure Gateway"]
        Tunnel["🛡️ Cloudflare Zero-Trust Edge / HTTPS Gateway"]
    end

    subgraph Backend["⚡ Backend Core (FastAPI / ASGI)"]
        Router["🚀 FastAPI High-Throughput Endpoints"]
        AuthHandler["🔐 Google Auth & Session Engine"]
        VisionPipeline["🧠 Vision Pre-Processing & OCR Filter"]
    end

    subgraph Intelligence["🔮 AI & Database Layer"]
        Groq["⚡ Groq Vision AI (Qwen 2.5 / Llama 3.2 Multimodal)"]
        SQLStore[("💾 SQLite WAL Engine (Thread-Safe Lock)")]
        Catalog[("📚 1,005 Preloaded Master Catalog")]
    end

    Cam -->|Base64 Frame| Router
    Bridge <-->|Query Phone Accounts| AccMgr
    Bridge -->|Auto-Detect Accounts| UI
    UI -->|1-Tap Switch| AuthHandler
    Router -->|Cover Image Payload| VisionPipeline
    VisionPipeline -->|Vision Query| Groq
    Groq -->|Extracted Title & Author| Router
    Router <-->|Query & Persist| SQLStore
    Router <-->|Similarity Search| Catalog
    Router -->|JSON Response| UI
```

---

## 🔍 How Cover Recognition Works

1. **Capture & Preprocessing**: The web app captures a camera frame or uploaded image and posts it to `/scan` (up to 10 MiB).
2. **Local Feature Matching**: The backend extracts visual features using ConvNeXt embeddings / perceptual image fingerprints, cross-referencing against our preloaded 1,005-book master catalog.
3. **Groq Multimodal AI Fallback**: If local confidence falls below threshold, the image is passed to Groq Vision AI (`qwen-2.5-32b` / `llama-3.2-11b-vision-preview`), extracting exact title, author, and edition metadata.
4. **Metadata Enrichment**: Matches are cross-referenced with Google Books API and Open Library for descriptions, page counts, and publication details.
5. **Ownership Status**: Returns status (`found`, `owned`, or `not_found`) with 3D porcelain cards.

---

## 🚀 Key Engineering Features

- **Zero Hardcoded Localhost**: Dynamically resolves endpoints through relative `/api` paths and Cloudflare HTTPS edge tunneling.
- **Physical Phone Google Account Sync**: Seamlessly queries device accounts via native Android `AccountManager` and `AndroidAuthBridge`, enabling instant one-tap switching.
- **Thread-Safe SQLite WAL Engine**: Full WAL (Write-Ahead Logging) mode with explicit threading locks, guaranteeing crash resilience and zero race conditions under concurrent reads and writes.
- **3D Porcelain Micro-Interactions**: Custom CSS hardware-accelerated transforms, gesture swipes, Pomodoro focus timer with audible chimes, and a floating interactive AI companion.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/` | Serves the 3D Single Page Web App |
| `POST` | `/scan` | Uploads book cover image for Groq AI Vision identification |
| `GET` | `/api/library` | Retrieves user's personal book collection |
| `POST` | `/api/library/add` | Adds a new book to the personal library |
| `POST` | `/api/library/update` | Updates reading progress, status, pages, or rating |
| `DELETE` | `/api/library/remove` | Removes a book from personal collection |
| `POST` | `/api/timer/session` | Logs reading stopwatch session (minutes & pages) |
| `POST` | `/api/chat` | AI Story Companion conversation endpoint |
| `GET` | `/api/profile` | Fetches user profile metrics, goals, and badges |
| `POST` | `/api/profile` | Auto-saves profile adjustments (name, goal) |
| `POST` | `/api/profile/avatar` | Handles custom photo/avatar uploads |
| `POST` | `/api/auth/google` | Authenticates Google user account and mirrors profile |
| `GET` | `/download` | Downloads the compiled Android release APK |

---

## ⚙️ Configuration & Environment Variables

| Variable | Description |
| --- | --- |
| `GROQ_API_KEY` | Remote multimodal vision and AI storytelling via Groq cloud API. |
| `FIREBASE_KEY` | Optional Firebase Admin service-account JSON for ID-token verification. |
| `PORT` | Backend service port (defaults to `8000`). |

---

## 📁 Code Map

| Path | Responsibility |
| --- | --- |
| [`main.py`](main.py) | FastAPI routes for scans, library entries, reading sessions, recommendations, and chat |
| [`image_embedder.py`](image_embedder.py) & [`image_search.py`](image_search.py) | Cover features and vector similarity search |
| [`vision_ai/`](vision_ai/) | Groq multimodal integration, metadata lookups, and book discussions |
| [`firebase_service.py`](firebase_service.py) | Thread-safe SQLite persistence and optional Firebase integration |
| [`android/`](android/) | Standalone Android Native Java wrapper with `AccountManager` bridge |
| [`static/`](static/) | 3D Porcelain web interface, WebRTC camera, and PWA assets |
| [`docs/screenshots/`](docs/screenshots/) | High-DPI privacy-clean production screenshots |

---

## 👨‍💻 Author

**Sivanesan Balu**
- GitHub: [@Sivanesanbalu](https://github.com/Sivanesanbalu)
- Project Repository: [Book_AI_Backend](https://github.com/Sivanesanbalu/Book_AI_Backend)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
