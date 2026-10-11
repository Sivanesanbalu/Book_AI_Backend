// BookNest AI - Mobile Frontend Controller
let currentStream = null;
let activeBook = null;
let chatHistory = [];
let libraryBooks = [];

// DOM Elements
const videoEl = document.getElementById('camera-video');
const shutterBtn = document.getElementById('shutter-btn');
const photoInput = document.getElementById('photo-input');
const uploadBtn = document.getElementById('upload-btn');
const switchCamBtn = document.getElementById('switch-cam-btn');
const resultCard = document.getElementById('scan-result-card');
const chatDrawer = document.getElementById('chat-drawer');
const toastEl = document.getElementById('toast');

let facingMode = 'environment';

// =========================================================
// 🚀 INITIALIZATION
// =========================================================
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initCamera();
  loadLibrary();
  loadProfile();
  loadNetworkInfo();
  registerPWA();

  // Shutter action
  shutterBtn.addEventListener('click', captureAndScan);

  // File upload fallback
  uploadBtn.addEventListener('click', () => photoInput.click());
  photoInput.addEventListener('change', handleFileSelected);

  // Switch camera
  if (switchCamBtn) {
    switchCamBtn.addEventListener('click', toggleCamera);
  }

  // Chat close
  document.getElementById('close-chat-btn').addEventListener('click', () => {
    chatDrawer.classList.remove('open');
  });

  // Chat send
  document.getElementById('chat-form').addEventListener('submit', (e) => {
    e.preventDefault();
    sendMessage();
  });

  // Prompt suggestions
  document.querySelectorAll('.prompt-pill').forEach((pill) => {
    pill.addEventListener('click', () => {
      document.getElementById('chat-input').value = pill.textContent.trim();
      sendMessage();
    });
  });

  // Library search
  document.getElementById('library-search')?.addEventListener('input', filterLibrary);
});

// =========================================================
// 🧭 TAB NAVIGATION
// =========================================================
function initNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  const tabs = document.querySelectorAll('.tab-content');

  navItems.forEach((item) => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const targetTab = item.getAttribute('data-tab');

      navItems.forEach((n) => n.classList.remove('active'));
      tabs.forEach((t) => t.classList.remove('active'));

      item.classList.add('active');
      const activeTabEl = document.getElementById(targetTab);
      if (activeTabEl) activeTabEl.classList.add('active');

      // Camera lifecycle
      if (targetTab === 'tab-scan') {
        if (!currentStream) initCamera();
      } else {
        stopCamera();
      }

      if (targetTab === 'tab-library') loadLibrary();
      if (targetTab === 'tab-profile') loadProfile();
      if (targetTab === 'tab-connect') loadNetworkInfo();
    });
  });
}

// =========================================================
// 📷 CAMERA WEBRTC STREAM
// =========================================================
async function initCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    console.warn('getUserMedia not supported on this browser');
    return;
  }

  try {
    stopCamera();
    const constraints = {
      video: {
        facingMode: facingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    };

    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    if (videoEl) {
      videoEl.srcObject = currentStream;
      await videoEl.play();
    }
  } catch (err) {
    console.warn('Camera access denied or unavailable:', err);
    showToast('⚠️ Camera preview unavailable, use Upload button');
  }
}

function stopCamera() {
  if (currentStream) {
    currentStream.getTracks().forEach((track) => track.stop());
    currentStream = null;
  }
}

function toggleCamera() {
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  initCamera();
}

// =========================================================
// 🔍 SCANNING & IDENTIFICATION
// =========================================================
async function captureAndScan() {
  if (!videoEl || !videoEl.videoWidth) {
    photoInput.click();
    return;
  }

  // Create canvas snapshot
  const canvas = document.createElement('canvas');
  canvas.width = videoEl.videoWidth;
  canvas.height = videoEl.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

  canvas.toBlob((blob) => {
    if (blob) {
      const file = new File([blob], 'scan.jpg', { type: 'image/jpeg' });
      processScanFile(file);
    }
  }, 'image/jpeg', 0.9);
}

function handleFileSelected(e) {
  const file = e.target.files[0];
  if (file) {
    processScanFile(file);
  }
}

async function processScanFile(file) {
  showToast('🔍 Analyzing book cover...');
  shutterBtn.style.pointerEvents = 'none';
  shutterBtn.style.opacity = '0.5';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/scan', {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer default_user'
      },
      body: formData
    });

    const data = await res.json();
    console.log('SCAN RESULT:', data);

    if (data.status === 'not_found' || !data.title) {
      showToast('❌ Could not recognize book. Try closer angle.');
    } else {
      displayScanResult(data);
    }
  } catch (err) {
    console.error('Scan error:', err);
    showToast('⚠️ Network or server error during scan');
  } finally {
    shutterBtn.style.pointerEvents = 'auto';
    shutterBtn.style.opacity = '1';
  }
}

// =========================================================
// 🎯 DISPLAY SCAN RESULT & DB STATUS
// =========================================================
function displayScanResult(data) {
  const isOwned = data.status === 'owned';
  const confidencePct = Math.round((data.confidence || 0.92) * 100);

  activeBook = {
    title: data.title,
    author: data.author || 'Author Unknown',
    genre: data.genre || 'Literature',
    cover_url: data.cover_url || 'https://images.unsplash.com/photo-1544947950-fa07a98d237f?auto=format&fit=crop&w=400&q=80',
    overview: data.overview || '',
    scenario: data.scenario || null,
    confidence: data.confidence || 0.92,
    isOwned: isOwned
  };

  resultCard.style.display = 'block';
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  // Badge
  const badgeEl = document.getElementById('result-badge');
  if (isOwned) {
    badgeEl.className = 'result-badge badge-owned';
    badgeEl.innerHTML = '✓ Already In Your Database (உங்களிடம் உள்ளது)';
  } else {
    badgeEl.className = 'result-badge badge-new';
    badgeEl.innerHTML = '★ New Book - Not In Database (புதிய புத்தகம்)';
  }

  // Meta
  document.getElementById('result-title').textContent = activeBook.title;
  document.getElementById('result-author').textContent = activeBook.author;
  document.getElementById('result-genre').textContent = activeBook.genre;
  document.getElementById('result-cover').src = activeBook.cover_url;
  document.getElementById('result-conf-text').textContent = `${confidencePct}% Match`;
  document.getElementById('result-conf-fill').style.width = `${confidencePct}%`;

  // Action buttons
  const saveBtn = document.getElementById('save-book-btn');
  if (isOwned) {
    saveBtn.innerHTML = '✓ In Your Library';
    saveBtn.style.opacity = '0.7';
    saveBtn.disabled = true;
  } else {
    saveBtn.innerHTML = '+ Add to My Library';
    saveBtn.style.opacity = '1';
    saveBtn.disabled = false;
    saveBtn.onclick = () => saveActiveBook();
  }

  document.getElementById('open-ai-chat-btn').onclick = () => openChatForBook(activeBook);
}

async function saveActiveBook() {
  if (!activeBook) return;

  try {
    const res = await fetch('/api/library/add', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer default_user'
      },
      body: JSON.stringify(activeBook)
    });

    if (res.ok) {
      showToast(`🎉 '${activeBook.title}' added to your library!`);
      activeBook.isOwned = true;
      const saveBtn = document.getElementById('save-book-btn');
      saveBtn.innerHTML = '✓ In Your Library';
      saveBtn.disabled = true;
      const badgeEl = document.getElementById('result-badge');
      badgeEl.className = 'result-badge badge-owned';
      badgeEl.innerHTML = '✓ Already In Your Database';
      loadLibrary();
      loadProfile();
    }
  } catch (err) {
    showToast('⚠️ Error saving to library');
  }
}

// =========================================================
// 🤖 MINI CHATBOT & STORYTELLER DRAWER
// =========================================================
function openChatForBook(book) {
  if (!book) return;
  activeBook = book;
  chatHistory = [];

  // Update Drawer Book Banner
  document.getElementById('drawer-book-title').textContent = book.title;
  const genreTag = document.getElementById('drawer-genre-tag');
  genreTag.textContent = book.genre || 'Fiction';

  const scenarioDesc = document.getElementById('drawer-scenario-desc');
  if (book.scenario && book.scenario.story_narrative) {
    scenarioDesc.textContent = book.scenario.story_narrative;
  } else {
    scenarioDesc.textContent = book.overview || `A gripping narrative by ${book.author}. Tap below to discover the scenario, story, and takeaways!`;
  }

  // Clear messages & inject initial greeting
  const messagesList = document.getElementById('chat-messages-list');
  messagesList.innerHTML = `
    <div class="msg-row ai">
      <div class="msg-avatar avatar-ai">AI</div>
      <div class="msg-bubble">
        வணக்கம்! நான் உங்கள் <strong>BookNest AI</strong> உதவியாளர். 
        <br><br>
        <strong>'${book.title}'</strong> புத்தகத்தின் பின்னணி, கதைச் சுருக்கம் (scenario & story) அல்லது முக்கியமான கருத்துக்களை என்னிடம் கேட்கலாம்!
      </div>
    </div>
  `;

  chatDrawer.classList.add('open');
}

async function sendMessage() {
  const inputEl = document.getElementById('chat-input');
  const text = inputEl.value.trim();
  if (!text || !activeBook) return;

  inputEl.value = '';

  const messagesList = document.getElementById('chat-messages-list');

  // Append user bubble
  const userRow = document.createElement('div');
  userRow.className = 'msg-row user';
  userRow.innerHTML = `
    <div class="msg-avatar avatar-user">Me</div>
    <div class="msg-bubble">${escapeHtml(text)}</div>
  `;
  messagesList.appendChild(userRow);
  messagesList.scrollTop = messagesList.scrollHeight;

  // Append typing bubble
  const typingRow = document.createElement('div');
  typingRow.className = 'msg-row ai';
  typingRow.innerHTML = `
    <div class="msg-avatar avatar-ai">AI</div>
    <div class="msg-bubble"><em>Thinking... 💭</em></div>
  `;
  messagesList.appendChild(typingRow);
  messagesList.scrollTop = messagesList.scrollHeight;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        book_title: activeBook.title,
        book_author: activeBook.author,
        book_description: activeBook.overview || '',
        book_genre: activeBook.genre || '',
        user_message: text,
        history: chatHistory
      })
    });

    const data = await res.json();
    const replyText = data.reply || 'Sorry, I could not generate a response.';

    typingRow.querySelector('.msg-bubble').innerHTML = formatMarkdown(replyText);
    messagesList.scrollTop = messagesList.scrollHeight;

    chatHistory.push({ sender: 'user', text: text });
    chatHistory.push({ sender: 'assistant', text: replyText });

  } catch (err) {
    typingRow.querySelector('.msg-bubble').textContent = '⚠️ Network error. Please try again.';
  }
}

// =========================================================
// 📚 LIBRARY / DATABASE VIEW
// =========================================================
async function loadLibrary() {
  const grid = document.getElementById('library-books-grid');
  if (!grid) return;

  try {
    const res = await fetch('/api/library', {
      headers: { 'Authorization': 'Bearer default_user' }
    });
    libraryBooks = await res.json();
    renderLibrary(libraryBooks);
  } catch (err) {
    console.error('Error loading library:', err);
  }
}

function renderLibrary(books) {
  const grid = document.getElementById('library-books-grid');
  const countEl = document.getElementById('library-count');
  if (countEl) countEl.textContent = `${books.length} Books`;

  if (!books || books.length === 0) {
    grid.innerHTML = `
      <div style="text-align: center; padding: 40px 20px; color: var(--text-dim);">
        <div style="font-size: 2.5rem; margin-bottom: 8px;">📖</div>
        <h3>No books in your database yet</h3>
        <p style="font-size: 0.85rem; margin-top: 4px;">Tap the Scan button to scan your first book!</p>
      </div>
    `;
    return;
  }

  grid.innerHTML = books.map((b) => `
    <div class="library-item" data-title="${escapeHtml(b.title)}">
      <img src="${b.cover_url || 'https://images.unsplash.com/photo-1544947950-fa07a98d237f?auto=format&fit=crop&w=200&q=80'}" class="lib-cover" alt="cover">
      <div class="lib-meta">
        <div>
          <div class="lib-title">${escapeHtml(b.title)}</div>
          <div class="lib-author">${escapeHtml(b.author || 'Author Unknown')}</div>
        </div>
        <div class="lib-footer">
          <span class="status-tag">${escapeHtml(b.status || 'Reading')}</span>
          <div style="display: flex; gap: 8px;">
            <button class="icon-btn" style="width: 28px; height: 28px;" onclick="openChatForSavedBook('${escapeHtml(b.title)}')">
              💬
            </button>
            <button class="del-btn" onclick="deleteBook('${escapeHtml(b.title)}')">
              ✕
            </button>
          </div>
        </div>
      </div>
    </div>
  `).join('');
}

function filterLibrary() {
  const query = document.getElementById('library-search').value.toLowerCase();
  const filtered = libraryBooks.filter((b) =>
    b.title.toLowerCase().includes(query) || (b.author && b.author.toLowerCase().includes(query))
  );
  renderLibrary(filtered);
}

function openChatForSavedBook(title) {
  const found = libraryBooks.find((b) => b.title === title);
  if (found) {
    openChatForBook(found);
  }
}

async function deleteBook(title) {
  if (!confirm(`Remove '${title}' from your library?`)) return;

  try {
    const res = await fetch(`/api/library/remove?title=${encodeURIComponent(title)}`, {
      method: 'DELETE',
      headers: { 'Authorization': 'Bearer default_user' }
    });
    if (res.ok) {
      showToast(`Removed '${title}'`);
      loadLibrary();
      loadProfile();
    }
  } catch (err) {
    showToast('⚠️ Error removing book');
  }
}

// =========================================================
// 👤 USER PROFILE VIEW
// =========================================================
async function loadProfile() {
  try {
    const res = await fetch('/api/profile', {
      headers: { 'Authorization': 'Bearer default_user' }
    });
    const profile = await res.json();

    document.getElementById('profile-name').textContent = profile.name || 'Brother';
    document.getElementById('profile-email').textContent = profile.email || 'brother@bookai.local';
    document.getElementById('stat-total-books').textContent = profile.total_books || 0;
    document.getElementById('stat-total-scans').textContent = profile.total_scans || 0;
    document.getElementById('stat-goal').textContent = `${profile.total_books || 0} / ${profile.reading_goal || 25}`;
    document.getElementById('backend-mode-badge').textContent = profile.backend_mode || 'SQLite Persistent';
  } catch (err) {
    console.error('Profile load error:', err);
  }
}

// =========================================================
// 📲 PHONE CONNECT & QR CODE GENERATOR
// =========================================================
async function loadNetworkInfo() {
  try {
    const res = await fetch('/api/network-info');
    const info = await res.json();
    const phoneUrl = info.mobile_url || window.location.href;

    document.getElementById('connect-url-display').textContent = phoneUrl;

    // Render QR Code using Google Chart or QR API SVG
    const qrContainer = document.getElementById('qrcode-box');
    qrContainer.innerHTML = `
      <img src="https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${encodeURIComponent(phoneUrl)}&margin=4" 
           alt="Scan with phone" style="width: 100%; height: 100%; border-radius: 8px;">
    `;
  } catch (err) {
    console.error('Network info error:', err);
  }
}

// =========================================================
// 🛠 UTILITIES
// =========================================================
function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  setTimeout(() => toastEl.classList.remove('show'), 2600);
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function formatMarkdown(text) {
  if (!text) return '';
  let formatted = escapeHtml(text);
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
  formatted = formatted.replace(/\n/g, '<br>');
  return formatted;
}

function registerPWA() {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').catch(() => {});
  }
}
