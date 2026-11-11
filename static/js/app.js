// =========================================================
// BookNest AI — Client Controller (Clean White 3D Edition)
// =========================================================

let currentStream = null;
let activeBook = null;
let selectedModalBook = null;
let chatHistory = [];
let libraryBooks = [];
let currentCategoryFilter = 'ALL';
let currentStatusFilter = 'ALL';
let facingMode = 'environment';

// Current Authenticated User (saved in localStorage)
let currentUser = {
  uid: 'default_user',
  name: 'Brother Reader',
  email: 'brother@bookai.local',
  avatar: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=240&q=80',
  reading_goal: 25
};

// DOM Elements
const videoEl = document.getElementById('camera-video');
const shutterBtn = document.getElementById('shutter-btn');
const photoInput = document.getElementById('photo-input');
const uploadBtn = document.getElementById('upload-btn');
const switchCamBtn = document.getElementById('switch-cam-btn');
const resultCard = document.getElementById('scan-result-card');
const scanningOverlay = document.getElementById('scanning-overlay');
const chatDrawer = document.getElementById('chat-drawer');
const toastEl = document.getElementById('toast');
const bookModal = document.getElementById('book-modal');
const authModal = document.getElementById('auth-modal');
const themeToggleBtn = document.getElementById('theme-toggle-btn');
const avatarFileInput = document.getElementById('avatar-file-input');

// =========================================================
// 🚀 INITIALIZATION
// =========================================================
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  loadStoredUser();
  initNavigation();
  initAuthUI();
  initModals();
  initCategoryFilters();
  initDashboardEvents();
  initTimerEvents();
  loadLibrary();
  loadProfile();
  loadNetworkInfo();
  registerPWA();
  checkFirstTimeUser();
  initDailyReminderNotification();

  // Shutter action
  if (shutterBtn) shutterBtn.addEventListener('click', captureAndScan);

  // File upload fallback
  if (uploadBtn) uploadBtn.addEventListener('click', () => photoInput.click());
  if (photoInput) photoInput.addEventListener('change', handleFileSelected);

  // Switch camera (single clean flip button)
  if (switchCamBtn) switchCamBtn.addEventListener('click', toggleCamera);

  // Flash / Torch toggle
  const flashToggleBtn = document.getElementById('flash-toggle-btn');
  if (flashToggleBtn) flashToggleBtn.addEventListener('click', toggleTorch);

  // Theme toggle
  if (themeToggleBtn) themeToggleBtn.addEventListener('click', toggleTheme);
  const profileThemeBtn = document.getElementById('profile-theme-toggle-btn');
  if (profileThemeBtn) profileThemeBtn.addEventListener('click', toggleTheme);

  // Chat close
  document.getElementById('close-chat-btn')?.addEventListener('click', () => {
    chatDrawer.classList.remove('open');
  });

  // Chat send
  document.getElementById('chat-form')?.addEventListener('submit', (e) => {
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

  // Result card status chip clicks
  document.querySelectorAll('.status-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.status-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
    });
  });

  // Save profile settings
  document.getElementById('save-profile-btn')?.addEventListener('click', saveProfileSettings);
});

// =========================================================
// 🌓 THEME CONTROLLER (CLEAN WHITE 3D / DARK)
// =========================================================
function initTheme() {
  const savedTheme = localStorage.getItem('booknest_theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcon(savedTheme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'light';
  const newTheme = current === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('booknest_theme', newTheme);
  updateThemeIcon(newTheme);
  showToast(newTheme === 'light' ? '☀️ Clean White Theme Active' : '🌙 Dark Luxury Mode Active');
}

function updateThemeIcon(theme) {
  const icon = document.getElementById('theme-icon');
  if (icon) icon.textContent = theme === 'light' ? '🌙' : '☀️';
  const profileThemeName = document.getElementById('profile-theme-name');
  if (profileThemeName) {
    profileThemeName.textContent = theme === 'light' ? 'Clean White Porcelain ☀️' : 'Dark Luxury Midnight 🌙';
  }
}

// =========================================================
// 🧭 TAB NAVIGATION
// =========================================================
function switchToTab(targetTab) {
  const navItems = document.querySelectorAll('.nav-item');
  const tabs = document.querySelectorAll('.tab-content');

  navItems.forEach((n) => {
    if (n.getAttribute('data-tab') === targetTab) n.classList.add('active');
    else n.classList.remove('active');
  });

  tabs.forEach((t) => {
    if (t.id === targetTab) t.classList.add('active');
    else t.classList.remove('active');
  });

  if (targetTab === 'tab-scan') {
    if (!currentStream) initCamera();
  } else {
    stopCamera();
  }

  if (targetTab === 'tab-dashboard') updateDashboardView();
  if (targetTab === 'tab-timer') initReadingTimer();
  if (targetTab === 'tab-library') loadLibrary();
  if (targetTab === 'tab-profile') loadProfile();
}

function initNavigation() {
  const navItems = document.querySelectorAll('.nav-item');

  navItems.forEach((item) => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const targetTab = item.getAttribute('data-tab');
      switchToTab(targetTab);
    });
  });
}

// =========================================================
// 👤 USER PROFILE & GOOGLE AUTH
// =========================================================
function loadStoredUser() {
  const saved = localStorage.getItem('booknest_user');
  if (saved) {
    try {
      currentUser = { ...currentUser, ...JSON.parse(saved) };
    } catch (e) {
      console.warn('Could not parse stored user');
    }
  }
  updateUserUI();
}

function checkFirstTimeUser() {
  const isAuth = localStorage.getItem('booknest_user_authenticated');
  if (!isAuth) {
    setTimeout(() => {
      if (authModal && !authModal.open) {
        authModal.showModal();
      }
    }, 700);
  }
}

function updateUserUI() {
  const headerAvatar = document.getElementById('header-avatar');
  const headerName = document.getElementById('header-user-name');
  const profileAvatar = document.getElementById('profile-avatar-img');
  const profileName = document.getElementById('profile-name');
  const profileEmail = document.getElementById('profile-email');
  const editName = document.getElementById('edit-profile-name');
  const editGoal = document.getElementById('edit-reading-goal');

  if (headerAvatar) headerAvatar.src = currentUser.avatar;
  if (headerName) headerName.textContent = currentUser.name.split(' ')[0] || 'Reader';
  if (profileAvatar) profileAvatar.src = currentUser.avatar;
  if (profileName) profileName.textContent = currentUser.name;
  if (profileEmail) profileEmail.textContent = currentUser.email;
  if (editName) editName.value = currentUser.name;
  if (editGoal) editGoal.value = currentUser.reading_goal || 25;
}

function initAuthUI() {
  const authHeaderBtn = document.getElementById('auth-header-btn');
  const googleLoginBtn = document.getElementById('google-login-btn');
  const instantGoogleBtn = document.getElementById('instant-google-btn');
  const quickGoogleUserBtn = document.getElementById('google-quick-user-btn');
  const closeAuthModal = document.getElementById('close-auth-modal');
  const changeAvatarBtn = document.getElementById('change-avatar-btn');
  const uploadCustomAvatarBtn = document.getElementById('upload-custom-avatar-btn');

  // Open Auth Modal
  authHeaderBtn?.addEventListener('click', () => authModal.showModal());
  googleLoginBtn?.addEventListener('click', () => authModal.showModal());

  // Close Auth Modal (mark as acknowledged so it won't re-pop)
  closeAuthModal?.addEventListener('click', () => {
    localStorage.setItem('booknest_user_authenticated', 'true');
    authModal.close();
  });

  // Direct 1-Click Google Sign-In with apsiva69@gmail.com (Sivanesan Balu)
  quickGoogleUserBtn?.addEventListener('click', () => {
    performGoogleLogin(
      'Sivanesan Balu',
      'apsiva69@gmail.com',
      'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=240&q=80'
    );
  });

  // Switch / Custom Google Login prompt
  instantGoogleBtn?.addEventListener('click', handleGoogleSignIn);

  // Avatar presets click
  document.querySelectorAll('.avatar-choice').forEach(choice => {
    choice.addEventListener('click', () => {
      const newAvatar = choice.getAttribute('data-avatar');
      applyNewAvatar(newAvatar);
    });
  });

  // Avatar file upload triggers
  changeAvatarBtn?.addEventListener('click', () => avatarFileInput.click());
  uploadCustomAvatarBtn?.addEventListener('click', () => avatarFileInput.click());

  avatarFileInput?.addEventListener('change', handleAvatarFileUpload);
}

async function performGoogleLogin(name, email, avatar) {
  const uid = 'google_' + btoa(email.toLowerCase()).substring(0, 16);
  const payload = {
    uid: uid,
    name: name,
    email: email,
    avatar: avatar || `https://api.dicebear.com/7.x/notionists/svg?seed=${encodeURIComponent(name)}&backgroundColor=e0e7ff`
  };

  try {
    const res = await fetch('/api/auth/google', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    currentUser = { ...currentUser, ...payload };
    localStorage.setItem('booknest_user', JSON.stringify(currentUser));
    localStorage.setItem('booknest_user_authenticated', 'true');
    updateUserUI();
    authModal.close();
    showToast(`✅ Welcome, ${currentUser.name}! Signed in with Google`);
    loadLibrary();
    loadProfile();
  } catch (err) {
    showToast('⚠️ Could not complete Google sign-in');
  }
}

async function handleGoogleSignIn() {
  const promptName = prompt('Enter your Google Account Name:', currentUser.name !== 'Brother Reader' ? currentUser.name : 'Sivanesan Balu');
  if (!promptName) return;

  const email = promptName.toLowerCase().replace(/\s+/g, '.') + '@gmail.com';
  const googleAvatar = `https://api.dicebear.com/7.x/notionists/svg?seed=${encodeURIComponent(promptName)}&backgroundColor=e0e7ff`;

  await performGoogleLogin(promptName, email, googleAvatar);
}

async function applyNewAvatar(avatarUrl) {
  currentUser.avatar = avatarUrl;
  localStorage.setItem('booknest_user', JSON.stringify(currentUser));
  updateUserUI();
  authModal.close();
  showToast('✨ Profile Avatar Updated!');

  try {
    await fetch('/api/profile', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${currentUser.uid}`
      },
      body: JSON.stringify({ avatar: avatarUrl })
    });
  } catch (e) {
    console.error('Failed to mirror avatar update to server', e);
  }
}

async function handleAvatarFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  showToast('Uploading photo...');
  try {
    const res = await fetch('/api/profile/avatar', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${currentUser.uid}` },
      body: formData
    });
    const data = await res.json();
    if (data.avatar_url) {
      applyNewAvatar(data.avatar_url);
    }
  } catch (err) {
    // Fallback to local FileReader base64
    const reader = new FileReader();
    reader.onload = (event) => applyNewAvatar(event.target.result);
    reader.readAsDataURL(file);
  }
}

async function saveProfileSettings() {
  const name = document.getElementById('edit-profile-name').value.trim();
  const goal = parseInt(document.getElementById('edit-reading-goal').value) || 25;

  if (name) currentUser.name = name;
  currentUser.reading_goal = goal;
  localStorage.setItem('booknest_user', JSON.stringify(currentUser));
  updateUserUI();

  try {
    await fetch('/api/profile', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${currentUser.uid}`
      },
      body: JSON.stringify({ name: currentUser.name, reading_goal: goal })
    });
    showToast('💾 Profile settings saved!');
    loadProfile();
  } catch (e) {
    showToast('⚠️ Error saving settings');
  }
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
  showToast(`📷 Switched camera`);
}

let isTorchActive = false;
async function toggleTorch() {
  if (!currentStream) {
    showToast('⚠️ Camera stream is not active');
    return;
  }
  const track = currentStream.getVideoTracks()[0];
  if (!track) {
    showToast('⚠️ Camera track not found');
    return;
  }

  try {
    isTorchActive = !isTorchActive;
    await track.applyConstraints({
      advanced: [{ torch: isTorchActive }]
    });

    const flashBtn = document.getElementById('flash-toggle-btn');
    const flashLabel = document.getElementById('flash-btn-label');
    if (flashBtn) {
      flashBtn.classList.toggle('active', isTorchActive);
    }
    if (flashLabel) {
      flashLabel.textContent = isTorchActive ? 'Flash ON' : 'Flash';
    }
    showToast(isTorchActive ? '⚡ Flash ON' : '⚡ Flash OFF');
  } catch (err) {
    console.warn('Torch constraint error:', err);
    isTorchActive = false;
    showToast('💡 Flash not supported on this lens');
  }
}

// =========================================================
// 🔍 3D SCANNING & IDENTIFICATION
// =========================================================
async function captureAndScan() {
  if (!videoEl || !videoEl.videoWidth) {
    photoInput.click();
    return;
  }

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
  // Show 3D Holographic Scanning Overlay
  showScanningOverlay(true);
  shutterBtn.style.pointerEvents = 'none';
  shutterBtn.style.opacity = '0.5';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/scan', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${currentUser.uid}`
      },
      body: formData
    });

    const data = await res.json();
    console.log('SCAN RESULT:', data);

    if (data.status === 'not_found' || !data.title) {
      showToast('❌ Could not recognize book. Try a clearer angle.');
    } else {
      displayScanResult(data);
    }
  } catch (err) {
    console.error('Scan error:', err);
    showToast('⚠️ Network or server error during scan');
  } finally {
    showScanningOverlay(false);
    shutterBtn.style.pointerEvents = 'auto';
    shutterBtn.style.opacity = '1';
  }
}

function showScanningOverlay(show) {
  if (!scanningOverlay) return;
  if (show) {
    scanningOverlay.style.display = 'flex';
    const stepText = document.getElementById('scan-step-text');
    const stepSub = document.getElementById('scan-step-sub');
    const fill = document.getElementById('scan-progress-fill');

    fill.style.width = '20%';
    stepText.textContent = '📸 Capturing Cover Frame...';
    stepSub.textContent = 'Optimizing image clarity & lighting';

    setTimeout(() => {
      fill.style.width = '55%';
      stepText.textContent = '⚡ Groq Neural Vision AI...';
      stepSub.textContent = 'Recognizing title, typography & art';
    }, 700);

    setTimeout(() => {
      fill.style.width = '85%';
      stepText.textContent = '📚 Matching Database & Scenario...';
      stepSub.textContent = 'Fetching author metadata & story synopsis';
    }, 1600);

  } else {
    scanningOverlay.style.display = 'none';
  }
}

// =========================================================
// 🎯 DISPLAY SCAN RESULT & CATEGORY ASSIGNMENT
// =========================================================
function displayScanResult(data) {
  // Check against locally loaded library books as well as backend status
  const existingBook = libraryBooks.find(b => {
    if (!b || !b.title) return false;
    const t1 = b.title.toLowerCase().trim();
    const t2 = (data.title || '').toLowerCase().trim();
    return t1 === t2 || (t1.length > 5 && t2.length > 5 && (t1.includes(t2) || t2.includes(t1)));
  });

  const isOwned = data.status === 'owned' || !!existingBook;
  const confidencePct = Math.round((data.confidence || 0.95) * 100);

  // Auto detect likely category
  let defaultGenre = data.genre || (existingBook ? existingBook.genre : 'Fiction 📚');
  const rawGenre = (data.genre || '').toLowerCase();
  if (rawGenre.includes('comic') || rawGenre.includes('graphic')) defaultGenre = 'Comics 💥';
  else if (rawGenre.includes('manga') || rawGenre.includes('anime')) defaultGenre = 'Manga 🎌';
  else if (rawGenre.includes('self') || rawGenre.includes('habit') || rawGenre.includes('mind')) defaultGenre = 'Self-Help 💡';
  else if (rawGenre.includes('tech') || rawGenre.includes('program') || rawGenre.includes('comput')) defaultGenre = 'Technology & Code 💻';
  else if (rawGenre.includes('sci-fi') || rawGenre.includes('fantasy')) defaultGenre = 'Sci-Fi & Fantasy 🚀';
  else if (rawGenre.includes('myster') || rawGenre.includes('crime') || rawGenre.includes('thrill')) defaultGenre = 'Mystery & Thriller 🕵️';
  else if (rawGenre.includes('hist') || rawGenre.includes('bio')) defaultGenre = 'History & Bio 📜';

  activeBook = {
    title: data.title,
    author: data.author || (existingBook ? existingBook.author : 'Author Unknown'),
    genre: defaultGenre,
    cover_url: data.cover_url || (existingBook ? existingBook.cover_url : 'https://images.unsplash.com/photo-1544947950-fa07a98d237f?auto=format&fit=crop&w=400&q=80'),
    overview: data.overview || (existingBook ? existingBook.overview : ''),
    scenario: data.scenario || null,
    confidence: data.confidence || 0.95,
    isOwned: isOwned,
    status: existingBook ? (existingBook.status || 'Reading') : 'Reading'
  };

  resultCard.style.display = 'block';
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  // Badge & Already-In-Library Banner
  const badgeEl = document.getElementById('result-badge');
  const alreadyBanner = document.getElementById('already-in-lib-banner');
  const alreadyDesc = document.getElementById('already-lib-desc');
  const jumpBtn = document.getElementById('jump-to-lib-btn');
  const saveBtn = document.getElementById('save-book-btn');

  if (isOwned) {
    const curStatus = existingBook ? (existingBook.status || 'Reading') : 'Saved';
    badgeEl.className = 'result-badge badge-owned';
    badgeEl.innerHTML = '✓ Already In Your Library';

    if (alreadyBanner) {
      alreadyBanner.style.display = 'flex';
      if (alreadyDesc) {
        alreadyDesc.textContent = `You already have '${activeBook.title}' in your collection! Status: ${curStatus}.`;
      }
    }

    if (jumpBtn) {
      jumpBtn.onclick = () => {
        switchToTab('tab-library');
        openBookModal(existingBook || activeBook);
      };
    }

    saveBtn.innerHTML = '📖 View in Library';
    saveBtn.disabled = false;
    saveBtn.style.opacity = '1';
    saveBtn.onclick = () => {
      switchToTab('tab-library');
      openBookModal(existingBook || activeBook);
    };

    // User requested explicit notification: "told me its already in your library like a message"
    showToast(`📖 Already in your Library! '${activeBook.title}' is in your collection.`);
  } else {
    badgeEl.className = 'result-badge badge-new';
    badgeEl.innerHTML = '★ New Discovery';

    if (alreadyBanner) {
      alreadyBanner.style.display = 'none';
    }

    saveBtn.innerHTML = '+ Save to My Library';
    saveBtn.disabled = false;
    saveBtn.style.opacity = '1';
    saveBtn.onclick = () => saveActiveBook();

    showToast(`✨ Book Found: '${activeBook.title}'`);
  }

  // Meta
  document.getElementById('result-title').textContent = activeBook.title;
  document.getElementById('result-author').textContent = activeBook.author;
  document.getElementById('result-cover').src = activeBook.cover_url;
  document.getElementById('result-conf-text').textContent = `${confidencePct}% Match`;
  document.getElementById('result-conf-fill').style.width = `${confidencePct}%`;

  // Pre-select category in dropdown
  const catSelect = document.getElementById('result-category-select');
  if (catSelect) {
    for (let opt of catSelect.options) {
      if (opt.value === defaultGenre || opt.text.includes(defaultGenre)) {
        catSelect.value = opt.value;
        break;
      }
    }
  }

  document.getElementById('open-ai-chat-btn').onclick = () => openChatForBook(activeBook);
}

async function saveActiveBook() {
  if (!activeBook) return;

  const catSelect = document.getElementById('result-category-select');
  if (catSelect) activeBook.genre = catSelect.value;

  const activeChip = document.querySelector('.status-chip.active');
  if (activeChip) activeBook.status = activeChip.getAttribute('data-status');

  try {
    const res = await fetch('/api/library/add', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${currentUser.uid}`
      },
      body: JSON.stringify(activeBook)
    });

    if (res.ok) {
      if (activeBook.status === 'Completed') {
        triggerConfettiCelebration(activeBook.title);
      } else {
        showToast(`🎉 '${activeBook.title}' added to your library!`);
      }

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
// 📚 LIBRARY & CATEGORIES VIEW
// =========================================================
let customTopics = [];
try {
  customTopics = JSON.parse(localStorage.getItem('booknest_custom_topics') || '[]');
} catch (e) {
  customTopics = [];
}

function initCategoryFilters() {
  const chipsBar = document.getElementById('category-chips-bar');
  const openAddBtn = document.getElementById('open-add-topic-btn');
  const addBox = document.getElementById('add-topic-container');
  const newTopicInput = document.getElementById('new-topic-input');
  const saveTickBtn = document.getElementById('save-topic-tick-btn');
  const cancelTopicBtn = document.getElementById('cancel-topic-btn');

  // 1. Initial click handler for default chips
  function setupChipEvents(chip) {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.cat-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      currentCategoryFilter = chip.getAttribute('data-category');
      filterLibrary();
    });
  }

  document.querySelectorAll('.cat-chip:not(.add-topic-chip)').forEach(setupChipEvents);

  // 2. Helper to create and inject a custom topic chip
  function renderCustomTopicChip(topicName) {
    if (!topicName || !chipsBar || !openAddBtn) return null;
    
    // Check if chip already exists
    const existing = chipsBar.querySelector(`[data-category="${CSS.escape(topicName)}"]`);
    if (existing) return existing;

    const chip = document.createElement('button');
    chip.className = 'cat-chip custom-cat-chip';
    chip.setAttribute('data-category', topicName);
    chip.textContent = topicName;
    setupChipEvents(chip);

    // Insert just before the + Add Topic button
    chipsBar.insertBefore(chip, openAddBtn);

    // Also populate category dropdowns in book modals
    const resultCatSelect = document.getElementById('result-category-select');
    const modalCatSelect = document.getElementById('modal-category-select');
    [resultCatSelect, modalCatSelect].forEach(sel => {
      if (sel && !sel.querySelector(`option[value="${CSS.escape(topicName)}"]`)) {
        const opt = document.createElement('option');
        opt.value = topicName;
        opt.textContent = topicName;
        sel.appendChild(opt);
      }
    });

    return chip;
  }

  // Render stored custom topics
  customTopics.forEach(t => renderCustomTopicChip(t));

  // 3. Open add topic input
  openAddBtn?.addEventListener('click', () => {
    if (!addBox) return;
    const isHidden = addBox.style.display === 'none' || !addBox.style.display;
    addBox.style.display = isHidden ? 'flex' : 'none';
    if (isHidden && newTopicInput) {
      newTopicInput.focus();
    }
  });

  // 4. Cancel add topic
  cancelTopicBtn?.addEventListener('click', () => {
    if (addBox) addBox.style.display = 'none';
    if (newTopicInput) newTopicInput.value = '';
  });

  // 5. Save Topic Action via Tick Button (✓) or Enter
  function saveNewTopic() {
    if (!newTopicInput) return;
    const val = newTopicInput.value.trim();
    if (!val) {
      showToast('⚠️ Please enter a topic name');
      return;
    }

    if (!customTopics.includes(val)) {
      customTopics.push(val);
      localStorage.setItem('booknest_custom_topics', JSON.stringify(customTopics));
    }

    const newChip = renderCustomTopicChip(val);
    if (newChip) {
      newChip.click(); // Select and filter immediately!
    }

    newTopicInput.value = '';
    if (addBox) addBox.style.display = 'none';
    showToast(`✓ Topic '${val}' saved and selected!`);
  }

  saveTickBtn?.addEventListener('click', saveNewTopic);
  newTopicInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      saveNewTopic();
    } else if (e.key === 'Escape') {
      if (addBox) addBox.style.display = 'none';
    }
  });

  // Secondary status filter buttons
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentStatusFilter = btn.getAttribute('data-status-filter');
      filterLibrary();
    });
  });
}

async function loadLibrary() {
  const grid = document.getElementById('library-books-grid');
  if (!grid) return;

  try {
    const res = await fetch('/api/library', {
      headers: { 'Authorization': `Bearer ${currentUser.uid}` }
    });
    libraryBooks = await res.json();
    updateDashboardStats(libraryBooks);
    updateDashboardView();
    filterLibrary();
  } catch (err) {
    console.error('Error loading library:', err);
  }
}

function updateDashboardStats(books) {
  const total = books.length;
  const completed = books.filter(b => (b.status || '').toLowerCase().includes('complet') || (b.status || '').toLowerCase().includes('finish')).length;
  const reading = books.filter(b => (b.status || '').toLowerCase().includes('read') && !(b.status || '').toLowerCase().includes('complet')).length;
  const goal = currentUser.reading_goal || 25;
  const goalPct = Math.min(100, Math.round((completed / goal) * 100));

  const totalEl = document.getElementById('metric-total');
  const compEl = document.getElementById('metric-completed');
  const readEl = document.getElementById('metric-reading');
  const goalPctEl = document.getElementById('metric-goal-pct');
  const goalTextEl = document.getElementById('goal-progress-text');
  const goalFillEl = document.getElementById('goal-progress-fill');
  const countBadge = document.getElementById('library-count-badge');

  if (totalEl) totalEl.textContent = total;
  if (compEl) compEl.textContent = completed;
  if (readEl) readEl.textContent = reading;
  if (goalPctEl) goalPctEl.textContent = `${goalPct}%`;
  if (goalTextEl) goalTextEl.textContent = `${completed} / ${goal} Finished (${goalPct}%)`;
  if (goalFillEl) goalFillEl.style.width = `${goalPct}%`;
  if (countBadge) countBadge.textContent = `${total} Books`;
}

let displayedBookLimit = 24;
let activeFilteredBooks = [];

function filterLibrary() {
  const searchInput = document.getElementById('library-search');
  const query = searchInput ? searchInput.value.toLowerCase().trim() : '';
  displayedBookLimit = 24; // Reset to first page on search or filter change

  activeFilteredBooks = libraryBooks.filter(b => {
    // 1. Text Search Filter
    const matchesQuery = !query || 
      b.title.toLowerCase().includes(query) || 
      (b.author && b.author.toLowerCase().includes(query)) ||
      (b.genre && b.genre.toLowerCase().includes(query));

    // 2. Category Filter
    let matchesCat = true;
    if (currentCategoryFilter && currentCategoryFilter.toUpperCase() !== 'ALL') {
      const bCat = (b.genre || '').toLowerCase();
      const targetCat = currentCategoryFilter.toLowerCase().trim();
      const firstWord = targetCat.split(/\s+/)[0];
      matchesCat = bCat.includes(targetCat) || 
                   bCat.includes(firstWord) || 
                   targetCat.includes(bCat) ||
                   (b.title && b.title.toLowerCase().includes(firstWord)) ||
                   (b.overview && b.overview.toLowerCase().includes(firstWord));
    }

    // 3. Status Filter
    let matchesStatus = true;
    if (currentStatusFilter !== 'ALL') {
      const bStatus = (b.status || 'reading').toLowerCase();
      const targetStatus = currentStatusFilter.toLowerCase();
      if (targetStatus === 'completed') matchesStatus = bStatus.includes('complet') || bStatus.includes('finish');
      else if (targetStatus === 'reading') matchesStatus = bStatus.includes('read') && !bStatus.includes('complet');
      else if (targetStatus === 'wishlist') matchesStatus = bStatus.includes('wish') || bStatus.includes('want');
    }

    return matchesQuery && matchesCat && matchesStatus;
  });

  renderLibrary(activeFilteredBooks);
}

function renderLibrary(books) {
  const grid = document.getElementById('library-books-grid');
  if (!grid) return;

  if (!books || books.length === 0) {
    grid.innerHTML = `
      <div style="grid-column: 1 / -1; text-align: center; padding: 48px 20px; background: var(--bg-card); border-radius: 20px; border: 1px solid var(--border-card);">
        <div style="font-size: 3rem; margin-bottom: 10px;">📚</div>
        <h3 style="font-size: 1.1rem; font-weight: 800;">No books found</h3>
        <p style="font-size: 0.82rem; color: var(--text-muted); margin-top: 4px;">
          Try changing your category filter or scan a new book cover!
        </p>
      </div>
    `;
    return;
  }

  const visibleBooks = books.slice(0, displayedBookLimit);

  const cardsHtml = visibleBooks.map((b) => {
    const isCompleted = (b.status || '').toLowerCase().includes('complet') || (b.status || '').toLowerCase().includes('finish');
    const isWishlist = (b.status || '').toLowerCase().includes('wish');
    const statusClass = isCompleted ? 'tag-completed' : (isWishlist ? 'tag-wishlist' : 'tag-reading');
    const statusLabel = isCompleted ? '🎉 Read' : (isWishlist ? '⭐ Wishlist' : '📖 Reading');

    return `
      <div class="library-item 3d-card" onclick="openBookModal('${encodeURIComponent(b.title)}')">
        <div class="lib-cover-wrap">
          <img src="${b.cover_url || 'https://images.unsplash.com/photo-1544947950-fa07a98d237f?auto=format&fit=crop&w=300&q=80'}" class="lib-cover" alt="cover" loading="lazy">
        </div>
        <div class="lib-meta">
          <div>
            <div class="lib-category-tag">${escapeHtml(b.genre || 'General')}</div>
            <div class="lib-title">${escapeHtml(b.title)}</div>
            <div class="lib-author">${escapeHtml(b.author || 'Author Unknown')}</div>
          </div>
          <div class="lib-footer">
            <span class="status-tag ${statusClass}">${statusLabel}</span>
            <span style="font-size: 0.72rem; color: #f59e0b; font-weight: 700;">★ ${b.rating || 5}</span>
          </div>
        </div>
      </div>
    `;
  }).join('');

  let loadMoreHtml = '';
  if (books.length > displayedBookLimit) {
    loadMoreHtml = `
      <div style="grid-column: 1 / -1; text-align: center; margin: 18px 0 10px;">
        <button id="load-more-books-btn" class="btn-secondary 3d-btn-secondary" style="width: 100%; padding: 14px 20px; font-weight: 800; font-size: 0.9rem;" onclick="loadMoreBooks()">
          📖 Load More Books (+24) • Showing ${displayedBookLimit} of ${books.length} Books
        </button>
      </div>
    `;
  }

  grid.innerHTML = cardsHtml + loadMoreHtml;
}

function loadMoreBooks() {
  displayedBookLimit += 24;
  renderLibrary(activeFilteredBooks);
}

// =========================================================
// 📖 3D BOOK READING & DETAIL MODAL (<dialog>)
// =========================================================
function initModals() {
  document.getElementById('close-book-modal')?.addEventListener('click', () => bookModal.close());

  // Close congrats modal
  document.getElementById('close-congrats-modal')?.addEventListener('click', () => {
    document.getElementById('congrats-modal')?.close();
  });
  document.getElementById('congrats-done-btn')?.addEventListener('click', () => {
    document.getElementById('congrats-modal')?.close();
  });

  // Star ratings in modal
  document.querySelectorAll('#modal-star-rating .star').forEach(star => {
    star.addEventListener('click', () => {
      const rating = parseInt(star.getAttribute('data-rating'));
      setModalRating(rating);
      if (selectedModalBook) {
        selectedModalBook.rating = rating;
        updateBookOnServer(selectedModalBook);
      }
    });
  });

  // Modal Status buttons
  document.querySelectorAll('.m-status-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.m-status-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const newStatus = btn.getAttribute('data-m-status');
      if (selectedModalBook) {
        selectedModalBook.status = newStatus;
        if (newStatus === 'Completed') {
          triggerSessionCelebration(selectedModalBook.title, 0, 0, selectedModalBook.total_pages || 300, selectedModalBook.total_pages || 300);
        }
        updateBookOnServer(selectedModalBook);
        loadLibrary();
        loadProfile();
        updateDashboardView();
      }
    });
  });

  // Modal Category select change
  document.getElementById('modal-category-select')?.addEventListener('change', (e) => {
    if (selectedModalBook) {
      selectedModalBook.genre = e.target.value;
      updateBookOnServer(selectedModalBook);
      loadLibrary();
    }
  });

  // Modal Page Progress Bookmark Save
  document.getElementById('modal-save-page-btn')?.addEventListener('click', async () => {
    if (!selectedModalBook) return;
    const curP = parseInt(document.getElementById('modal-current-page-input').value) || 0;
    const totP = parseInt(document.getElementById('modal-total-pages-input').value) || 300;

    selectedModalBook.current_page = curP;
    selectedModalBook.total_pages = totP;

    const pct = Math.min(100, Math.round((curP / totP) * 100));
    const progText = document.getElementById('modal-page-progress-text');
    const progFill = document.getElementById('modal-page-progress-fill');
    if (progText) progText.textContent = `Page ${curP} / ${totP} (${pct}%)`;
    if (progFill) progFill.style.width = `${pct}%`;

    let newStatus = selectedModalBook.status;
    if (curP >= totP && totP > 0) {
      newStatus = 'Completed';
      selectedModalBook.status = 'Completed';
      triggerSessionCelebration(selectedModalBook.title, 0, 0, curP, totP);
    }

    try {
      await fetch('/api/library/update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${currentUser.uid}`
        },
        body: JSON.stringify({
          title: selectedModalBook.title,
          current_page: curP,
          total_pages: totP,
          status: newStatus
        })
      });
      showToast(`🔖 Saved Bookmark: Page ${curP} of ${totP}!`);
      updateDashboardView();
      loadLibrary();
    } catch (e) {
      showToast('⚠️ Could not save page bookmark');
    }
  });

  // Modal Launch Timer with current book
  document.getElementById('modal-launch-timer-btn')?.addEventListener('click', () => {
    if (selectedModalBook) {
      selectTimerBook(selectedModalBook.title);
      bookModal.close();
      switchToTab('tab-timer');
    }
  });

  // Modal chat button
  document.getElementById('modal-chat-btn')?.addEventListener('click', () => {
    bookModal.close();
    if (selectedModalBook) openChatForBook(selectedModalBook);
  });

  // Modal delete button
  document.getElementById('modal-delete-btn')?.addEventListener('click', () => {
    if (selectedModalBook && confirm(`Remove '${selectedModalBook.title}' from your library?`)) {
      deleteBook(selectedModalBook.title);
      bookModal.close();
    }
  });
}

function openBookModal(encodedTitle) {
  const title = decodeURIComponent(encodedTitle);
  const book = libraryBooks.find(b => b.title === title);
  if (!book) return;

  selectedModalBook = book;

  document.getElementById('modal-book-title').textContent = book.title;
  document.getElementById('modal-book-author').textContent = `by ${book.author || 'Author Unknown'}`;
  document.getElementById('modal-book-cover').src = book.cover_url || 'https://images.unsplash.com/photo-1544947950-fa07a98d237f?auto=format&fit=crop&w=400&q=80';
  document.getElementById('modal-book-genre-pill').textContent = book.genre || 'General';

  // Set category dropdown
  const catSelect = document.getElementById('modal-category-select');
  if (catSelect) {
    catSelect.value = book.genre || 'Fiction 📚';
  }

  // Set status button
  document.querySelectorAll('.m-status-btn').forEach(b => {
    b.classList.remove('active');
    if (b.getAttribute('data-m-status') === (book.status || 'Reading')) {
      b.classList.add('active');
    }
  });

  // Set star rating
  setModalRating(book.rating || 5);

  // Set Page Progress & Inputs
  const curPage = book.current_page || 0;
  const totPage = book.total_pages || 300;
  const pct = Math.min(100, Math.round((curPage / totPage) * 100));

  const curInput = document.getElementById('modal-current-page-input');
  const totInput = document.getElementById('modal-total-pages-input');
  const progText = document.getElementById('modal-page-progress-text');
  const progFill = document.getElementById('modal-page-progress-fill');

  if (curInput) curInput.value = curPage;
  if (totInput) totInput.value = totPage;
  if (progText) progText.textContent = `Page ${curPage} / ${totPage} (${pct}%)`;
  if (progFill) progFill.style.width = `${pct}%`;

  // Set scenario / synopsis
  const scenarioBox = document.getElementById('modal-scenario-content');
  if (book.overview || book.scenario) {
    scenarioBox.innerHTML = `
      <div style="font-weight: 700; margin-bottom: 4px;">📖 Premise & Scenario:</div>
      <div>${escapeHtml(book.overview || 'Atmospheric story scenario crafted by AI.')}</div>
    `;
  } else {
    scenarioBox.textContent = `A celebrated ${book.genre || 'book'} by ${book.author}. Tap Chat to explore the story scenario with AI!`;
  }

  bookModal.showModal();
}

function setModalRating(rating) {
  document.querySelectorAll('#modal-star-rating .star').forEach(star => {
    const starRating = parseInt(star.getAttribute('data-rating'));
    if (starRating <= rating) {
      star.classList.add('active');
    } else {
      star.classList.remove('active');
    }
  });
}

async function updateBookOnServer(book) {
  try {
    await fetch('/api/library/update', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${currentUser.uid}`
      },
      body: JSON.stringify({
        title: book.title,
        status: book.status,
        rating: book.rating,
        genre: book.genre,
        current_page: book.current_page || 0,
        total_pages: book.total_pages || 300
      })
    });
    showToast(`✓ Updated '${book.title}'`);
  } catch (e) {
    console.error('Failed to update book on server', e);
  }
}

async function deleteBook(title) {
  try {
    const res = await fetch(`/api/library/remove?title=${encodeURIComponent(title)}`, {
      method: 'DELETE',
      headers: { 'Authorization': `Bearer ${currentUser.uid}` }
    });
    if (res.ok) {
      showToast(`🗑️ Removed '${title}'`);
      loadLibrary();
      loadProfile();
    }
  } catch (err) {
    showToast('⚠️ Error removing book');
  }
}

// =========================================================
// 🎉 CELEBRATION EFFECT (CONFETTI ON BOOK COMPLETION)
// =========================================================
function triggerConfettiCelebration(bookTitle) {
  if (typeof confetti === 'function') {
    confetti({
      particleCount: 120,
      spread: 80,
      origin: { y: 0.6 },
      colors: ['#6366f1', '#10b981', '#06b6d4', '#f59e0b', '#ec4899']
    });
  }
  showToast(`🎉 Congratulations! You completed '${bookTitle}'!`);
}

// =========================================================
// 🤖 MINI CHATBOT & STORYTELLER DRAWER
// =========================================================
function openChatForBook(book) {
  if (!book) return;
  activeBook = book;
  chatHistory = [];

  document.getElementById('drawer-book-title').textContent = book.title;
  document.getElementById('drawer-genre-tag').textContent = book.genre || 'Fiction';

  const scenarioDesc = document.getElementById('drawer-scenario-desc');
  if (book.scenario && book.scenario.story_narrative) {
    scenarioDesc.textContent = book.scenario.story_narrative;
  } else {
    scenarioDesc.textContent = book.overview || `A gripping narrative by ${book.author}. Tap below to discover the scenario, story, and takeaways!`;
  }

  const messagesList = document.getElementById('chat-messages-list');
  messagesList.innerHTML = `
    <div class="msg-row ai">
      <div class="msg-avatar avatar-ai">AI</div>
      <div class="msg-bubble">
        Hello! I am your <strong>BookNest AI</strong> reading companion.
        <br><br>
        Ask me anything about the background, scenario, storyline, or key life takeaways from <strong>'${escapeHtml(book.title)}'</strong>!
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
      headers: { 'Content-Type': 'application/json' },
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
// 👤 PROFILE & DIAGNOSTICS VIEW
// =========================================================
async function loadProfile() {
  try {
    const res = await fetch('/api/profile', {
      headers: { 'Authorization': `Bearer ${currentUser.uid}` }
    });
    const profile = await res.json();

    const totalEl = document.getElementById('stat-total-books');
    const compEl = document.getElementById('stat-completed-books');
    const scansEl = document.getElementById('stat-total-scans');
    const goalEl = document.getElementById('stat-goal');
    const modeBadge = document.getElementById('backend-mode-badge');

    const totalBooks = profile.total_books || libraryBooks.length;
    const completedBooks = profile.completed_books || libraryBooks.filter(b => (b.status || '').toLowerCase().includes('complet')).length;
    const goal = profile.reading_goal || currentUser.reading_goal || 25;

    if (totalEl) totalEl.textContent = totalBooks;
    if (compEl) compEl.textContent = completedBooks;
    if (scansEl) scansEl.textContent = profile.total_scans || 0;
    if (goalEl) goalEl.textContent = `${completedBooks} / ${goal}`;
    if (modeBadge) modeBadge.textContent = profile.backend_mode || 'SQLite Mirror';

  } catch (err) {
    console.error('Error loading profile:', err);
  }
}

// =========================================================
// 📲 NETWORK INFO & QR CODE
// =========================================================
async function loadNetworkInfo() {
  try {
    const res = await fetch('/api/network-info');
    const data = await res.json();
    const targetUrl = data.mobile_url || data.public_url || `http://${data.local_ip}:${data.port}`;

    const urlDisplay = document.getElementById('connect-url-display');
    if (urlDisplay) urlDisplay.textContent = targetUrl;

    const qrBox = document.getElementById('qrcode-box');
    if (qrBox) {
      const qrApiUrl = `https://api.qrserver.com/v1/create-qr-code/?size=180x180&data=${encodeURIComponent(targetUrl)}&margin=10`;
      qrBox.innerHTML = `<img src="${qrApiUrl}" width="180" height="180" alt="Mobile QR Code" style="border-radius: 12px;">`;
    }
  } catch (err) {
    console.error('Error loading network info:', err);
  }
}

// =========================================================
// 🔔 UTILITIES (TOAST & ESCAPE)
// =========================================================
function showToast(msg) {
  if (!toastEl) return;
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  setTimeout(() => toastEl.classList.remove('show'), 3200);
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

function formatMarkdown(text) {
  if (!text) return '';
  let formatted = escapeHtml(text);
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
  formatted = formatted.replace(/\n\n/g, '<br><br>');
  formatted = formatted.replace(/\n/g, '<br>');
  return formatted;
}

function registerPWA() {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').catch(console.warn);
  }
}

// =========================================================
// 🏠 DASHBOARD CONTROLLER & HABIT REMINDERS
// =========================================================
const READING_QUOTES = [
  { quote: '"A reader lives a thousand lives before he dies."', author: 'George R.R. Martin' },
  { quote: '"Today a reader, tomorrow a leader."', author: 'Margaret Fuller' },
  { quote: '"Reading is to the mind what exercise is to the body."', author: 'Joseph Addison' },
  { quote: '"Books are a uniquely portable magic."', author: 'Stephen King' },
  { quote: '"The more that you read, the more things you will know."', author: 'Dr. Seuss' },
  { quote: '"Think before you speak. Read before you think."', author: 'Fran Lebowitz' }
];

function initDashboardEvents() {
  document.getElementById('quick-start-reminder-btn')?.addEventListener('click', () => {
    switchToTab('tab-timer');
  });

  document.getElementById('see-all-reading-btn')?.addEventListener('click', () => {
    switchToTab('tab-library');
  });

  document.getElementById('view-all-wishlist-btn')?.addEventListener('click', () => {
    switchToTab('tab-library');
    const wishChip = document.querySelector('.filter-btn[data-status-filter="Wishlist"]');
    if (wishChip) wishChip.click();
  });

  document.getElementById('dash-browse-wishlist-btn')?.addEventListener('click', () => {
    switchToTab('tab-library');
  });

  document.getElementById('dash-explore-btn')?.addEventListener('click', () => {
    switchToTab('tab-scan');
  });

  document.getElementById('dash-quick-scan-btn')?.addEventListener('click', () => {
    switchToTab('tab-scan');
  });

  document.getElementById('dash-quick-timer-btn')?.addEventListener('click', () => {
    switchToTab('tab-timer');
  });
}

function updateDashboardView() {
  const greetingEl = document.getElementById('dash-greeting-text');
  const quoteEl = document.getElementById('dash-daily-quote');
  const heroAvatar = document.getElementById('dash-hero-avatar');
  const streakText = document.getElementById('dash-streak-text');
  const reminderMsg = document.getElementById('daily-reminder-msg');

  // Greeting
  const firstName = (currentUser.name || 'Reader').split(' ')[0];
  const hour = new Date().getHours();
  const timeGreeting = hour < 12 ? 'Good morning' : (hour < 18 ? 'Good afternoon' : 'Good evening');
  if (greetingEl) greetingEl.textContent = `${timeGreeting}, ${firstName}!`;

  // Hero Avatar
  if (heroAvatar) heroAvatar.src = currentUser.avatar;

  // Streak
  const streak = parseInt(localStorage.getItem('booknest_streak') || '3');
  if (streakText) streakText.textContent = `${streak} Day Streak 🔥`;

  // Quote of the Day
  if (quoteEl) {
    const q = READING_QUOTES[Math.floor(Math.random() * READING_QUOTES.length)];
    quoteEl.textContent = `${q.quote} — ${q.author}`;
  }

  // Aggregate Metrics from libraryBooks
  const total = libraryBooks.length;
  const completed = libraryBooks.filter(b => (b.status || '').toLowerCase().includes('complet') || (b.status || '').toLowerCase().includes('finish')).length;
  const reading = libraryBooks.filter(b => (b.status || '').toLowerCase().includes('read') && !(b.status || '').toLowerCase().includes('complet')).length;
  const totalMinutes = libraryBooks.reduce((sum, b) => sum + (parseInt(b.reading_minutes) || 0), 0);
  const goal = currentUser.reading_goal || 25;
  const goalPct = Math.min(100, Math.round((completed / goal) * 100));

  const dTotal = document.getElementById('dash-total-books');
  const dReading = document.getElementById('dash-reading-books');
  const dCompleted = document.getElementById('dash-completed-books');
  const dMinutes = document.getElementById('dash-reading-mins');
  const dGoalPct = document.getElementById('dash-goal-pct');
  const dGoalFill = document.getElementById('dash-goal-fill');
  const dGoalSub = document.getElementById('dash-goal-sub');

  if (dTotal) dTotal.textContent = total;
  if (dReading) dReading.textContent = reading;
  if (dCompleted) dCompleted.textContent = completed;
  if (dMinutes) dMinutes.textContent = `${totalMinutes}m`;
  if (dGoalPct) dGoalPct.textContent = `${goalPct}%`;
  if (dGoalFill) dGoalFill.style.width = `${goalPct}%`;
  if (dGoalSub) dGoalSub.textContent = `${completed} of ${goal} Books Completed`;

  // Find Active Reading Book
  const activeBook = libraryBooks.find(b => (b.status || '').toLowerCase() === 'reading') || libraryBooks[0];
  const activeReadingCard = document.getElementById('active-reading-card');
  const noReadingState = document.getElementById('no-reading-placeholder');

  if (activeBook) {
    if (activeReadingCard) activeReadingCard.style.display = 'flex';
    if (noReadingState) noReadingState.style.display = 'none';

    const arCover = document.getElementById('ar-cover');
    const arTitle = document.getElementById('ar-title');
    const arAuthor = document.getElementById('ar-author');
    const arGenre = document.getElementById('ar-genre');

    if (arCover) arCover.src = activeBook.cover_url || 'https://images.unsplash.com/photo-1544947950-fa07a98d237f?auto=format&fit=crop&w=200&q=80';
    if (arTitle) arTitle.textContent = activeBook.title;
    if (arAuthor) arAuthor.textContent = `by ${activeBook.author || 'Author Unknown'}`;
    if (arGenre) arGenre.textContent = activeBook.genre || 'Reading';

    const curP = activeBook.current_page || 0;
    const totP = activeBook.total_pages || 300;
    const pct = Math.min(100, Math.round((curP / totP) * 100));

    const arPageText = document.getElementById('ar-page-text');
    const arPct = document.getElementById('ar-progress-pct');
    const arFill = document.getElementById('ar-fill');

    if (arPageText) arPageText.textContent = `Page ${curP} of ${totP}`;
    if (arPct) arPct.textContent = `${pct}%`;
    if (arFill) arFill.style.width = `${pct}%`;

    if (reminderMsg) {
      reminderMsg.textContent = `You're currently reading '${activeBook.title}'! Dive into page ${curP || 1} today to keep your daily streak alive 🚀`;
    }

    const arTimerBtn = document.getElementById('ar-open-timer-btn');
    if (arTimerBtn) {
      arTimerBtn.onclick = () => {
        selectTimerBook(activeBook.title);
        switchToTab('tab-timer');
      };
    }

    const arUpdateBtn = document.getElementById('ar-update-page-btn');
    if (arUpdateBtn) {
      arUpdateBtn.onclick = () => {
        openBookModal(encodeURIComponent(activeBook.title));
      };
    }
  } else {
    if (activeReadingCard) activeReadingCard.style.display = 'none';
    if (noReadingState) noReadingState.style.display = 'block';
    if (reminderMsg) {
      reminderMsg.textContent = `Scan your first book cover or pick from library to start your daily reading journey!`;
    }
  }

  // Populate Wishlist Shelf ("Whistle Book List")
  const wishlistBooks = libraryBooks.filter(b => {
    const s = (b.status || '').toLowerCase();
    return s.includes('wish') || s.includes('want');
  });

  const wishCountBadge = document.getElementById('dash-wishlist-count-badge');
  const wishContainer = document.getElementById('dash-wishlist-container');
  const noWishPlaceholder = document.getElementById('no-wishlist-placeholder');

  if (wishCountBadge) wishCountBadge.textContent = wishlistBooks.length;

  if (wishContainer) {
    if (wishlistBooks.length > 0) {
      wishContainer.style.display = 'flex';
      if (noWishPlaceholder) noWishPlaceholder.style.display = 'none';

      wishContainer.innerHTML = wishlistBooks.map(b => `
        <div class="wishlist-card 3d-card" onclick="openBookModal('${encodeURIComponent(b.title)}')">
          <img src="${b.cover_url || 'https://images.unsplash.com/photo-1544947950-fa07a98d237f?auto=format&fit=crop&w=200&q=80'}" class="wishlist-card-cover" alt="cover">
          <div class="wishlist-card-title">${escapeHtml(b.title)}</div>
          <div class="wishlist-card-author">by ${escapeHtml(b.author || 'Author Unknown')}</div>
          <button type="button" class="wishlist-card-btn" onclick="event.stopPropagation(); startReadingWishlistBook('${encodeURIComponent(b.title)}')">
            ▶ Start Read
          </button>
        </div>
      `).join('');
    } else {
      wishContainer.style.display = 'none';
      if (noWishPlaceholder) noWishPlaceholder.style.display = 'block';
    }
  }
}

async function startReadingWishlistBook(encodedTitle) {
  const title = decodeURIComponent(encodedTitle);
  const book = libraryBooks.find(b => b.title === title);
  if (!book) return;

  book.status = 'Reading';
  await updateBookOnServer(book);
  selectTimerBook(book.title);
  switchToTab('tab-timer');
  showToast(`📖 Now reading '${book.title}'! Let's go!`);
  updateDashboardView();
  loadLibrary();
}

// =========================================================
// ⏱️ READING TIMER ENGINE (PRESETS, STOPWATCH, LOGGING)
// =========================================================
let timerInterval = null;
let timerRunning = false;
let timerTargetDuration = 25 * 60; // 25 min default
let timerElapsedSeconds = 0;
let timerMode = 'countdown'; // 'countdown' or 'stopwatch'
let activeTimerBookTitle = '';

function initTimerEvents() {
  // Preset chips
  document.querySelectorAll('.timer-preset-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.timer-preset-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const mins = parseInt(chip.getAttribute('data-mins'));
      if (mins === 0) {
        timerMode = 'stopwatch';
        timerTargetDuration = 0;
      } else {
        timerMode = 'countdown';
        timerTargetDuration = mins * 60;
      }
      resetTimer();
    });
  });

  // Toggle Start / Pause
  document.getElementById('timer-toggle-btn')?.addEventListener('click', toggleTimer);

  // Finish session
  document.getElementById('timer-finish-btn')?.addEventListener('click', finishTimerSession);

  // Reset
  document.getElementById('timer-reset-btn')?.addEventListener('click', resetTimer);

  // Book select change
  document.getElementById('timer-book-select')?.addEventListener('change', (e) => {
    onTimerBookSelected(e.target.value);
  });
}

function initReadingTimer() {
  const select = document.getElementById('timer-book-select');
  if (!select) return;

  const currentVal = select.value;
  select.innerHTML = '<option value="">-- Choose a Book to Read --</option>';

  libraryBooks.forEach(b => {
    const opt = document.createElement('option');
    opt.value = b.title;
    opt.textContent = `${b.title} (${b.current_page || 0}/${b.total_pages || 300}p)`;
    select.appendChild(opt);
  });

  if (activeTimerBookTitle) {
    select.value = activeTimerBookTitle;
    onTimerBookSelected(activeTimerBookTitle);
  } else if (currentVal) {
    select.value = currentVal;
    onTimerBookSelected(currentVal);
  } else if (libraryBooks.length > 0) {
    const readingBook = libraryBooks.find(b => (b.status || '').toLowerCase() === 'reading') || libraryBooks[0];
    select.value = readingBook.title;
    onTimerBookSelected(readingBook.title);
  }
}

function selectTimerBook(bookTitle) {
  activeTimerBookTitle = bookTitle;
  const select = document.getElementById('timer-book-select');
  if (select) {
    select.value = bookTitle;
    onTimerBookSelected(bookTitle);
  }
}

function onTimerBookSelected(title) {
  activeTimerBookTitle = title;
  const book = libraryBooks.find(b => b.title === title);
  if (book) {
    const startInput = document.getElementById('timer-start-page');
    const endInput = document.getElementById('timer-end-page');
    const totalInput = document.getElementById('timer-total-pages');

    const curP = book.current_page || 0;
    const totP = book.total_pages || 300;

    if (startInput) startInput.value = curP;
    if (endInput) endInput.value = Math.min(totP, curP + 10);
    if (totalInput) totalInput.value = totP;
  }
}

function toggleTimer() {
  const toggleBtn = document.getElementById('timer-toggle-btn');
  const toggleIcon = document.getElementById('timer-toggle-icon');
  const toggleText = document.getElementById('timer-toggle-text');
  const finishBtn = document.getElementById('timer-finish-btn');
  const statusLabel = document.getElementById('timer-status-label');

  if (timerRunning) {
    // Pause
    clearInterval(timerInterval);
    timerRunning = false;
    if (toggleIcon) toggleIcon.textContent = '▶';
    if (toggleText) toggleText.textContent = 'Resume Reading';
    if (toggleBtn) toggleBtn.style.background = 'var(--gradient-primary)';
    if (statusLabel) statusLabel.textContent = 'Session Paused ⏸';
  } else {
    // Start / Resume
    timerRunning = true;
    if (toggleIcon) toggleIcon.textContent = '⏸';
    if (toggleText) toggleText.textContent = 'Pause Reading';
    if (toggleBtn) toggleBtn.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
    if (finishBtn) finishBtn.style.display = 'inline-flex';
    if (statusLabel) statusLabel.textContent = 'Focus Reading in Progress... 📖';

    timerInterval = setInterval(() => {
      timerElapsedSeconds++;
      updateTimerDisplay();

      if (timerMode === 'countdown' && timerElapsedSeconds >= timerTargetDuration) {
        clearInterval(timerInterval);
        timerRunning = false;
        showToast('🔔 Reading Goal Reached! Great session!');
        finishTimerSession();
      }
    }, 1000);
  }
}

function resetTimer() {
  clearInterval(timerInterval);
  timerRunning = false;
  timerElapsedSeconds = 0;

  const toggleIcon = document.getElementById('timer-toggle-icon');
  const toggleText = document.getElementById('timer-toggle-text');
  const toggleBtn = document.getElementById('timer-toggle-btn');
  const finishBtn = document.getElementById('timer-finish-btn');
  const statusLabel = document.getElementById('timer-status-label');

  if (toggleIcon) toggleIcon.textContent = '▶';
  if (toggleText) toggleText.textContent = 'Start Reading';
  if (toggleBtn) toggleBtn.style.background = 'var(--gradient-primary)';
  if (finishBtn) finishBtn.style.display = 'none';
  if (statusLabel) statusLabel.textContent = 'Ready to Read';

  updateTimerDisplay();
}

function updateTimerDisplay() {
  const display = document.getElementById('timer-display');
  if (!display) return;

  let secondsToShow = timerMode === 'countdown' 
    ? Math.max(0, timerTargetDuration - timerElapsedSeconds) 
    : timerElapsedSeconds;

  const hrs = Math.floor(secondsToShow / 3600);
  const mins = Math.floor((secondsToShow % 3600) / 60);
  const secs = secondsToShow % 60;

  const pad = (n) => String(n).padStart(2, '0');
  display.textContent = `${pad(hrs)}:${pad(mins)}:${pad(secs)}`;
}

async function finishTimerSession() {
  clearInterval(timerInterval);
  timerRunning = false;

  const minutesRead = Math.max(1, Math.round(timerElapsedSeconds / 60));
  const startPage = parseInt(document.getElementById('timer-start-page')?.value) || 0;
  const endPage = parseInt(document.getElementById('timer-end-page')?.value) || startPage + 1;
  const totalPages = parseInt(document.getElementById('timer-total-pages')?.value) || 300;
  const bookTitle = activeTimerBookTitle || 'Focus Session';

  // Find book and update local state
  const book = libraryBooks.find(b => b.title === bookTitle);
  if (book) {
    book.current_page = endPage;
    book.total_pages = totalPages;
    book.reading_minutes = (book.reading_minutes || 0) + minutesRead;
    if (endPage >= totalPages) {
      book.status = 'Completed';
    }
  }

  // Update streak in localStorage
  const currentStreak = parseInt(localStorage.getItem('booknest_streak') || '3') + 1;
  localStorage.setItem('booknest_streak', currentStreak);

  // Send session log to server
  try {
    await fetch('/api/reading/log', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${currentUser.uid}`
      },
      body: JSON.stringify({
        title: bookTitle,
        minutes_read: minutesRead,
        start_page: startPage,
        end_page: endPage,
        total_pages: totalPages
      })
    });
  } catch (err) {
    console.error('Error logging reading session:', err);
  }

  // Trigger celebration popup & recommendations
  triggerSessionCelebration(bookTitle, minutesRead, startPage, endPage, totalPages);

  // Reset timer UI
  resetTimer();
  updateDashboardView();
  loadLibrary();
  loadProfile();
}

// =========================================================
// 🏆 COMPLETION CELEBRATION & SMART RECOMMENDATIONS
// =========================================================
async function triggerSessionCelebration(bookTitle, minutesRead, startPage, endPage, totalPages) {
  const modal = document.getElementById('congrats-modal');
  if (!modal) return;

  const isCompleted = endPage >= totalPages && totalPages > 0;
  const titleEl = document.getElementById('congrats-title');
  const bodyEl = document.getElementById('congrats-body');
  const minVal = document.getElementById('congrats-minutes-val');
  const pageVal = document.getElementById('congrats-pages-val');
  const bmVal = document.getElementById('congrats-bookmark-val');
  const recsGrid = document.getElementById('congrats-recs-grid');

  if (titleEl) titleEl.textContent = isCompleted ? '🏆 Congratulations! Book Completed!' : 'Fantastic Reading Sprint! 🎉';
  if (bodyEl) {
    bodyEl.textContent = isCompleted 
      ? `You just finished '${bookTitle}'! Outstanding commitment. Here are curated recommendations from your 1,005 book catalog:`
      : `Awesome focus! You logged ${minutesRead}m of mindful reading and advanced to Page ${endPage}. Every page makes you wiser!`;
  }

  if (minVal) minVal.textContent = `${minutesRead}m`;
  const pagesRead = Math.max(0, endPage - startPage);
  if (pageVal) pageVal.textContent = `${pagesRead || 1}`;
  if (bmVal) bmVal.textContent = `Page ${endPage}`;

  // Confetti
  if (typeof confetti === 'function') {
    confetti({
      particleCount: 120,
      spread: 80,
      origin: { y: 0.6 },
      colors: ['#6366f1', '#10b981', '#06b6d4', '#f59e0b', '#ec4899']
    });
  }

  // Fetch recommendations based on genre
  if (recsGrid) {
    recsGrid.innerHTML = '<div style="font-size: 0.8rem; color: var(--text-muted); text-align: center; grid-column: 1/-1;">Loading personalized recommendations...</div>';
    try {
      const book = libraryBooks.find(b => b.title === bookTitle);
      const genre = book?.genre || 'Fiction';
      const res = await fetch(`/api/recommendations?genre=${encodeURIComponent(genre)}&limit=3`);
      const recs = await res.json();

      if (recs && recs.length > 0) {
        recsGrid.innerHTML = recs.map(r => `
          <div class="rec-card 3d-card" style="display: flex; align-items: center; gap: 10px; padding: 10px; background: var(--bg-card); border: 1px solid var(--border-card); border-radius: 12px;">
            <img src="${r.cover_url || 'https://images.unsplash.com/photo-1544947950-fa07a98d237f?auto=format&fit=crop&w=120&q=80'}" style="width: 44px; height: 60px; object-fit: cover; border-radius: 6px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);" alt="cover">
            <div style="flex: 1; min-width: 0;">
              <div style="font-weight: 700; font-size: 0.85rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${escapeHtml(r.title)}</div>
              <div style="font-size: 0.72rem; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">by ${escapeHtml(r.author)}</div>
              <span style="display: inline-block; font-size: 0.65rem; padding: 2px 6px; border-radius: 8px; background: rgba(99,102,241,0.1); color: var(--accent-primary); font-weight: 600; margin-top: 4px;">${escapeHtml(r.genre || 'Recommended')}</span>
            </div>
            <button type="button" class="btn-secondary 3d-btn-secondary" style="padding: 6px 10px; font-size: 0.75rem; white-space: nowrap;" onclick="addRecommendedBook('${encodeURIComponent(JSON.stringify(r))}')">
              + Add
            </button>
          </div>
        `).join('');
      } else {
        recsGrid.innerHTML = '<div style="font-size: 0.8rem; color: var(--text-muted); text-align: center; grid-column: 1/-1;">Explore more titles in your Library tab!</div>';
      }
    } catch (e) {
      console.warn('Could not fetch recommendations', e);
      recsGrid.innerHTML = '<div style="font-size: 0.8rem; color: var(--text-muted); text-align: center; grid-column: 1/-1;">Check out your library for more!</div>';
    }
  }

  modal.showModal();
}

async function addRecommendedBook(encodedJson) {
  try {
    const bookData = JSON.parse(decodeURIComponent(encodedJson));
    const res = await fetch('/api/library/save', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${currentUser.uid}`
      },
      body: JSON.stringify({
        title: bookData.title,
        author: bookData.author,
        genre: bookData.genre,
        cover_url: bookData.cover_url,
        overview: bookData.overview,
        status: 'Want to Read',
        rating: 5,
        total_pages: 300
      })
    });
    if (res.ok) {
      showToast(`✨ Added '${bookData.title}' to your Library!`);
      loadLibrary();
      updateDashboardView();
    }
  } catch (err) {
    showToast('⚠️ Could not add recommended book');
  }
}

// =========================================================
// 🔔 DAILY READING NOTIFICATION & HABIT ATTRACTOR
// =========================================================
function initDailyReminderNotification() {
  const toggle = document.getElementById('daily-notif-toggle');
  const timeSelect = document.getElementById('notif-time-select');
  const testBtn = document.getElementById('test-daily-notif-btn');
  const statusDesc = document.getElementById('notif-status-desc');

  // Load saved preferences
  const enabled = localStorage.getItem('booknest_daily_notif_enabled') !== 'false';
  const savedTime = localStorage.getItem('booknest_notif_time') || '20:00';

  if (toggle) toggle.checked = enabled;
  if (timeSelect) timeSelect.value = savedTime;

  if (statusDesc) {
    statusDesc.textContent = enabled
      ? `Active • Daily reminder scheduled at ${formatTimeString(savedTime)}`
      : 'Disabled • Tap toggle to receive daily habit reminders';
  }

  // Toggle switch handler
  toggle?.addEventListener('change', async (e) => {
    const isChecked = e.target.checked;
    localStorage.setItem('booknest_daily_notif_enabled', isChecked ? 'true' : 'false');

    if (isChecked) {
      if ('Notification' in window && Notification.permission !== 'granted') {
        const perm = await Notification.requestPermission();
        if (perm === 'granted') {
          showToast('🔔 Daily Reading Notifications Enabled!');
        } else {
          showToast('⚠️ Notifications blocked in browser settings');
        }
      } else {
        showToast('🔔 Daily Reading Notifications Active!');
      }
    } else {
      showToast('🔕 Daily Notifications Paused');
    }

    if (statusDesc) {
      statusDesc.textContent = isChecked
        ? `Active • Daily reminder scheduled at ${formatTimeString(timeSelect?.value || '20:00')}`
        : 'Disabled • Tap toggle to receive daily habit reminders';
    }
  });

  // Time select handler
  timeSelect?.addEventListener('change', (e) => {
    localStorage.setItem('booknest_notif_time', e.target.value);
    showToast(`⏰ Reminder set to ${formatTimeString(e.target.value)}`);
    if (statusDesc && toggle?.checked) {
      statusDesc.textContent = `Active • Daily reminder scheduled at ${formatTimeString(e.target.value)}`;
    }
  });

  // Test button handler
  testBtn?.addEventListener('click', async () => {
    if ('Notification' in window && Notification.permission !== 'granted') {
      const perm = await Notification.requestPermission();
      if (perm !== 'granted') {
        showToast('⚠️ Please allow notifications in prompt');
      }
    }
    sendDailyAttractionNotification(true);
  });

  // Check on load & every 60 seconds
  checkDailyAttractionNotification();
  setInterval(checkDailyAttractionNotification, 60000);
}

function formatTimeString(timeStr) {
  if (!timeStr) return '8:00 PM';
  const [h, m] = timeStr.split(':').map(Number);
  const period = h >= 12 ? 'PM' : 'AM';
  const hour12 = h % 12 || 12;
  return `${hour12}:${String(m).padStart(2, '0')} ${period}`;
}

function checkDailyAttractionNotification() {
  const isEnabled = localStorage.getItem('booknest_daily_notif_enabled') !== 'false';
  if (!isEnabled) return;

  const todayStr = new Date().toISOString().split('T')[0];
  const lastSentDate = localStorage.getItem('booknest_last_notif_date');
  if (lastSentDate === todayStr) return; // Only once a day!

  const targetTime = localStorage.getItem('booknest_notif_time') || '20:00';
  const now = new Date();
  const currentMinutes = now.getHours() * 60 + now.getMinutes();
  const [tH, tM] = targetTime.split(':').map(Number);
  const targetMinutes = tH * 60 + tM;

  // Trigger if within scheduled time
  if (currentMinutes >= targetMinutes) {
    sendDailyAttractionNotification(false);
    localStorage.setItem('booknest_last_notif_date', todayStr);
  }
}

function sendDailyAttractionNotification(isTest = false) {
  const streak = parseInt(localStorage.getItem('booknest_streak') || '3');
  const readingBook = libraryBooks.find(b => (b.status || '').toLowerCase() === 'reading');
  const wishlistBook = libraryBooks.find(b => {
    const s = (b.status || '').toLowerCase();
    return s.includes('wish') || s.includes('want');
  });

  let notifTitle = '📖 Time for Your Daily Reading!';
  let notifBody = 'Spend 15 minutes today with your books! Pick up where you left off.';

  if (readingBook) {
    const p = readingBook.current_page || 1;
    notifTitle = `📖 Dive back into '${readingBook.title}'!`;
    notifBody = `Page ${p} is waiting! Read 15 mins today to keep your ${streak}-day streak burning 🔥`;
  } else if (wishlistBook) {
    notifTitle = `⭐ Next on your list: '${wishlistBook.title}'`;
    notifBody = `Start chapter 1 today! Your reading journey is waiting 🚀`;
  } else {
    notifTitle = '✨ Daily Reading Habit Reminder';
    notifBody = `"A reader lives a thousand lives before he dies." Open your book today and fuel your mind!`;
  }

  // 1. Browser OS Push Notification
  if ('Notification' in window && Notification.permission === 'granted') {
    try {
      if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
        navigator.serviceWorker.ready.then(reg => {
          reg.showNotification(notifTitle, {
            body: notifBody,
            icon: '/static/icons/icon-192.png',
            badge: '/static/icons/icon-192.png',
            vibrate: [200, 100, 200],
            tag: 'booknest-daily-reminder'
          });
        });
      } else {
        new Notification(notifTitle, {
          body: notifBody,
          icon: '/static/icons/icon-192.png'
        });
      }
    } catch (e) {
      console.warn('Could not dispatch OS notification:', e);
    }
  }

  // 2. In-App Attraction Notification Toast
  if (isTest) {
    showToast(`🔔 Test Alert: ${notifTitle} (Daily reminder active!)`);
  } else {
    showToast(`🔔 ${notifTitle} — ${notifBody}`);
  }

  // Vibrate phone if supported
  if ('vibrate' in navigator) {
    try { navigator.vibrate([150, 75, 150]); } catch (e) {}
  }
}
