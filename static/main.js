// ====================================================================
// main.js — Chat + Tiles + Rich bubbles + Project modal (FLIP)
// ====================================================================

document.addEventListener("DOMContentLoaded", () => {
  // -------- Routing flags --------
  const CHAT_FLAG = "chat";     // ?chat=1
  const SECTION_Q = "section";  // ?section=projects|skills|me|contact

  // IMPORTANT: Let the SERVER decide what to show; client doesn't pre-route.
  const USE_LOCAL_ROUTER = false;

  const url = new URL(window.location.href);
  const qp  = url.searchParams;

  const cleanPath   = window.location.pathname.replace(/\/+$/, "");
  const onChatPage  = (cleanPath === "/chat") || qp.has(CHAT_FLAG);

  // -------- Elements --------
  const body       = document.body;
  const miniHeader = document.getElementById("mini-header");

  const chat       = document.getElementById("chat");
  const chatHeader = document.getElementById("chat-header");
  const chatStream = document.getElementById("chat-stream");

  // remember only the FIRST starter of the current chat session
  const LS_FIRST_KIND  = "pf.chat.first.kind";   // "q" | "section"
  const LS_FIRST_VALUE = "pf.chat.first.value";

  initCertLightbox();

//Ai chat animation
  function createTypewriter(el, speedMs = 14) {
  let pendingText = "";
  let shownText = "";
  let rafId = null;
  let last = 0;

  function tick(t) {
    if (!last) last = t;
    const dt = t - last;

    if (dt >= speedMs) {
      last = t;

      // reveal a few chars per tick (looks smoother than 1 char)
      const step = Math.min(3, pendingText.length);
      if (step > 0) {
        shownText += pendingText.slice(0, step);
        pendingText = pendingText.slice(step);
        el.textContent = shownText;
      }
    }

    if (pendingText.length > 0) rafId = requestAnimationFrame(tick);
    else rafId = null;
  }

  return {
    pushText(txt) {
      if (!txt) return;
      pendingText += txt;
      if (!rafId) rafId = requestAnimationFrame(tick);
    },
    stop() {
      if (rafId) cancelAnimationFrame(rafId);
      rafId = null;
    },
    getShown() { return shownText; }
  };
}


  function rememberFirst(kind, value){
    try {
      if (localStorage.getItem(LS_FIRST_KIND)) return; // already have one
      localStorage.setItem(LS_FIRST_KIND,  kind);
      localStorage.setItem(LS_FIRST_VALUE, String(value || ""));
    } catch {}
  }
  function clearFirst(){
    try { localStorage.removeItem(LS_FIRST_KIND); localStorage.removeItem(LS_FIRST_VALUE); } catch {}
  }
  function replayFirstIfAny(){
    try {
      const kind = localStorage.getItem(LS_FIRST_KIND);
      const val  = (localStorage.getItem(LS_FIRST_VALUE) || "").trim();
      if (!kind || !val) return false;

      // do not auto-scroll when replaying after refresh
      temporarilySuppressScroll();

      // we only replay the *starter*, not the whole chat
      if (kind === "q") {
        askStream(val);   // askStream will add the user bubble itself
        return true;
      }
      if (kind === "section") {
        const title = val.charAt(0).toUpperCase() + val.slice(1);
        setFeatureHeader(title);
        if (val === "projects")      showProjectsInChat(true);
        else if (val === "skills")   showSkillsInChat(true);
        else if (val === "me")       showAboutInChat(true);
        else if (val === "contact")  showContactInChat(true);
        else                         askStream(`Show the ${val} section.`);
        return true;
      }
      return false;
    } catch { return false; }
  }

  // --- Robustly resolve #tiles / #ask when HTML accidentally duplicated ---
  function findReal(id, neededSelector) {
    const els = Array.from(document.querySelectorAll(`#${id}`));
    if (els.length === 0) return null;
    if (els.length === 1) return els[0];
    // Prefer the one that contains the expected content
    const withNeeded = els.find(el => el.querySelector(neededSelector));
    if (withNeeded) return withNeeded;
    // Otherwise prefer the one with more stuff inside
    return els.sort((a,b) => (b.innerHTML.length - a.innerHTML.length))[0];
  }

  // Ask holder & pieces
  const askHolder  = findReal("ask", "#ask-bottom, #ask-box");
  const askBottom  = document.getElementById("ask-bottom");
  const askBox     = document.getElementById("ask-box");
  const askInput   = document.getElementById("ask-input");
  const askSend    = document.getElementById("ask-send");

  // Tiles
  const tilesWrap  = findReal("tiles", ".feature-grid");
  const tiles      = document.querySelectorAll(".feature-tile");

  // Glass footer wrapper (class, not id)
  const chatFooter = document.querySelector(".chat-footer");

// Data seeds
//const projDataEl   = document.getElementById("projects-data");
const skillsDataEl = document.querySelector('script#skills-logos[type="application/json"]')
                    || document.getElementById("skills-logos");

if (!askBox || !askInput || !askSend) return;

  // -------- Navigation to chat --------
  function navigateToChat(section, q) {
    const u = new URL(window.location.href);
    u.pathname = "/";
    u.searchParams.set("chat", "1");
    if (section) u.searchParams.set("section", section);
    if (q && q.trim()) u.searchParams.set("q", q.trim());   // 🔑 carry the question
    window.location.href = u.toString();
  }

// -------- Initial landing state --------
if (!onChatPage) {
  body.classList.remove("chat-mode");
  tilesWrap?.classList.remove("docked", "tiles-compact");
  askHolder?.classList.remove("docked");

  if (chat) {
    chat.style.display = "none";
  }
  if (chatFooter) chatFooter.hidden = true;

  // leaving chat context -> forget any stored starter
  clearFirst();

  // 🔥 trigger hero + tiles entrance animation on landing
// inside DOMContentLoaded, inside the home-page condition
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        document.body.classList.add("hero-animate");
      });
    });

}

  // -------- State --------
  let inChatMode = false;
  let movedAsk   = false;
  let dockingOn  = false;
  let dockRaf    = null;

  // ---- Simple session memory for this page ----
  const seen = { about:false, projects:false, skills:false, contact:false };
  let lastPrompt = null;

  function markSeenFromEl(el){
    if (!el) return;
    if (el.classList.contains("about-wrap"))           seen.about = true;
    if (el.classList.contains("projects-gallery"))     seen.projects = true;
    if (el.classList.contains("skills-wrap"))          seen.skills = true;
    if (el.classList.contains("contact-wrap"))         seen.contact = true;
  }

  function alreadyShown(selector){
    return !!chatStream.querySelector(selector);
  }

  // quick natural-language checks
  const YES_RE = /\b(yes|yeah|yep|sure|ok|okay|y|affirmative)\b/i;
  const NO_RE  = /\b(no|nah|nope|n)\b/i;

function microReply(html, delay = 350){
  // quick typing animation for short follow-ups
  const t = addTypingBubble();
  setTimeout(() => {
    try { t?.remove(); } catch {}
    addAiBubble(html);
    if (!__suppressScroll) scrollToBottom();
  }, delay);
}

  // ---------- Tiny phrasing engine (per-page memory) ----------
  const lastVariantIndex = {};     // key -> last index used
  let   lastUserNL = "";           // last normalized user text

  const PHRASES = {
    identity: [
      "I’m <b>Kareena’s AI twin</b> 🤖 — I speak in first person, but everything I say is based on Kareena’s real projects, skills, and experience.",
      "Think of me as a <b>digital Kareena</b>. I’m here to guide you through my projects, skills, background, and contact info.",
      "I’m Kareena in AI form 😄 Ask me anything about my portfolio — like projects, skills, or how to reach me."
    ],
    about_follow: [
      "Hey, still me, <b>Kareena</b> 😊 What are you curious about now?",
      "You already met me, Kareena here — try checking out my projects and skills.",
      "Hey, what's up! You can ask things like “tell me about your projects” or “what do you use for backend?”.",
      "Hey, I’m still here if you want details on my work, studies, or future goals."
    ],
    refusal: [
      "Well I can only help with my portfolio. Try <b>projects</b>, <b>skills</b>, or <b>contact</b>.",
      "Umm let’s keep it about me and my work — ask about <b>projects</b>, <b>skills</b>, or <b>contact</b>.",
      "Sorryyy, I’m scope-limited to my portfolio. Want <b>projects</b>, <b>skills</b>, or <b>contact</b>?"
    ],
    projects_hint: [
      "Projects are up — tap a card or ask about one by name.",
      "My projects are listed above — want details on a specific one?",
      "Projects are open. Pick a card or ask, e.g., “tell me about the Android app”."
    ],
    skills_hint: [
      "Skills are above — ask for any tool to dive deeper.",
      "You can ask about a specific skill, e.g., <b>Flask</b> or <b>Android</b>.",
      "Skills shown — want examples of where I used one?"
    ],
    contact_hint: [
      "Contact card is above — email & LinkedIn are there.",
      "You’ll find my email and LinkedIn in the contact card.",
      "Contact details are up there, my official email: kareenazaman@gmail.com"
    ]
  };

  function pickPhrase(key){
    const arr = PHRASES[key] || [];
    if (!arr.length) return "";
    let i = Math.floor(Math.random() * arr.length);
    if (arr.length > 1 && i === (lastVariantIndex[key] ?? -1)) {
      i = (i + 1) % arr.length; // avoid same line twice
    }
    lastVariantIndex[key] = i;
    return arr[i];
  }

  // -------- 🔕 scroll controller --------
  let __suppressScroll = false;      // when true, ignore any scroll-to-bottom calls
function temporarilySuppressScroll(ms = 450) {
  __suppressScroll = true;
  setTimeout(() => {
    __suppressScroll = false;
  }, ms);
}

  // Smooth scroll so an element sits near the top (accounts for sticky headers)
  function scrollToBubbleTop(el, offset = 72){
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const y = window.scrollY + rect.top - offset;
    window.scrollTo({ top: y < 0 ? 0 : y, behavior: "smooth" });
  }

  // -------- Utilities --------
function scrollToBottom(force = false) {
  // If force === true, ignore the suppression flag
  if (!force && __suppressScroll) return;

  const stream = document.getElementById("chat-stream");
  if (!stream) return;

  requestAnimationFrame(() => {
    window.scrollTo({
      top: document.body.scrollHeight,
      behavior: "smooth"
    });
  });
}


  function scrollBubbleToTop(el, extraOffset = 96) {
  if (!el) return;

  const header = document.querySelector(".mini-header");
  const headerHeight = header ? header.offsetHeight : 0;

  const rect = el.getBoundingClientRect();
  const targetY = window.scrollY + rect.top - headerHeight - extraOffset;

  window.scrollTo({
    top: Math.max(0, targetY),
    behavior: "smooth",
  });
}


  function markChatStarted() { tilesWrap?.classList.add("tiles-compact"); }

  // -------- Docking helpers (disabled when glass footer is active) --------
function setDocked(on) {
  if (!tilesWrap || !askHolder || !chatStream) return; // ⬅️ add chatStream check
  if (body.classList.contains("using-chat-footer")) on = false;

  if (on) {
    tilesWrap.classList.add("docked");
    askHolder.classList.add("docked");
    startDockLoop();
  } else {
    tilesWrap.classList.remove("docked");
    askHolder.classList.remove("docked");
    stopDockLoop();
    tilesWrap.style.bottom = "";
    chatStream.style.paddingBottom = "";
  }
}

  function startDockLoop() {
    if (dockingOn || body.classList.contains("using-chat-footer")) return;
    dockingOn = true;

    const recalc = () => {
      dockRaf = null;
      const askRect   = askHolder.getBoundingClientRect();
      const tilesRect = tilesWrap.getBoundingClientRect();
      const gap = 12;

      const askHeight = Math.max(askRect.height, 64);
      tilesWrap.style.bottom = (askHeight + gap) + "px";

      const safePad = askHeight + tilesRect.height + gap + 36;
      chatStream.style.paddingBottom = safePad + "px";
    };

    const schedule = () => {
      if (dockRaf) return;
      dockRaf = requestAnimationFrame(recalc);
    };

    startDockLoop._onScroll = schedule;
    startDockLoop._onResize = schedule;

    window.addEventListener("scroll", startDockLoop._onScroll, { passive: true });
    window.addEventListener("resize", startDockLoop._onResize, { passive: true });

    startDockLoop._mo = new MutationObserver(schedule);
    startDockLoop._mo.observe(chatStream, { childList: true, subtree: true });

    schedule();
  }

  function stopDockLoop() {
    if (!dockingOn) return;
    dockingOn = false;
    if (dockRaf) cancelAnimationFrame(dockRaf);
    dockRaf = null;

    if (startDockLoop._onScroll) {
      window.removeEventListener("scroll", startDockLoop._onScroll);
      startDockLoop._onScroll = null;
    }
    if (startDockLoop._onResize) {
      window.removeEventListener("resize", startDockLoop._onResize);
      startDockLoop._onResize = null;
    }
    if (startDockLoop._mo) {
      startDockLoop._mo.disconnect();
      startDockLoop._mo = null;
    }
  }

  // -------- Glass footer: keep chat padded to its height --------
  let footerObs = null;
  function syncPadWithFooter() {
    if (!chatFooter) return;
    const h = chatFooter.getBoundingClientRect().height || 0;
    chatStream.style.paddingBottom = (h + 24) + "px";
  }
  function startFooterPadLoop() {
    if (!chatFooter) return;
    stopFooterPadLoop();
    footerObs = new ResizeObserver(syncPadWithFooter);
    footerObs.observe(chatFooter);
    window.addEventListener("resize", syncPadWithFooter, { passive: true });
    syncPadWithFooter();
  }
  function stopFooterPadLoop() {
    if (footerObs) { try { footerObs.disconnect(); } catch {} footerObs = null; }
    window.removeEventListener("resize", syncPadWithFooter);
  }

  // Move tiles + ask into the single glass wrapper once
  function enableGlassFooter() {
    if (!chatFooter || body.classList.contains("using-chat-footer")) return;

    // Ensure we are moving the *real* populated sections
    if (tilesWrap) chatFooter.appendChild(tilesWrap);
    if (askHolder) chatFooter.appendChild(askHolder);

    chatFooter.hidden = false;
    body.classList.add("using-chat-footer");
    tilesWrap?.classList.add("tiles-compact");

    // Disable legacy docking and drive padding via footer height
    setDocked(false);
    stopDockLoop();
    startFooterPadLoop();
  }

  // -------- Chat helpers --------
  function enterChatMode() {
    if (!onChatPage || inChatMode) return;
    inChatMode = true;
    body.classList.add("chat-mode");
    if (miniHeader) miniHeader.style.display = "";
    chat.style.display = "block";

    // Use the unified glass footer for tiles + ask
    enableGlassFooter();

    markChatStarted();
  }

  function moveAskBoxBelowTiles() {
    if (!onChatPage || movedAsk) return;
    if (askBottom && askBox && askBox.parentElement !== askBottom) {
      askBottom.appendChild(askBox);
    }
    movedAsk = true;
  }

  function setFeatureHeader(label) {
    if (!onChatPage || !chatHeader) return;
    chatHeader.innerHTML = `
      <div class="chat-feature-header">
        <span class="chat-feature-chip"></span>
      </div>
    `;
  }

  function addUserBubble(text) {
    if (!onChatPage) return;
    const b = document.createElement("div");
    b.className = "bubble user";
    b.textContent = text;
    chatStream.appendChild(b);
    markChatStarted();
    if (!__suppressScroll) scrollToBottom();
  }

// Add this function (modify your existing addTypingBubble if needed)
function addTypingBubble() {
  if (!onChatPage) return null;
  const b = document.createElement("div");
  b.className = "bubble typing";
  b.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  chatStream.appendChild(b);
  markChatStarted();
  if (!__suppressScroll) scrollToBottom();
  return b;
}


function addAiBubble(html = "") {
  if (!onChatPage) return null;
  const b = document.createElement("div");
  b.className = "bubble ai";
  // let CSS control width
  b.innerHTML = html;
  chatStream.appendChild(b);
  markChatStarted();

  // ❌ no auto-scroll here – AI bubbles should not move the viewport
  // If you ever want special cases later, you can add checks here.

  return b;
}


  function focusAsk() { if (onChatPage) try { askInput.focus(); } catch(_) {} }

  // ===================================================================
  // ABOUT — 3 cards bubble (v2 hero)
  // ===================================================================
  function aboutHTML(opts = {}) {
    const {
      photo   = "/static/images/about/me.jpg",
      city    = "BC",
      country = "CANADA"
    } = opts;

    return `
      <div class="about-wrap">
        <h1 class="about-title v2">ABOUT ME</h1>
        <section class="about-card about-v2">

        <div class="about-hero about-hero--v2">

          <!-- LEFT COLUMN: photo + location -->
          <div class="about-left">
            <img class="about-photo v2" src="${photo}" alt="Kareena Zaman portrait">

            <div class="about-loc about-loc--v2">
              <span class="about-pin">📍</span>
              <div class="loc-text">
                <div class="loc-country">${city}, ${country}</div>
              </div>
            </div>

            <!-- ✅ DESKTOP version (under main photo + location) -->
            <div class="about-photo-row about-photo-row--desktop">
              <img src="/static/images/presentation.jpg" alt="Kareena presenting">
              <img src="/static/images/nasa-spaceapps.jpg" alt="NASA Space Apps project">
            </div>
          </div>

          <!-- RIGHT COLUMN: text -->
          <div class="about-copy v2">
            <p>
              Hey 👋 I’m <b>Kareena Zaman</b>, a <b>Computer Science student</b> at Thompson Rivers
              University in Canada, originally from Bangladesh <span class="about-flag">🇧🇩</span>,
              and an <b>Assistant Manager</b> at Suzanne’s clothing store because I love fashion as much as tech.
              I build software, mobile apps, and AI tools that make everyday tasks simpler, from Python & Flask APIs to full-stack projects.
            </p>
            <br>
            <!-- ✅ MOBILE version (right after first paragraph) -->
            <div class="about-photo-row about-photo-row--mobile">
              <img src="/static/images/presentation.jpg" alt="Kareena presenting">
              <img src="/static/images/nasa-spaceapps.jpg" alt="NASA Space Apps project">
            </div>
            <p>
              I’ve also participated in <b>various hackathons and competitions</b>, which helped me
              improve my teamwork and problem-solving abilities while gaining hands-on experience.
            </p>
            <br>
            <p>
              I’m passionate about clean design, intuitive user experiences, and practical automation. I love turning ideas into real products and hope to inspire more girls to pursue <b>STEM</b> ✨
            </p>
          </div>

        </div>

        </section>

        <section class="about-card cta-card">
          <div class="cta-title">HAVE A PROJECT IDEA?</div>
          <a class="cta-btn" href="#contact" onclick="document.querySelector('[data-section=contact]')?.click()">
            Let’s Connect →
          </a>
        </section>
      </div>
    `;
  }

  function showAboutInChat(fromTile = false) {
    return new Promise((resolve) => {
      if (fromTile) temporarilySuppressScroll();
      if (fromTile) addUserBubble("Tell me about yourself.");
      const t = addTypingBubble();
      setTimeout(() => {
        try { t?.remove(); } catch {}
        const el = addAiBubble(aboutHTML());
        el.classList.add("about-wrap"); // mark for render-once (root already has it; harmless)
        markSeenFromEl(el);
        resolve(el);
      }, 700);
    });
  }

  // ===================================================================
  // PROJECTS — gallery bubble + modal (FLIP)
  // ===================================================================
function getProjectsFromDOM() {
  const el = document.getElementById("projects-data");
  if (!el) return [];
  try { return JSON.parse(el.textContent || "[]"); }
  catch { return []; }
}


  function projectCardHTML(p) {
    const img   = (p.image || "/static/project-default.jpg");
    const stack = (p.stack || []).slice(0, 3);
    return `
      <article class="project-card" style="background-image:url('${img}')" data-key="${(p.title || "").toLowerCase()}">
        <div class="project-card__content">
          <div class="project-card__eyebrow">Project</div>
          <div class="project-card__title">${p.title || "Untitled"}</div>
          <div class="project-card__tags">
            ${stack.map(s => `<span class="project-card__tag">${s}</span>`).join("")}
          </div>
        </div>
      </article>
    `;
  }

  function projectsBubbleHTML(projects) {
    return `
      <div class="projects-gallery">
        <div class="heading">My Projects</div>
        <div class="project-grid">
          ${projects.map(projectCardHTML).join("")}
        </div>
      </div>
    `;
  }

  // Create modal once
  function ensureProjectModal() {
    let modal = document.getElementById("project-modal");
    if (modal) return modal;

    modal = document.createElement("div");
    modal.id = "project-modal";
    modal.innerHTML = `
      <div class="pm-backdrop"></div>
      <div class="pm-panel" role="dialog" aria-modal="true" aria-label="Project details">
        <button class="pm-close" aria-label="Close">×</button>
        <div class="pm-scroller">
          <div class="pm-header">
            <div class="pm-eyebrow"></div>
            <h2 class="pm-title"></h2>
            <div class="pm-tags"></div>
          </div>
          <div class="pm-desc"></div>
          <div class="pm-links"></div>
          <img class="pm-image" alt="">
          <div class="pm-gallery"></div>
        </div>
      </div>
    `;
    // Match CSS flex-centering for accurate FLIP math
    modal.style.display = "none";
    modal.style.alignItems = "center";
    modal.style.justifyContent = "center";
    modal.style.position = "fixed";
    modal.style.inset = "0";
    modal.style.zIndex = "9999";

    document.body.appendChild(modal);

    // Close handlers
    modal.querySelector(".pm-close").addEventListener("click", () => closeProjectModal());
    modal.querySelector(".pm-backdrop").addEventListener("click", () => closeProjectModal());
    return modal;
  }

  function fillProjectModal(project) {
    const modal  = ensureProjectModal();
    const tagsEl = modal.querySelector(".pm-tags");
    const gallery = project.gallery || [];

    modal.querySelector(".pm-title").textContent   = project.title || "Project";
    modal.querySelector(".pm-eyebrow").textContent = project.year || project.category || "—";
    modal.querySelector(".pm-image").src           = project.image || "/static/project-default.jpg";
    modal.querySelector(".pm-image").alt           = project.title || "Project image";
    modal.querySelector(".pm-desc").innerHTML      = project.desc ? `<p>${project.desc}</p>` : "";

        // NEW — build gallery section inside modal
    const galleryHTML = gallery.length
      ? gallery.map(img => `<img class="pm-gallery-img" src="${img}">`).join("")
      : "";

    modal.querySelector(".pm-gallery").innerHTML = galleryHTML;

    const links = project.links || {};
    const gh    = links.github ? `<a target="_blank" href="${links.github}">GitHub</a>` : "";
    const demo  = links.demo   ? `<a target="_blank" href="${links.demo}">Live</a>`     : "";

    const linksContainer = modal.querySelector(".pm-links");

    const renderedLinks = [gh, demo].filter(Boolean).join(" • ");

    if (renderedLinks) {
      linksContainer.innerHTML = renderedLinks;
      linksContainer.style.display = "block";
    } else {
      linksContainer.innerHTML = "";
      linksContainer.style.display = "none";   // ✅ completely hide section
    }

    tagsEl.innerHTML = (project.stack || []).map(s => `<span>${s}</span>`).join("");
    return modal;
  }

  // Open with FLIP from the clicked card
  function openProjectModalFromCard(cardEl, project){
    const modal = fillProjectModal(project);
    const panel = modal.querySelector(".pm-panel");
    const backdrop = modal.querySelector(".pm-backdrop");

    modal.style.display = "flex";
    document.body.classList.add("modal-open");

    // Force layout to get final panel rect (centered by flex)
    const panelRect = panel.getBoundingClientRect();
    const src = cardEl.getBoundingClientRect();

    // Compute deltas from viewport center to card center
    const vw = window.innerWidth, vh = window.innerHeight;
    const cardCx = src.left + src.width  / 2;
    const cardCy = src.top  + src.height / 2;
    const viewCx = vw / 2;
    const viewCy = vh / 2;

    const dx = cardCx - viewCx;
    const dy = cardCy - viewCy;

    // Scale from card size to panel size
    const sx = Math.max(0.05, src.width  / Math.max(1, panelRect.width));
    const sy = Math.max(0.05, src.height / Math.max(1, panelRect.height));

    // Save for closing
    modal._flip = { cardEl };

    // Initial state (over the card)
    panel.style.transition = "none";
    backdrop.style.transition = "none";
    panel.style.transform = `translate3d(${dx}px, ${dy}px, 0) scale(${sx}, ${sy})`;
    panel.style.opacity = "0.6";
    backdrop.style.opacity = "0";

    // Animate to center/1:1
    requestAnimationFrame(() => {
      panel.style.transition   = "transform 320ms cubic-bezier(.2,.8,.2,1), opacity 220ms ease";
      backdrop.style.transition = "opacity 220ms ease";
      panel.style.transform = "translate3d(0,0,0) scale(1)";
      panel.style.opacity   = "1";
      backdrop.style.opacity = "1";
    });

    // ESC to close
    document.addEventListener("keydown", function esc(e){
      if (e.key === "Escape") { closeProjectModal(); document.removeEventListener("keydown", esc); }
    }, { once: true });
  }

  function closeProjectModal(){
    const modal = document.getElementById("project-modal");
    if (!modal) return;
    const panel = modal.querySelector(".pm-panel");
    const backdrop = modal.querySelector(".pm-backdrop");

    const cardEl = modal._flip?.cardEl;
    if (!cardEl || !cardEl.isConnected){
      // graceful fade
      panel.style.transition   = "opacity 160ms ease";
      backdrop.style.transition = "opacity 160ms ease";
      panel.style.opacity = "0";
      backdrop.style.opacity = "0";
      panel.addEventListener("transitionend", () => {
        modal.style.display = "none";
        document.body.classList.remove("modal-open");
      }, { once: true });
      return;
    }

    // Re-measure for accuracy (scroll may have changed)
    const src = cardEl.getBoundingClientRect();
    const panelRect = panel.getBoundingClientRect();
    const vw = window.innerWidth, vh = window.innerHeight;
    const cardCx = src.left + src.width  / 2;
    const cardCy = src.top  + src.height / 2;
    const viewCx = vw / 2;
    const viewCy = vh / 2;

    const dx = cardCx - viewCx;
    const dy = cardCy - viewCy;
    const sx = Math.max(0.05, src.width  / Math.max(1, panelRect.width));
    const sy = Math.max(0.05, src.height / Math.max(1, panelRect.height));

    panel.style.transition   = "transform 280ms cubic-bezier(.2,.7,.2,1), opacity 220ms ease";
    backdrop.style.transition = "opacity 200ms ease";
    panel.style.transform = `translate3d(${dx}px, ${dy}px, 0) scale(${sx}, ${sy})`;
    panel.style.opacity   = "0.6";
    backdrop.style.opacity = "0";

    panel.addEventListener("transitionend", () => {
      modal.style.display = "none";
      document.body.classList.remove("modal-open");
    }, { once: true });
  }

  function showProjectsInChat(fromTile = false) {
    return new Promise((resolve) => {
      if (fromTile) temporarilySuppressScroll();
      const projects = getProjectsFromDOM();
      if (fromTile) addUserBubble("Show your projects.");
      const typing = addTypingBubble();

      setTimeout(() => {
        try { typing.remove(); } catch {}
        const bubble = addAiBubble(projectsBubbleHTML(projects));
        bubble.classList.add("projects-gallery");
        markSeenFromEl(bubble);

        // click -> modal
        bubble?.querySelector(".project-grid")?.addEventListener("click", (e) => {
          const card = e.target.closest(".project-card");
          if (!card) return;
          const key = (card.getAttribute("data-key") || "").toLowerCase();
          const p   = projects.find(x => (x.title || "").toLowerCase() === key);
          if (!p) return;
          openProjectModalFromCard(card, p);
        });


        resolve(bubble);
      }, 180);
    });
  }

  // ===================================================================
  // SKILLS — logos grid bubble
  // ===================================================================
  function getSkillsListFromDOM() {
    if (!skillsDataEl) return [];
    let raw = (skillsDataEl.textContent || skillsDataEl.innerText || "").trim();
    if (!raw) return [];
    try {
      raw = raw
        .replace(/\/\/.*$/gm, '')
        .replace(/\/\*[\s\S]*?\*\//g, '')
        .replace(/,\s*]/g, ']')
        .replace(/,\s*}/g, '}')
        .replace(/'/g, '"');
      const data = JSON.parse(raw);
      return Array.isArray(data) ? data.filter(Boolean).map(String) : [];
    } catch { return []; }
  }

function skillsGridHTML(names) {
  if (!names.length) {
    return `<p>No skills logos found. Put a JSON array in <code>#skills-logos</code> and PNGs in <code>/static/images/skills/</code>.</p>`;
  }

  // --- logo cells (same as before) ---
  const logoCells = names.map((name) => {
    const file  = String(name).trim();
    const src   = `/static/images/skills/${file}.png`;
    const label = file.replace(/[-_]/g, " ");
    return `
      <div class="skills-item" title="${label}">
        <img src="${src}" alt="${label}" loading="lazy"
             onerror="this.style.opacity=0.3; this.title='Missing: ${file}.png'">
        <div class="skills-caption">${label}</div>
      </div>
    `;
  }).join("");

  // --- full skills bubble: title + logos + chips ---
  return `
    <div class="skills-wrap">

      <h2 class="skills-expertise-title">Skills &amp; Expertise</h2>

      <!-- logo row -->
      <div class="skills-logo-row">
        <div class="skills-grid">
          ${logoCells}
        </div>
      </div>

      <!-- chip groups -->
      <div class="skills-expertise">

        <div class="skills-expertise-group">
          <div class="skills-expertise-heading">
            <span class="skills-expertise-icon">💫</span>
            <span>Frontend &amp; Product Experience</span>
          </div>
          <div class="skills-expertise-chips">
            <span class="skills-chip">Responsive UI (HTML, CSS, JS)</span>
            <span class="skills-chip">Modern component layouts</span>
            <span class="skills-chip">User experience design</span>
            <span class="skills-chip">Clean, readable interfaces</span>
          </div>
        </div>

        <div class="skills-expertise-group">
          <div class="skills-expertise-heading">
            <span class="skills-expertise-icon">⚙️</span>
            <span>Backend, APIs &amp; Databases</span>
          </div>
          <div class="skills-expertise-chips">
            <span class="skills-chip">Flask / FastAPI development</span>
            <span class="skills-chip">REST API integration</span>
            <span class="skills-chip">SQLite / MySQL</span>
            <span class="skills-chip">Firebase Auth &amp; Firestore</span>
          </div>
        </div>

        <div class="skills-expertise-group">
          <div class="skills-expertise-heading">
            <span class="skills-expertise-icon">📱</span>
            <span>Mobile Development</span>
          </div>
          <div class="skills-expertise-chips">
            <span class="skills-chip">Android Studio (Java)</span>
            <span class="skills-chip">Flutter UI design</span>
            <span class="skills-chip">Location &amp; Map integrations</span>
          </div>
        </div>

        <div class="skills-expertise-group">
          <div class="skills-expertise-heading">
            <span class="skills-expertise-icon">🧠</span>
            <span>Data, ML &amp; AI</span>
          </div>
          <div class="skills-expertise-chips">
            <span class="skills-chip">Python data stack (NumPy, Pandas)</span>
            <span class="skills-chip">ML pipelines (TF-IDF, Logistic Regression)</span>
            <span class="skills-chip">AI-powered portfolio assistant</span>
          </div>
        </div>

        <div class="skills-expertise-group">
          <div class="skills-expertise-heading">
            <span class="skills-expertise-icon">🤝</span>
            <span>Leadership &amp; Collaboration</span>
          </div>
          <div class="skills-expertise-chips">
            <span class="skills-chip">Leadership role</span>
            <span class="skills-chip">Team projects &amp; hackathons</span>
            <span class="skills-chip">Planning &amp; scheduling</span>
          </div>
        </div>

              <!-- Certifications -->
      <div class="skills-expertise-group">
        <div class="skills-expertise-heading">
          <span class="skills-expertise-icon">📜</span>
          <span>Certifications</span>
        </div>

        <div class="skills-certs-row">
          <!-- One card per certification -->
          <div class="skills-cert-card">
            <img src="/static/images/skills/nasa.png" alt="Galactic Problem Solver - Certificate of Participation">
          </div>

          <div class="skills-cert-card">
            <img src="/static/images/skills/Acres.png" alt="Certificate of Participation – Acres Industry Innovation Competition">
          </div>

          <div class="skills-cert-card">
            <img src="/static/images/skills/Google.png" alt="Introduction to Generative AI">
          </div>
          <div class="skills-cert-card">
            <img src="/static/images/skills/Suzanne.png" alt="Certified Customer Service Representative (CCSR)">
          </div>
        </div>
      </div>

      </div>
    </div>
  `;
}

function initCertLightbox() {
  // We'll treat "mobile" as <= 700px wide
  const mq = window.matchMedia("(max-width: 700px)");

  document.addEventListener("click", (e) => {
    // Only do this on small screens
    if (!mq.matches) return;

    const img = e.target.closest(".skills-cert-card img");
    if (!img) return;

    const src = img.getAttribute("src");
    if (!src) return;

    // Create overlay
    const overlay = document.createElement("div");
    overlay.className = "skills-cert-lightbox";
    overlay.innerHTML = `<img src="${src}" alt="Certification">`;

    // Tap anywhere to close
    overlay.addEventListener("click", () => {
      overlay.remove();
    });

    document.body.appendChild(overlay);
  });
}



  function showSkillsInChat(fromTile = false) {
    return new Promise((resolve) => {
      if (fromTile) temporarilySuppressScroll();
      const names = getSkillsListFromDOM();
      if (fromTile) addUserBubble("What programming languages and frameworks are you skilled in?");
      const typing = addTypingBubble();
      setTimeout(() => {
        try { typing.remove(); } catch {}
        const b = addAiBubble(skillsGridHTML(names));

        b.classList.add("skills-wrap"); // mark for render-once
        markSeenFromEl(b);
        resolve(b);
      }, 180);
    });
  }

  // ===================================================================
  // CONTACT — 3 cards + LinkedIn row
  // ===================================================================
  function contactHTML(opts = {}) {
    const S = window.STATIC_BASE || "/static/";
    const email = opts.email || "kareenazaman@gmail.com";

    return `
      <div class="contact-wrap">
        <h2 class="contact-title">Contact Information</h2>

        <div class="contact-list">
          <!-- Email -->
          <div class="contact-item">
            <img class="contact-icon" src="${S}images/contacts/emailicon.png" alt="Email">
            <a href="mailto:${email}" class="contact-text email-link">${email}</a>
            <button class="copy-btn" data-copy="${email}" aria-label="Copy email">
              <img src="${S}images/contacts/copyicon.png" alt="Copy">
            </button>
          </div>

          <!-- Location -->
          <div class="contact-item">
            <img class="contact-icon" src="${S}images/contacts/locationicon.png" alt="Location">
            <span class="contact-text">BC, Canada</span>
          </div>
        </div>

        <!-- LinkedIn -->
        <div class="linkedin-row">
          <a class="linkedin-link"
             href="https://www.linkedin.com/in/kareena-zaman/"
             target="_blank"
             rel="noopener"
             aria-label="LinkedIn">
            <img class="linkedin-logo" src="${S}images/contacts/linkedinlogo.png" alt="LinkedIn">
          </a>
        </div>
      </div>
    `;
  }

  function showContactInChat(fromTile = false) {
    return new Promise((resolve) => {
      if (fromTile) temporarilySuppressScroll();
      if (fromTile) addUserBubble("Share your contact info.");
      const t = addTypingBubble();
      setTimeout(() => {
        try { t?.remove(); } catch {}
        const b = addAiBubble(contactHTML());
        b.classList.add("contact-wrap"); // mark for render-once
        markSeenFromEl(b);

        b.querySelectorAll(".copy-btn").forEach((btn) => {
          btn.addEventListener("click", async () => {
            const val = btn.getAttribute("data-copy") || "";
            try { await navigator.clipboard.writeText(val); } catch {}
            btn.classList.add("copied");
            setTimeout(() => btn.classList.remove("copied"), 1100);
          });
        });

        resolve(b);
      }, 180);
    });
  }

  // ===================================================================
  // RENDER-ONCE helper with rotating follow-ups
  // ===================================================================
async function ensureOnce(selector, renderAsync, followupKey){
  const existing = chatStream.querySelector(selector);
  if (existing) {
    // ✅ no little sayings on repeat clicks
    requestAnimationFrame(() => scrollToBubbleTop(existing));
    return existing;
  }
  const el = await renderAsync();
  if (selector.startsWith(".")) el?.classList?.add(selector.slice(1));
  markSeenFromEl(el);
  return el;
}

  // ===================================================================
  // Streaming chat (freeform) — server routes; client renders once
  // ===================================================================
  let __askInflight = false;

  async function askStream(question) {
    if (!onChatPage) { navigateToChat(); return; }
    if (__askInflight) return;
    __askInflight = true;

    const q  = String(question || "").trim();
    const ql = q.toLowerCase();

    const repeatedUser = (ql === lastUserNL);
    lastUserNL = ql;

    enterChatMode();
    moveAskBoxBelowTiles();
    addUserBubble(q);

    // ---- local intents / follow-ups (no server call) ----
    const asksIdentity = /\b(what\s*are\s*you|what\s*r\s*u|are\s*you\s*a\s*bot|who\s*(made|built)\s*you|who\s*are\s*you|who\s*r\s*u)\b/.test(ql);
    const asksAbout =
      // greetings still count
      /\b(hi|hello|hey|hiya|morning|evening)\b/.test(ql) ||

      // only treat "about" as ABOUT ME when it's clearly about you
      /\b(about\s+(you|yourself|kareena)|introduce\s+yourself|your\s+bio|profile\s+summary)\b/.test(ql) ||

      // also handle short “about” style prompts
      (ql === "about" || ql === "about me");
    const asksProjects = /\b(project|projects)\b/.test(ql);
    const asksSkills   = /\b(skill|skills|framework|frameworks)\b/.test(ql);
    const asksContact  = /\b(contact|reach|email|e-?mail|linkedin|link\s*edin|connect)\b/.test(ql);

    // quick yes/no response to the last micro prompt
    if (YES_RE.test(q) || NO_RE.test(q)) {
      if (lastPrompt === "about-follow") {
        if (YES_RE.test(q)) {
          microReply("Great — want to see my <b>projects</b> or <b>skills</b>? You can also ask for <b>contact</b>.");
        } else {
          microReply("All good! Ask me anything specific — e.g., “show projects”, “Android app details”, or “share contact”.");
        }
        __askInflight = false;
        return;
      }
    }

    // Identity — always answer "who are you" as identity (not about-follow)
    if (asksIdentity) {
      microReply(pickPhrase("identity"));
      __askInflight = false;
      return;
    }

    // About — render once, otherwise follow-up
    if (asksAbout) {
      if (alreadyShown(".about-wrap")) {
        microReply(pickPhrase("about_follow"));
        lastPrompt = "about-follow";
        __askInflight = false;
        return;
      }
      await ensureOnce(".about-wrap", showAboutInChat, "about_follow");
      lastPrompt = "about-follow";
      __askInflight = false;
      return;
    }

    // Section shortcuts — render once with rotating hints
    if (asksProjects) {
      await ensureOnce(".projects-gallery", showProjectsInChat, "projects_hint");
      __askInflight = false;
      return;
    }
    if (asksSkills) {
      await ensureOnce(".skills-wrap", showSkillsInChat, "skills_hint");
      __askInflight = false;
      return;
    }
    if (asksContact) {
      await ensureOnce(".contact-wrap", showContactInChat, "contact_hint");
      __askInflight = false;
      return;
    }

    // ----------------- Otherwise call backend stream -----------------
    const typing = addTypingBubble();
    let gotFirstChunk = false;
    let ai = null;
    let buffer = "";
    let typer = null;

    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q })
      });

      if (!res.ok || !res.body) {
        const fb = await fallbackChat(q);
        try { typing?.remove(); } catch {}
        if (fb && fb.trim()) addAiBubble(fb);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      // make sure typing dots are visible at least this long
      const minTypingMs = 500;
      const startTime = performance.now();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        if (!chunk) continue;

        buffer += chunk;

        if (!gotFirstChunk) {
          gotFirstChunk = true;

          // wait if needed so typing is visible
          const elapsed = performance.now() - startTime;
          const wait = Math.max(0, minTypingMs - elapsed);
          if (wait > 0) await new Promise(r => setTimeout(r, wait));

          // swap typing bubble for an AI bubble
          try { typing?.remove(); } catch {}
          ai = addAiBubble("");

          // caret blink class (CSS)
          ai.classList.add("streaming");

          // "ChatGPT-like" typing effect
          typer = createTypewriter(ai, 14); // increase to slow (18–22)
        }

        // typed preview while streaming (avoid partial HTML breaking DOM)
        const plainChunk = chunk
          .replace(/<\/p>\s*<p>/gi, "\n\n")     // paragraph breaks
          .replace(/<br\s*\/?>/gi, "\n")        // line breaks
          .replace(/<[^>]*>/g, "");             // remove remaining tags

        typer?.pushText(plainChunk);


        if (!__suppressScroll) scrollToBottom();

        // Server placeholders -> render once
        if (buffer.includes("projects-gallery")) {
          ai?.remove();
          await ensureOnce(".projects-gallery", showProjectsInChat, "projects_hint");
          return;
        }
        if (buffer.includes("skills-wrap")) {
          ai?.remove();
          await ensureOnce(".skills-wrap", showSkillsInChat, "skills_hint");
          return;
        }
        if (buffer.includes("about-wrap")) {
          ai?.remove();
          await ensureOnce(".about-wrap", showAboutInChat, "about_follow");
          lastPrompt = "about-follow";
          return;
        }
        if (buffer.includes("contact-wrap")) {
          ai?.remove();
          await ensureOnce(".contact-wrap", showContactInChat, "contact_hint");
          return;
        }
      }

      // No chunks? fall back to non-stream
      if (!gotFirstChunk) {
        const fb = await fallbackChat(q);
        try { typing?.remove(); } catch {}
        if (fb && fb.trim()) addAiBubble(fb);
        return;
      }

      // stream finished: swap in final formatted HTML once
      try { typing?.remove(); } catch {}
    if (ai) {
      typer?.stop();
      ai.classList.remove("streaming");

      requestAnimationFrame(() => {
        ai.innerHTML = buffer;
        if (!__suppressScroll) scrollToBottom();
      });
    }

    } catch (e) {
      const fb = await fallbackChat(q);
      try { typing?.remove(); } catch {}
      if (fb && fb.trim()) addAiBubble(fb);
    } finally {
      __askInflight = false;
    }
  }


  async function fallbackChat(question) {
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });
      const data = await res.json();
      const html = String(data.html || "");

      // Handle placeholders from server (render once + rotating hints)
      if (html.includes("projects-gallery")) {
        await ensureOnce(".projects-gallery", showProjectsInChat, "projects_hint");
        return "";
      }
      if (html.includes("skills-wrap")) {
        await ensureOnce(".skills-wrap", showSkillsInChat, "skills_hint");
        return "";
      }
      if (html.includes("about-wrap")) {
        await ensureOnce(".about-wrap", showAboutInChat, "about_follow");
        return "";
      }
      if (html.includes("contact-wrap")) {
        await ensureOnce(".contact-wrap", showContactInChat, "contact_hint");
        return "";
      }

      // Guard: never allow blank html
      if (!html.trim()) {
        return "<p>Hi! I’m <strong>Kareena’s AI assistant</strong> 🤖 — created by Kareena to answer questions about her skills, projects, and how to contact her.</p>";
      }

      return html;
    } catch {
      return "<p>Sorry, something went wrong.</p>";
    }
  }

// ===================================================================
// Tile actions (ONLY tiles)
// ===================================================================
tiles.forEach((btn) => {
  btn.addEventListener("click", async () => {
    hidePortfolioNotice();
    const section = btn.getAttribute("data-section");
    if (!section) return;

    // From landing → navigate to chat
    if (!onChatPage) {
      navigateToChat(section);
      return;
    }

    // Already on chat page
    const title = section.charAt(0).toUpperCase() + section.slice(1);
    setFeatureHeader(title);
    enterChatMode();
    moveAskBoxBelowTiles();

    rememberFirst("section", section);
    temporarilySuppressScroll();

    let el = null;
    if (section === "projects") el = await ensureOnce(".projects-gallery", () => showProjectsInChat(true), "projects_hint");
    else if (section === "skills") el = await ensureOnce(".skills-wrap", () => showSkillsInChat(true), "skills_hint");
    else if (section === "me") el = await ensureOnce(".about-wrap", () => showAboutInChat(true), "about_follow");
    else if (section === "contact") el = await ensureOnce(".contact-wrap", () => showContactInChat(true), "contact_hint");

    if (el) requestAnimationFrame(() => scrollToBubbleTop(el));
    focusAsk();
  });
});



// ===================================================================
// Ask box
// ===================================================================

// Landing-only helper
function goToChatWithIntro() {
  const u = new URL(window.location.href);
  u.pathname = "/";
  u.searchParams.set("chat", "1");
  u.searchParams.set("intro", "1");
  window.location.href = u.toString();
}

// Landing-only: focusing/clicking ask input jumps to chat (intro)
if (!onChatPage) {
  askInput.addEventListener("focus", goToChatWithIntro, { once: true });
  askInput.addEventListener("click", goToChatWithIntro, { once: true });
}

// Works on BOTH landing + chat
function submitAsk() {
  hidePortfolioNotice();
  const q = (askInput.value || "").trim();
  if (!q) return;

  // Landing → go to chat, carry q in URL
  if (!onChatPage) {
    navigateToChat(null, q);
    return;
  }

  // Chat → send normally
  askInput.value = "";
  rememberFirst("q", q);
  askStream(q);
}

askSend.addEventListener("click", submitAsk);
askInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") submitAsk();
});


function showPortfolioNoticeOnce() {
  if (document.querySelector(".portfolio-notice")) return;

  const wrap = document.createElement("div");
  wrap.className = "portfolio-notice";
  wrap.innerHTML = `
    <div class="chat-notice">
      <div class="chat-notice-title">Portfolio Assistant</div>
      <div class="chat-notice-text">
        This assistant is designed to answer questions about <b>Kareena Zaman’s</b> portfolio —
        projects, skills, experience, and contact info.
        Try: <b>“What backend tools do you use?”</b> or <b>“Show your projects.”</b>
      </div>
    </div>
  `;
  chatStream.prepend(wrap);
}

function hidePortfolioNotice() {
  const notice = document.querySelector(".portfolio-notice");
  if (!notice || notice.classList.contains("fade-out")) return;

  notice.classList.add("fade-out");

  // remove from DOM after animation
  setTimeout(() => notice.remove(), 420);
}


// ===================================================================
// Seeded section on /?chat=1
// ===================================================================
if (onChatPage) {
  const seed     = qp.get(SECTION_Q);       // "me", "projects", "skills", "contact"
  const prefillQ = (qp.get("q") || "").trim();

  enterChatMode();
  moveAskBoxBelowTiles();
  initScrollToBottomButton();

  if (qp.get("intro") === "1") {
    showPortfolioNoticeOnce();
    try {
      const cleaned = new URL(window.location.href);
      cleaned.searchParams.delete("intro");
      window.history.replaceState({}, "", cleaned.toString());
    } catch {}
  }

  if (prefillQ) {
    rememberFirst("q", prefillQ);
    temporarilySuppressScroll();
    askStream(prefillQ);

    try {
      const cleaned = new URL(window.location.href);
      cleaned.searchParams.delete("q");
      window.history.replaceState({}, "", cleaned.toString());
    } catch {}
    focusAsk();
    return;
  }

  if (seed) {
    rememberFirst("section", seed);
    const title = seed.charAt(0).toUpperCase() + seed.slice(1);
    setFeatureHeader(title);
    temporarilySuppressScroll();

    if (seed === "projects") showProjectsInChat(true);
    else if (seed === "skills") showSkillsInChat(true);
    else if (seed === "me") showAboutInChat(true);
    else if (seed === "contact") showContactInChat(true);

    focusAsk();
    return;
  }

  if (replayFirstIfAny()) {
    focusAsk();
    return;
  }

  focusAsk();
}
/* =========================================================
   Scroll-to-bottom button (WINDOW scroll version)
   - Your chat page scrolls the window (not chatStream)
   - Shows ONLY when user is not near the bottom
   - Click scrolls to bottom of the page
   - Function name MUST be: scrollChattoBottom
   ========================================================= */

function scrollChattoBottom() {
  // Smooth scroll the WINDOW to the bottom of the page
  window.scrollTo({
    top: document.documentElement.scrollHeight,
    behavior: "smooth",
  });
}

function initScrollToBottomButton() {
  // Only run on chat page
  if (typeof onChatPage !== "undefined" && !onChatPage) return;

  const btn = document.getElementById("scrollToBottomBtn");
  if (!btn) {
    // If you don’t add the HTML button, nothing can work
    console.warn("scroll btn: #scrollToBottomBtn not found in HTML");
    return;
  }

  // -----------------------------
  // Keep button ABOVE the glass footer with a nice gap
  // -----------------------------
function syncButtonAboveFooter() {
  const footer = document.getElementById("chat-footer");

  let footerH = 0;
  if (footer && !footer.hidden) {
    footerH = footer.getBoundingClientRect().height;
  }

  const gap = 22;

  document.documentElement.style.setProperty(
    "--scroll-btn-bottom",
    `${footerH + gap}px`
  );
}

// Run AFTER layout settles
requestAnimationFrame(syncButtonAboveFooter);
setTimeout(syncButtonAboveFooter, 0);
setTimeout(syncButtonAboveFooter, 250);

window.addEventListener("resize", syncButtonAboveFooter, { passive: true });

if (window.ResizeObserver) {
  const footer = document.getElementById("chat-footer");
  if (footer) {
    const ro = new ResizeObserver(syncButtonAboveFooter);
    ro.observe(footer);
  }
}


  // Click handler
  btn.addEventListener("click", scrollChattoBottom);

  // -----------------------------
  // Show button only when NOT near bottom
  // -----------------------------
  function isNearBottom() {
    // How far the user is from the bottom of the page
    const scrollBottom = window.scrollY + window.innerHeight;
    const pageBottom = document.documentElement.scrollHeight;
    const distanceFromBottom = pageBottom - scrollBottom;

    // Hide button when within 120px of bottom
    return distanceFromBottom < 120;
  }

  function updateButton() {
    btn.classList.toggle("show", !isNearBottom());
  }

  // Update when user scrolls the page
  window.addEventListener("scroll", updateButton, { passive: true });

  // Update when new chat bubbles get appended (page height changes)
  const chatStreamEl = document.getElementById("chat-stream");
  if (chatStreamEl) {
    const obs = new MutationObserver(() => updateButton());
    obs.observe(chatStreamEl, { childList: true, subtree: true });
  }

  // Initial state
  updateButton();
}
});   // ⬅️ close DOMContentLoaded



