# ===========================
# Flask app for AI Portfolio - FINAL REVISED
# - Natural conversation flow with context tracking
# - Smarter follow-up detection
# - Varied, non-repetitive responses
# - TF-IDF retrieval for portfolio Q&A
# ===========================

from flask import Flask, render_template, request, jsonify, Response, session
from pathlib import Path
import os, json, yaml, html, time, random
from typing import List, Dict, Any
import re

# ML deps
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from prompts import (
    EMPLOYER_PATTERNS,
    EMPLOYER_ANSWERS,
    JUDGMENT_PATTERNS,
    JUDGMENT_ANSWERS
)

# ---------- Paths / Flask ----------
BASE_DIR = Path(__file__).resolve().parent

# ---------- TF-IDF cache paths ----------
INDEX_DIR = BASE_DIR / "models"
VEC_PATH  = INDEX_DIR / "tfidf_vectorizer.joblib"
MAT_PATH  = INDEX_DIR / "tfidf_doc_mat.joblib"
DOCS_PATH = INDEX_DIR / "tfidf_docs.joblib"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# In production, I set FLASK_SECRET_KEY in the hosting environment.
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")


def auto_align_words(text):
    shorthand = {
        "u": "you", "r": "are", "ur": "your", "urself": "yourself",
        "abt": "about", "pls": "please", "thx": "thanks", "ty": "thank you"
    }
    words = text.lower().split()
    return " ".join([shorthand.get(w, w) for w in words])

# ============================================================
# Data helpers (YAML / Markdown)
# ============================================================
def _safe_yaml(path: Path, default):
    """Loads YAML safely and always returns a predictable default."""
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else default


def load_yaml(filename: str):
    """
    Loads data from /data.
    - skills.yml is a dict grouped by category
    - most other files are lists
    """
    p = BASE_DIR / "data" / filename
    if filename.endswith("skills.yml"):
        return _safe_yaml(p, {})
    return _safe_yaml(p, [])


def load_kb_markdown() -> List[Dict[str, Any]]:
    """
    Loads kb/*.md into retrieval chunks.
    Each markdown file becomes one searchable document.
    """
    kb_dir = BASE_DIR / "kb"
    chunks: List[Dict[str, Any]] = []
    if not kb_dir.exists():
        return chunks

    for md_file in kb_dir.glob("*.md"):
        try:
            text = md_file.read_text(encoding="utf-8").strip()
        except Exception:
            text = ""
        title = md_file.stem.replace("_", " ").title()
        if text:
            chunks.append({"id": md_file.stem, "title": title, "content": text})
    return chunks


def normalize_projects_to_text(proj: Dict[str, Any]) -> str:
    """Turns a project YAML entry into a clean, formatted block for chat."""
    title = proj.get("title", "")
    desc = proj.get("desc", "")
    stack = ", ".join((proj.get("stack", []) or []))
    links = proj.get("links", {}) or {}
    gh = links.get("github", "")
    dm = links.get("demo", "")

    parts = []
    if title:
        parts.append(f"**Project: {title}**")
    if desc:
        parts.append(desc)
    if stack:
        parts.append(f"**Stack:** {stack}")
    if gh:
        parts.append(f"**GitHub:** <a href='{gh}' target='_blank' style='text-decoration: underline;'>View Code</a>")
    if dm:
        parts.append(
            f"**Live Demo:** <a href='{dm}' target='_blank' style='text-decoration: underline;'>Visit Site</a>")

    # Join with newlines so the HTML formatter creates distinct visual lines
    return "\n".join(p for p in parts if p).strip()


def build_corpus_from_portfolio() -> List[Dict[str, Any]]:
    """
    Builds the retrieval corpus from:
    - kb markdown files
    - projects.yml
    - skills.yml (split into chunks by category)
    """
    corpus: List[Dict[str, Any]] = []

    # Markdown knowledge
    corpus.extend(load_kb_markdown())

    # Projects
    projects = load_yaml("projects.yml")
    for i, p in enumerate(projects or []):
        text = normalize_projects_to_text(p)
        if text:
            corpus.append({
                "id": f"project_{i}",
                "title": p.get("title", f"Project {i + 1}"),
                "content": text
            })

    # Skills (split into retrieval-friendly chunks)
    skills = load_yaml("skills.yml")
    if isinstance(skills, dict) and skills:
        for group, items in skills.items():
            if not items:
                continue

            # ✅ make sure every item becomes a string (handles dicts/lists safely)
            safe_items = []
            for it in (items if isinstance(items, list) else [items]):
                if isinstance(it, str):
                    safe_items.append(it)
                elif isinstance(it, dict):
                    # dict -> "key: value" lines (nice for YAML like {name: X, level: Y})
                    for k, v in it.items():
                        safe_items.append(f"{k}: {v}")
                else:
                    safe_items.append(str(it))

            joined = ", ".join(safe_items)
            corpus.append({
                "id": f"skills_{group.lower().replace(' ', '_')}",
                "title": f"Skills — {group}",
                "content": f"{group} skills, tools, frameworks, stack and technologies I use: {joined}."
            })

    return corpus


# ============================================================
# Smart handling for casual / off-topic / personality questions
# ============================================================
def handle_edge_case(question: str, qa_system=None) -> tuple[bool, str]:

    """
    Handles small-talk / personality / semi-related questions in a friendly,
    first-person way.
    Returns: (handled, html_response)
    """
    q_lower = (question or "").lower().strip()

    # ---------- Greeting ----------
    GREETING_PATTERNS = (
        "hi", "hello", "hey", "hiya", "yo",
        "good morning", "good afternoon", "good evening"
    )

    # exact greeting or starts-with greeting (handles: "hi", "hi!", "hey there")
    if any(q_lower == g or q_lower.startswith(g + " ") for g in GREETING_PATTERNS):
        return True, random.choice(RESPONSE_BANK["greeting"])

    # ---------- Smalltalk: "how are you" ----------
    SMALLTALK_PATTERNS = (
        "how are you", "how r u", "how are u", "hru",
        "how's it going", "how is it going",
        "how you doing", "how u doing",
        "wyd", "wsp", "what's up", "sup"
    )

    SMALLTALK_REPLIES = [
        "<p>I’m doing good 😊 Thanks for asking! Want to see my projects or skills?</p>",
        "<p>Doing great — in portfolio mode ✨ Want a quick overview or a project?</p>",
        "<p>All good here 😄 Tell me what you’re looking for and I’ll guide you.</p>",
        "<p>Vibing 😌 Do you want to explore projects, skills, or contact info?</p>",
    ]

    if any(p in q_lower for p in SMALLTALK_PATTERNS):
        # Avoid repeating the same smalltalk reply twice in a row
        s = session.setdefault("pf", {})
        last = s.get("last_smalltalk")
        choices = [r for r in SMALLTALK_REPLIES if r != last] or SMALLTALK_REPLIES
        reply = random.choice(choices)
        s["last_smalltalk"] = reply
        session.modified = True
        return True, reply

    # --- Acknowledgements / reactions (including combos like "lol okay") ---

    clean2 = q_lower.strip().strip("!?.,")

    # Tokenize basic words + a few emojis
    tokens = re.findall(r"[a-z']+|[😂🤣😭😆]", clean2)

    # Normalize stretchy reactions
    tokens = ["oh" if re.fullmatch(r"oh+", t) else t for t in tokens]
    tokens = ["hmm" if re.fullmatch(r"hm+", t) else t for t in tokens]
    token_set = set(tokens)

    ACK_WORDS = {
        "ok", "okay", "k", "kk", "alright", "sure", "cool", "nice",
        "sounds", "good", "got", "it", "yep", "ya", "yes",
        "oh", "okayyy", "okayy", "alr", "aight"
    }

    REACT_WORDS = {
        "lol", "lmao", "lmfao", "haha", "hehe",
        "ohh", "ohhh", "hmm", "huh", "ah", "aww"
    }
    REACT_EMOJIS = {"😂", "🤣", "😭", "😆"}

    # "oh" / "oh okay" / "oh ok" style acknowledgement
    if tokens and all(t in ACK_WORDS or t in REACT_WORDS for t in tokens) and "oh" in token_set:
        last_intent = session.get("pf", {}).get("last_intent")

        if last_intent == "projects":
            msg = "Oh okay 😄 Want me to pick a best project to start with?"
        elif last_intent == "skills":
            msg = "Oh okay! Do you want my frontend, backend, or AI/ML skills?"
        elif last_intent == "contact":
            msg = "Oh okay — want my email, LinkedIn, or both?"
        else:
            msg = _pick_nonrepeating_session(
                "smalltalk_oh",
                [
                    "Oh okay 😊 Want to explore Projects, Skills, or Contact?",
                    "Ohhh got you 😄 What should we look at next?",
                    "Oh okay! If you tell me what you're looking for, I’ll guide you.",
                ],
            )

        return True, f"<p>{msg}</p>"

    # Pure ack (single word)
    if clean2 in {"ok", "okay", "k", "kk", "alright", "sure", "cool", "nice", "yep", "yes"}:
        msg = _pick_nonrepeating_session(
            "smalltalk_ack",
            [
                "Okay 😄 What do you want to explore next — Projects, Skills, or Contact?",
                "Got it ✅ Want to see my projects or skills?",
                "Cool ✨ Try: “show projects”, “what skills do you have?”, or “contact”.",
                "Sounds good 😌",
            ],
        )
        return True, f"<p>{msg}</p>"

    # Pure reaction (single word / emoji)
    if (clean2 in REACT_WORDS) or (token_set & REACT_EMOJIS) or (clean2 in REACT_EMOJIS):
        msg = _pick_nonrepeating_session(
            "smalltalk_react",
            [
                "😂 Haha",
                "Haha 😄",
                "Hehe 😌",
                "lol glad that made you smile 😆",
                "😁",
            ],
        )
        return True, f"<p>{msg}</p>"

    # Mixed reaction+ack like "lol okay", "haha sure", "ok lol"
    all_smalltalk = all(
        (t in ACK_WORDS) or (t in REACT_WORDS) or (t in REACT_EMOJIS)
        for t in tokens
    )
    has_ack = any(t in ACK_WORDS for t in tokens)
    has_react = any((t in REACT_WORDS) or (t in REACT_EMOJIS) for t in tokens)

    if tokens and all_smalltalk and (has_ack and has_react):
        last_intent = session.get("pf", {}).get("last_intent")

        if last_intent == "projects":
            msg = "😂 Okay — want me to recommend a project to start with, or show the full list?"
        elif last_intent == "skills":
            msg = "Hehe 😄 Want frontend, backend, or AI/ML skills?"
        elif last_intent == "contact":
            msg = "Okay 😄 Want my email, LinkedIn, or both?"
        else:
            msg = _pick_nonrepeating_session(
                "smalltalk_combo",
                [
                    "😂 Okay — what do you want to explore next?",
                    "Hehe 😄 Want to see Projects, Skills, or Contact?",
                    "lol okay 😄 Try: “show projects” or “what skills do you use?”",
                ],
            )

        return True, f"<p>{msg}</p>"

    # ---------- Declines / "no" / "lol no" ----------
    NEG_WORDS = {"no", "nope", "nah", "na", "naw", "never"}
    # tokens are already computed above in your code:

    has_neg = any(t in NEG_WORDS for t in tokens)
    all_simple = all(
        (t in ACK_WORDS) or (t in REACT_WORDS) or (t in REACT_EMOJIS) or (t in NEG_WORDS) or (t == "thanks")
        for t in tokens
    )

    if tokens and all_simple and has_neg:
        last_intent = session.get("pf", {}).get("last_intent")

        if last_intent == "projects":
            msg = _pick_nonrepeating_session(
                "decline_projects",
                [
                    "No worries 😄 Want to see my skills instead?",
                    "All good! If projects aren’t what you need, I can share my skills or experience.",
                    "Okay 😊 Would you like a quick skills summary instead?"
                ],
            )
        elif last_intent == "skills":
            msg = _pick_nonrepeating_session(
                "decline_skills",
                [
                    "Got you 😄 Want to look at my projects instead?",
                    "No problem! We can switch to projects or contact info.",
                    "Okay 😊 What would you like to see instead — projects or contact?"
                ],
            )
        elif last_intent == "contact":
            msg = _pick_nonrepeating_session(
                "decline_contact",
                [
                    "No worries 😊 Want to see projects or skills instead?",
                    "All good! If you ever want to reach me later, it’s in the Contact section.",
                    "Okay 😄 Want a quick overview of my projects instead?"
                ],
            )
        else:
            msg = _pick_nonrepeating_session(
                "decline_generic",
                [
                    "lol fair 😄 What do you want to explore instead — Projects, Skills, or Contact?",
                    "No worries 😊 Want me to show projects, skills, or experience?",
                    "Okay 😄 Tell me what you’re looking for and I’ll point you to it."
                ],
            )

        return True, f"<p>{msg}</p>"

    # ---------- Farewell ----------
    if any(phrase == q_lower or phrase in q_lower for phrase in [
        "bye", "goodbye", "bye bye", "see you", "see you later",
        "talk to you later", "take care", "ttyl", "gotta go", "i have to go"
    ]):
        return True, random.choice(RESPONSE_BANK["farewell"])

    # ---------- "Who / what are you?" ----------
    if any(phrase in q_lower for phrase in [
        "who are you", "who r u", "who are u", "who r you",
        "what are you", "what r u", "what r you",
        "who is this", "who dis", "do u know kareena", "do you know kareena"
    ]):
        responses = [
            "<p>I’m Kareena’s AI version 🤖 I talk as her, everything I say is about my real work, skills, and experience.</p>",
            "<p>I’m basically a digital Kareena — this little bot exists just to walk you through my projects, skills, and background.</p>",
            "<p>Think of me as Kareena’s AI twin. I can tell you about what I build, what I study, and how to contact me.</p>",
        ]
        return True, random.choice(responses)

    # ---------- Location / where you live / where from ----------
    if any(phrase in q_lower for phrase in [
        "where do you live", "where do u live",
        "where are you from", "where r u from",
        "where you from", "where r you from",
        "where are u from", "where do you stay", "where do u stay",
        "location", "based", "live in", "from?", "kamloops", "bc", "british columbia",
        "canada"
    ]):
        # Try to confirm from the knowledge base if available
        if qa_system:
            for doc in qa_system.docs:
                content_lower = doc["content"].lower()
                if "kamloops" in content_lower or "british columbia" in content_lower:
                    return True, (
                        "<p>I’m based in Kamloops, BC, Canada 🇨🇦<br><br>"
                        "Most of my studying and projects are built around life here.</p>"
                    )

        return True, (
            "<p>I’m based in Canada 🇨🇦</p>"
            "<p>If you want to know more, ask about my studies or projects.</p>"
        )

    # ---------- Origin / Bangladeshi questions ----------
    if any(phrase in q_lower for phrase in [
        "are you from bangladesh", "r u from bangladesh", "are u from bangladesh",
        "are you bangladeshi", "are u bangladeshi",
        "are you from bd", "r u from bd",
        "originally from bangladesh", "your origin", "where are your roots",
        "were you born in bangladesh", "from bangladesh?",
        "where were u born", "where were you born"
    ]):
        responses = [
            "<p>I was born in Bangladesh 🇧🇩 and later moved to Canada. I’m currently based in Kamloops, BC studying Computer Science.</p>",
            "<p>I’m originally from Bangladesh 🇧🇩, but I now live in British Columbia, Canada where I’m pursuing my CS degree.</p>",
            "<p>My roots are in Bangladesh 🇧🇩 — that’s where I was born. Now I’m building my tech journey in Canada 🇨🇦.</p>",
            "<p>Born in Bangladesh 🇧🇩, now living in Canada 🇨🇦 as a Computer Science student.</p>",
            "<p>I was born in Bangladesh 🇧🇩, and today I’m based in Kamloops, BC focusing on software and AI projects.</p>",
            "<p>Bangladesh is where I’m from originally 🇧🇩 — but Canada is where I’m currently studying and building my career in tech.</p>",
        ]
        return True, random.choice(responses)

    # ---------- “Are you Kareena?” ----------
    if any(phrase in q_lower for phrase in [
        "are you kareena", "r u kareena", "are u kareena",
        "is this kareena", "is that kareena"
    ]):
        responses = [
            "<p>I’m not the human Kareena — I’m her AI twin. But all the projects and experience I talk about are really mine.</p>",
            "<p>Close enough 😄 I’m Kareena’s AI version, trained just to talk about my work, skills, and journey.</p>",
        ]
        return True, random.choice(responses)

    # ---------- AI capability questions ----------
    if any(word in q_lower for word in [
        "are you chatgpt", "are you ai", "real person", "human", "are u chatgpt", "r u chatgpt", "are you chat gpt", "are u chat gpt","chatgpt?", "gpt?",
    ]):
        return True, (
            "<p>I’m Kareena’s AI assistant, created to help walk you through her professional experience and portfolio — and no, I’m not ChatGPT 😄</p>"
        )

    # ---------- Age questions ----------
    AGE_RE = re.compile(r"\b(age|how old|what'?s (your|ur) age)\b", re.I)

    if AGE_RE.search(q_lower):
        responses = [
            "<p>I don’t list my exact age here — this space is more about my skills, projects, and experience.</p>",
            "<p>Age isn’t really the focus of this portfolio. I’d rather show what I’ve actually built and learned.</p>",
        ]
        return True, random.choice(responses)

    # ---------- User correcting / referencing the bot ----------
    if any(phrase in q_lower for phrase in [
        "you said", "you just said", "it says", "didn't you", "didnt you",
        "but you", "you told me", "you were saying"
    ]):
        responses = [
            "<p>Good catch — I might not have answered that perfectly.</p><p>If you ask me again more directly, I’ll try to be clearer.</p>",
            "<p>Fair point 😅 I’m a tiny local model, so I sometimes oversimplify. Tell me exactly what you want to know and I’ll retry.</p>",
        ]
        return True, random.choice(responses)

    # ---------- Rude / negative comments ----------
    if any(word in q_lower for word in [
        "stupid", "dumb", "useless", "annoying", "hate you", "hate u",
        "bad bot", "you suck", "u suck", "rude", "mean", "idiot"
    ]):
        responses = [
            "<p>Let’s keep it respectful 😊 I’m happy to help with questions about my projects, skills, or experience.</p>",
            "<p>😅 I’m doing my best — if something didn’t make sense, tell me what you meant and I’ll answer clearly.</p>",
            "<p>Please keep it respectful 😊 What would you like to know — projects, skills, education, or contact info?</p>",
            "<p>Fair 😄 If you’re testing me, try: “show projects” or “what tech stack do you use?”</p>",
        ]
        return True, random.choice(responses)

    # ---------- Inappropriate language ----------
    if any(word in q_lower for word in [
        "fuck", "fucking", "shit", "bitch", "asshole",
        "bastard", "wtf", "stfu", "slut", "shut the fuck up", "disgusting"
    ]):
        responses = [
            "<p>Let’s keep the conversation respectful. I’m here to help you learn about my professional work.</p>",
            "<p>I can’t engage with inappropriate language. Feel free to ask about my experience or projects instead.</p>",
            "<p>Please keep things professional. I’m happy to help if you’d like to explore my portfolio.</p>",
            "<p>I’m here to discuss my professional background and work — let’s keep it appropriate.</p>",
            "<p>If you have questions about my skills, experience, or projects, I’d be glad to help.</p>",
            "<p>I’m designed to assist with information about my professional life. Let’s keep the discussion respectful.</p>",
            "<p>I'm sorry that's a bit inappropriate to say to someone.</p>"
        ]
        return True, random.choice(responses)

    # ---------- Thanks / gratitude ----------
    tokens = re.findall(r"[a-z']+", q_lower)
    token_set = set(tokens)
    THANKS_WORDS = {"thanks", "thanx", "thank", "ty", "tysm"}

    if ("thank" in token_set) or (token_set & THANKS_WORDS) or ("thank you" in q_lower) or ("thank u" in q_lower):
        responses = [
            "<p>You’re welcome 🧡</p>",
            "<p>Glad I could help! 😊</p>",
            "<p>Anytime! If there’s anything else you’re curious about in my portfolio, just ask.</p>",
        ]
        return True, random.choice(responses)

    # ---------- "You're funny" / playful compliments ----------
    FUNNY_PATTERNS = (
        "you are funny", "you're funny", "youre funny",
        "u r funny", "ur funny", "u funny", "funny lol"
    )

    if any(p in q_lower for p in FUNNY_PATTERNS):
        msg = _pick_nonrepeating_session(
            "compliment_funny",
            [
                "Aww thank you 😄 I try! Let me know if you're curious about anything else in my portfolio",
                "Hehe thanks 😂 If you want, ask me about a project — I’ll keep it quick and clear.",
                "Thank youuu 😌 I'm nicely trained on various aspects of this portfolio too!",
            ],
        )
        return True, f"<p>{msg}</p>"

    # ---------- "How did you learn / self-taught" ----------
    # Put this near the top so it runs before any other returns.

    learn_re = re.search(r"\b(how|where|when)\b.*\blearn(ed)?\b", q_lower)
    self_taught_re = re.search(r"\bself[-\s]?taught\b|\bteach yourself\b|\bon your own\b", q_lower)

    mentions_skill_topic = any(w in q_lower for w in [
        "python", "java", "programming", "coding", "code", "skills", "developer"
    ])

    if (learn_re and mentions_skill_topic) or self_taught_re:
        msg = _pick_nonrepeating_session(
            "skills_learning",
            [
                "Mostly hands-on. University gave me the foundation, and then projects + tutorials + documentation is where my skills really leveled up.",
                "A mix of university + self-learning. I learn best by building real projects, breaking things, and fixing them 😄",
                "I learned through CS courses and a lot of self-teaching — projects, YouTube, official docs, and experimenting until it works.",
            ],
        )
        return True, f"<p>{msg}</p>"

    # ---------- Compliments / positive reactions ----------
    q_norm = " ".join((q_lower or "").split())

    is_question = (
        "?" in q_norm or
        q_norm.startswith(("are ", "is ", "do ", "did ", "can ", "could ", "would ", "will ",
                           "what ", "why ", "how ", "where ", "when ")) or
        q_norm.startswith(("r u ", "r you ", "r ur "))
    )

    if (not is_question) and any(word in q_norm for word in [
        "cool", "nice", "good", "awesome", "great", "impressive",
        "love this", "love it", "so clean", "beautiful", "pretty", "cute",
        "nice portfolio", "good portfolio", "amazing", "wow", "smart"
    ]):
        responses = [
            "<p>Thank you — that honestly means a lot 🧡</p>",
            "<p>Thank youuu! Let me know if I can guide you to something else related to my portfolio :)</p>",
            "<p>Thanks! If you’re curious about anything else, ask me.</p>",
        ]
        return True, random.choice(responses)

    # ---------- Resume / CV requests ----------
    if any(phrase in q_lower for phrase in [
        "can i see your resume", "can i see ur resume",
        "show your resume", "show ur resume",
        "your resume", "ur resume", "resume?",
        "can i see your cv", "can i see ur cv",
        "show your cv", "show ur cv",
        "your cv", "ur cv", "cv?",
        "download resume", "resume link", "cv link"
    ]):
        responses = [
            "<p>My resume contains some private details, so I don’t post it publicly.</p>"
            "<p>If you’d like a copy, please email me at <a href='mailto:kareenazaman@gmail.com'>kareenazaman@gmail.com</a> and I’ll send it right away 😊</p>",

            "<p>I keep my resume private (it has personal contact info).</p>"
            "<p>Email me at <a href='mailto:kareenazaman@gmail.com'>kareenazaman@gmail.com</a> and I’ll share the latest version 😊</p>",

            "<p>I don’t host my resume publicly for privacy reasons.</p>"
            "<p>Just send me a quick email at <a href='mailto:kareenazaman@gmail.com'>kareenazaman@gmail.com</a> and I’ll forward it 😊</p>",

            "<p>Yep — I can share my resume, I just don’t post it online 😊</p>"
            "<p>Please email <a href='mailto:kareenazaman@gmail.com'>kareenazaman@gmail.com</a> and I’ll send it over.</p>",
        ]
        return True, random.choice(responses)

    # ---------- Business / Freelance Inquiries ----------
    if any(phrase in q_lower for phrase in [
            "hire you", "freelance", "build a website", "make a website",
            "build an app", "make an app", "for my business", "for a business",
            "taking clients", "open to work", "can you build", "make me a",
            "create a website", "create an app", "agency"
        ]):
            msg = _pick_nonrepeating_session(
                "business_freelance",
                [
                    "I’m always open to discussing freelance projects! 😊 If you have a website or app in mind for your business, send me the details at <a href='mailto:kareenazaman@gmail.com'>kareenazaman@gmail.com</a> and we can chat.",
                    "I do take on freelance work. Shoot me an email at <a href='mailto:kareenazaman@gmail.com'>kareenazaman@gmail.com</a> with what you're looking to build!",
                    "I'd love to hear about your business idea! The best way to reach me for freelance work is by emailing <a href='mailto:kareenazaman@gmail.com'>kareenazaman@gmail.com</a>."
                ]
            )
            return True, f"<p>{msg}</p>"

    return False, ""


def get_smart_refusal(question: str) -> str:
    """
    Friendly fallback when a question is outside portfolio scope.
    Edge-cases are checked first so rude/compliment/location/etc. don't fall through.
    """
    handled, edge_html = handle_edge_case(question, qa_system=None)
    if handled:
        return edge_html

    q_lower = (question or "").lower()

    if any(word in q_lower for word in ["weather", "time", "news", "stock", "stocks", "sports", "score"]):
        return (
            "<p>I’m only set up to talk about my portfolio, not live data like weather, time, news, or stock prices.</p>"
            "<p>If you want, I can walk you through my projects, tech stack, or studies instead.</p>"
        )

    if any(word in q_lower for word in ["joke", "story", "game", "play", "bored"]):
        return (
            "<p>I’m more of a “show you my work” bot than an entertainment bot 😄</p>"
            "<p>But if you’d like something interesting, I can explain one of my projects in detail.</p>"
        )

    if any(word in q_lower for word in ["movie", "food", "music", "song", "book", "restaurant", "drink", "colour", "color"]):
        return (
            "<p>I don’t really keep personal favourites in this portfolio.</p>"
            "<p>I'm only trained to talk about me professional life 😅</p>"
            "<p>This space is mainly about what I build, the tools I use, and what I’m learning.</p>"
        )

    return (
        "<p>I’m not really sure how to answer that here 😅</p>"
        "<p> But I can definitely talk about my projects, skills, education, and experience if you’d like.</p>"
    )


# ============================================================
# Response Bank - Natural, Varied Responses
# ============================================================
RESPONSE_BANK = {
    "about": {
        "first": [
            (
                "<p>I'm Kareena — a CS student who loves building thoughtful apps and AI tools. "
                "I care a lot about clean code, calm UX, and making things that actually help people.</p>"
                "<p>You can ask me about my projects, tech stack, or what I’m working on now.</p>"
            ),
            (
                "<p>Hey! I’m Kareena. I build Android apps, Flask backends, and experiment with machine learning. "
                "I’m currently studying Computer Science and constantly shipping side projects.</p>"
                "<p>If you’d like, ask me about a specific project or technology I use.</p>"
            ),
            (
                "<p>I’m Kareena Zaman — CS student, app developer, and AI enthusiast. "
                "I like taking ideas from rough notes to working software.</p>"
                "<p>Curious about a project, my skills, or how I work? Just ask.</p>"
            ),
        ],
        "repeat": [
            "<p>Still the same Kareena! 😄 Want me to dive deeper into my background or studies?</p>",
            "<p>I'm still me! Do you want to hear more about my long-term goals or my experience?</p>"
        ],
    },

    "projects": {
        "first": [
            "<div class='projects-gallery'></div>",
            "<p>Here are some of my favourite projects:</p><div class='projects-gallery'></div>",
            "<p>These are the main things I’ve built recently:</p><div class='projects-gallery'></div>",
        ],
        "repeat": [
            "<p>I can show you the projects list again if you'd like!</p><div class='projects-gallery'></div>",
            "<p>Still want to talk projects? Here they are:</p><div class='projects-gallery'></div>"
        ],
    },

    "skills": {
        "first": [
            "<div class='skills-wrap'></div>",
            "<p>Here’s a quick look at the tools and technologies I work with:</p><div class='skills-wrap'></div>",
            "<p>These are the languages and frameworks I’m most comfortable with:</p><div class='skills-wrap'></div>",
        ],
        "repeat": [
            "<p>Here is my tech stack again!</p><div class='skills-wrap'></div>",
        ],
    },

    "contact": {
        "first": [
            (
                "<p>You can reach me here:<br>"
                "📧 <a href='mailto:kareenazaman@gmail.com'>kareenazaman@gmail.com</a><br>"
                "💼 <a href='https://linkedin.com/in/kareena-zaman' target='_blank'>LinkedIn</a></p>"
            ),
            (
                "<p>Best ways to contact me:<br>"
                "📧 <a href='mailto:kareenazaman@gmail.com'>kareenazaman@gmail.com</a><br>"
                "💼 <a href='https://linkedin.com/in/kareena-zaman' target='_blank'>LinkedIn</a></p>"
            ),
        ],
        "repeat": [
            "<p>Here is my info again: <a href='mailto:kareenazaman@gmail.com'>kareenazaman@gmail.com</a></p>",
        ],
    },

    "identity": {
        "first": [
            "<p>I’m Kareena in AI form 🤖 I talk in first person, but everything I say is based on my real projects, skills, and experience.</p>",
            "<p>Think of me as a digital version of Kareena — here to guide you through my work, studies, and what I build.</p>",
        ],
        "repeat": [
            "<p>Still me — Kareena’s AI self. What would you like to explore next?</p>",
            "<p>Yep, it’s still my AI version chatting with you. Ask me about my work, skills, or background.</p>",
            "<p>It’s me again. We can go deeper into projects, experience, or anything you’re curious about.</p>",
        ],
    },

    "personal": {
        "first": [
            (
                "<p>A bit about me personally: I’m originally from Bangladesh 🇧🇩 and now living in BC, Canada. "
                "I balance studying Computer Science, working as an Assistant Manager in retail, and building software projects I care about.</p>"
            ),
            (
                "<p>Personally, I’m a mix of tech, creativity, and a little bit of chaos in between 😄 "
                "I love fashion, content creation, and using tech to make everyday life smoother.</p>"
            ),
        ],
        "repeat": [
            "<p>You already know a little about me. Want to hear more about my routine, interests, or long-term goals?</p>",
            "<p>Still the same Kareena — curious what part of my life you want to know more about?</p>",
        ],
    },

    "location": {
        "first": [
            "<p>I’m based in Kamloops, BC, Canada 🇨🇦 That’s where I study, work, and build most of my projects.</p>",
            "<p>Right now I live in Kamloops, British Columbia. A lot of my ideas come from my life here as a student and worker.</p>",
        ],
        "repeat": [
            "<p>Yep, still in Kamloops, BC, Canada.</p>",
            "<p>Same place — Kamloops, BC. 🌲</p>",
        ],
    },

    "origin": {
        "first": [
            (
                "<p>I'm originally from Bangladesh 🇧🇩</p>"
                "<p>Right now I live in Kamloops, BC, Canada, where I’m studying Computer Science.</p>"
            ),
            (
                "<p>My roots are in Bangladesh 🇧🇩</p>"
                "<p>I moved to Canada for my studies and I’m currently based in Kamloops, BC.</p>"
            ),
        ],
        "repeat": [
            "<p>Yep — I’m originally from Bangladesh 🇧🇩</p>",
            "<p>Still Bangladeshi at heart, just studying in Canada now 💻🇨🇦</p>",
        ],
    },

    "study": {
        "first": [
            (
                "<p>I study Computer Science at Thompson Rivers University. "
                "My interests include software development, Android apps, AI/ML, and backend systems.</p>"
            ),
            (
                "<p>I’m doing my BSc in Computer Science at TRU. I’ve taken courses on data structures, algorithms, mobile dev, "
                "networks & security, and more.</p>"
            ),
        ],
        "repeat": [
            "<p>Still studying CS at TRU 😊 Want to know about courses, projects, or what I focus on?</p>",
            "<p>Yep, Computer Science at TRU. Ask if you want specifics about what I’ve learned or built there.</p>",
        ],
    },

    "experience": {
        "first": [
            (
                "<p>My experience is a mix of technical and real-world: I’ve built Android apps, Flask APIs, and ML models, "
                "and I work as an Assistant Manager at Suzanne’s where I handle leadership, operations, and problem-solving every day.</p>"
            ),
            (
                "<p>On the tech side, I’ve worked on AI tools, Android apps, and full-stack projects. "
                "On the people side, I’ve led teams and managed a clothing store as an Assistant Manager.</p>"
            ),
        ],
        "repeat": [
            "<p>Happy to go deeper — are you more curious about my technical experience or my leadership/management side?</p>",
            "<p>We’ve touched on my experience already. Tell me whether you want details on a specific project, role, or skill.</p>",
        ],
    },

    "followup": {
        "generic": [
            "<p>Sure! What part should I expand on?</p>",
            "<p>Happy to explain more — which detail are you curious about?</p>",
            "<p>Absolutely. Tell me what you want more clarity on.</p>",
            "<p>I can go deeper. What should I focus on — the tech, the idea, or the impact?</p>",
        ],
        "contextual": {
            "projects": [
                "<p>Do you want to know the tech stack, the problem it solves, or how I built it step by step?</p>",
                "<p>I can talk about the architecture, challenges, or what I’d improve next — your choice.</p>",
                "<p>We can go into features, design decisions, or technical details. What sounds good?</p>",
            ],
            "skills": [
                "<p>Is there a specific language or framework you want me to focus on?</p>",
                "<p>I can share how I’ve used any of these skills in real projects if that helps.</p>",
                "<p>Pick a skill (like Flask, Android, or ML) and I’ll explain how I use it.</p>",
            ],
            "about": [
                "<p>Want to know more about my education, background, or future goals?</p>",
                "<p>I can talk about how I got into CS, what motivates me, or what I’m aiming for next.</p>",
            ],
            "origin": [
                "<p>Want to hear more about my journey from Bangladesh to studying in Canada?</p>",
                "<p>I can tell you how studying CS in Canada has been compared to growing up in Bangladesh.</p>",
            ],
        },
    },

    "greeting": [
        "<p>Hey! 👋 I’m Kareena. Want to explore my projects, skills, or learn a bit about me?</p>",
        "<p>Hi! I’m Kareena 👋 You can ask about my projects, what I study, or what I’ve built.</p>",
        "<p>Hello! I'm Kareena 👋 We can talk about my work, my tech stack, or a lil bit about me — you choose.</p>",
        "<p>Hi and welcome! I’m Kareena. Where should we start? 😃</p>",
    ],

    "farewell": [
        "<p>Thanks for spending time with my portfolio ✨ Feel free to come back anytime.</p>",
        "<p>Glad we could chat! If you want to continue the conversation, my email and LinkedIn are open.</p>",
        "<p>Thanks for visiting 🧡 Hope you found what you were looking for.</p>",
        "<p>Appreciate you checking out my work! Don’t hesitate to reach out if something caught your interest.</p>",
        "<p>See you later! 👋</p>",
        "<p>Thanks for stopping by — have a great day!</p>",
    ],
}


def get_response(intent: str, is_repeat: bool, last_intent: str = None) -> str:
    """Returns a varied response based on intent + whether it's already been shown."""
    if intent == "followup":
        if last_intent and last_intent in RESPONSE_BANK["followup"]["contextual"]:
            return random.choice(RESPONSE_BANK["followup"]["contextual"][last_intent])
        return random.choice(RESPONSE_BANK["followup"]["generic"])

    if intent == "greeting":
        return random.choice(RESPONSE_BANK["greeting"])

    if intent == "farewell" and not is_repeat:
        return random.choice(RESPONSE_BANK["farewell"])

    if intent not in RESPONSE_BANK:
        return get_smart_refusal("")

    responses = RESPONSE_BANK[intent]
    key = "repeat" if is_repeat else "first"
    return random.choice(responses[key])


# ============================================================
# Retrieval QA (TF-IDF + cosine)
# ============================================================
IN_SCOPE_KEYWORDS = (
    "kareena", "portfolio", "project", "projects", "skills", "experience",
    "contact", "resume", "cv", "android", "flask", "mapbox", "site", "website",
    "aqi", "smoke", "wildfire", "nasa", "tempo", "about", "github", "linkedin",
    "python", "java", "javascript", "built", "developed", "work", "tech",
    "bc", "british columbia", "canada", "kamloops", "location", "where", "based",
    "bangladesh", "bd"
)


def _pick_nonrepeating_session(key: str, options: list[str]) -> str:
    """Same reply variety idea, but stored at the root session key."""
    last = session.get(key)
    pool = [o for o in options if o != last] or options
    out = random.choice(pool)
    session[key] = out
    session.modified = True
    return out


def format_text_to_html(text: str) -> str:
    if not text:
        return ""

    t = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()

    # Turn inline separators into bullet lines SAFELY
    # Only split if there is a clear space before AND after the hyphen
    t = re.sub(r"\s+-\s+(?=[A-Z])", "\n- ", t)

    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]

    html_parts = []
    lis = []

    for line in lines:
        line = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", line)

        if line.startswith("- "):
            lis.append(f"<li>{line[2:].strip()}</li>")
        else:
            if re.match(r"^[A-Za-z][A-Za-z &/]+:\s*$", line):
                html_parts.append(f"<div class='ai-section-title'>{line[:-1].strip()}</div>")
            else:
                html_parts.append(f"<div class='ai-line'>{line}</div>")

    if lis:
        html_parts.append("<ul class='ai-list'>" + "".join(lis) + "</ul>")

    return "".join(html_parts)


class KareenaQA:
    def __init__(self, corpus_items: List[Dict[str, Any]], thresh: float = 0.28) -> None:
        self.thresh = thresh
        self.docs = corpus_items[:]
        self.project_docs = []
        self.project_title_map = {}

        for d in self.docs:
            title = (d.get("title") or "").strip()
            if not title:
                continue
            if str(d.get("id", "")).startswith("project_"):
                self.project_docs.append(d)
                self.project_title_map[title.lower()] = d

        self._build_index()

    def _build_index(self):
        INDEX_DIR.mkdir(exist_ok=True, parents=True)

        # If cached index exists, load it
        if VEC_PATH.exists() and MAT_PATH.exists() and DOCS_PATH.exists():
            self.vectorizer = joblib.load(VEC_PATH)
            self.doc_mat = joblib.load(MAT_PATH)
            self.docs = joblib.load(DOCS_PATH)
            self.corpus_texts = [d["content"] for d in self.docs] or ["(empty)"]
            return

        # Otherwise build and save
        self.corpus_texts = [d["content"] for d in self.docs] or ["(empty)"]
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            lowercase=True,
            min_df=1
        )
        self.doc_mat = self.vectorizer.fit_transform(self.corpus_texts)

        joblib.dump(self.vectorizer, VEC_PATH)
        joblib.dump(self.doc_mat, MAT_PATH)
        joblib.dump(self.docs, DOCS_PATH)

    def reload(self):
        self.docs = build_corpus_from_portfolio()

        # 🔥 force rebuild: delete cached index files
        for p in (VEC_PATH, MAT_PATH, DOCS_PATH):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

        self._build_index()

    def _in_scope(self, q: str) -> bool:
        ql = q.lower()
        return any(k in ql for k in IN_SCOPE_KEYWORDS)

    def answer(self, query: str, top_k: int = 1) -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {"ok": False, "html": "<p>Please type a question.</p>"}

        # Edge cases first (pass self for KB access where needed)
        handled, edge_response = handle_edge_case(query, qa_system=self)
        if handled:
            return {"ok": True, "html": edge_response}

        ql = query.lower()

        # Force common "tech stack" questions to prefer skills chunks
        if any(w in ql for w in
               ("backend", "frontend", "tech stack", "stack", "tools", "framework", "database", "api")):
            best_sk = None
            best_sk_score = -1.0

            qv = self.vectorizer.transform([query])
            sims = cosine_similarity(qv, self.doc_mat)[0]

            for i, d in enumerate(self.docs):
                if str(d.get("id", "")).startswith("skills_"):
                    s = float(sims[i])
                    if s > best_sk_score:
                        best_sk_score = s
                        best_sk = d

            if best_sk and best_sk_score >= 0.08:  # lower mini-threshold just for skills
                return {"ok": True, "html": format_text_to_html(best_sk["content"])}

        triggers = ("tell me more about", "tell me about", "more about", "explain", "describe", "details on")
        proj_q = None
        for t in triggers:
            if t in ql:
                proj_q = ql.split(t, 1)[1].strip(" ?.!\"'")
                break

        if proj_q:
            if proj_q in self.project_title_map:
                d = self.project_title_map[proj_q]
                return {"ok": True, "html": format_text_to_html(d["content"])}

            pq = proj_q.replace(" ", "")
            for title_l, d in self.project_title_map.items():
                # FIX: Require the search term to be at least 3 characters long
                # before doing a loose substring match!
                if pq == title_l.replace(" ", "") or (len(proj_q) >= 3 and proj_q in title_l):
                    return {"ok": True, "html": format_text_to_html(d["content"])}

        # Recruiter / employer prompts (rotate answers)
        for key, patterns in EMPLOYER_PATTERNS.items():
            if any(p in ql for p in patterns):
                msg = _pick_nonrepeating_session(f"employer_{key}", EMPLOYER_ANSWERS[key])
                return {"ok": True, "html": f"<p>{msg}</p>"}

        # Judgment prompts (rotate answers)
        for key, patterns in JUDGMENT_PATTERNS.items():
            if any(p in ql for p in patterns):
                msg = _pick_nonrepeating_session(f"judge_{key}", JUDGMENT_ANSWERS[key])
                return {"ok": True, "html": f"<p>{msg}</p>"}

        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.doc_mat)[0]
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])

        if best_score < self.thresh:
            return {"ok": False, "html": get_smart_refusal(query)}

        ranked = sorted(
            [{"doc": self.docs[i], "score": float(sims[i])} for i in range(len(sims))],
            key=lambda x: x["score"], reverse=True
        )[:top_k]

        content = ranked[0]["doc"]["content"]

        def extract_section(text: str, header: str) -> str:
            lines = text.splitlines()
            out = []
            capture = False

            for line in lines:
                if line.strip().lower().startswith(f"## {header.lower()}"):
                    capture = True
                    continue
                if capture and line.startswith("## "):
                    break
                if capture:
                    out.append(line)

            return "\n".join(out).strip()

        # Section-aware responses
        if "build" in ql:
            section = extract_section(content, "What I Build")
            if section:
                return {"ok": True, "html": format_text_to_html(section)}

        if any(k in ql for k in ("skill", "language", "tool", "tech")):
            section = extract_section(content, "Technical Skills")
            if section:
                return {"ok": True, "html": format_text_to_html(section)}

        # Fallback: first paragraph only
        first_para = content.split("\n\n")[0]
        return {"ok": True, "html": format_text_to_html(first_para)}


# Initialize retrieval
qa = KareenaQA(build_corpus_from_portfolio(), thresh=0.15)

# ============================================================
# Intent classifier + session memory
# ============================================================
INTENT_MODEL_PATH = BASE_DIR / "models" / "intent_pipe.joblib"
INTENT_PIPE = None
try:
    INTENT_PIPE = joblib.load(INTENT_MODEL_PATH)
except FileNotFoundError:
    print(f"WARNING: intent model missing at {INTENT_MODEL_PATH} — running without ML intent model.")

INTENT_THRESH = float(os.getenv("INTENT_THRESH", "0.55"))


def _sess():
    """Session state for chat flow."""
    s = session.setdefault("pf", {})
    s.setdefault("shown", {
        "about": False,
        "projects": False,
        "skills": False,
        "contact": False,
        "identity": False,
        "origin": False,
    })
    s.setdefault("last_intent", None)
    s.setdefault("last_query", "")
    s.setdefault("conversation_turn", 0)
    s.setdefault("last_time", 0)
    return s


def route_intent(text: str):
    """Use trained pipeline to return (LABEL, CONFIDENCE)."""
    q = (text or "").strip().lower()

    if INTENT_PIPE is None:
        return "fallback", 0.0

    proba = INTENT_PIPE.predict_proba([q])[0]
    labels = INTENT_PIPE.classes_
    i = int(proba.argmax())
    return labels[i].upper(), float(proba[i])


def is_followup_question(question: str, last_query: str, last_intent: str) -> bool:
    """Detect if a question is likely referring to the previous turn."""
    if not last_intent or not last_query:
        return False

    q_lower = question.lower()

    followup_phrases = [
        "tell me more", "more about", "elaborate", "explain", "what about",
        "how about", "details", "specifically", "which one", "that one",
        "go on", "continue", "and", "also"
    ]

    pronouns = ["it", "that", "this", "them", "those", "these"]

    is_short = len(question.split()) <= 5
    has_pronoun = any(p in q_lower.split() for p in pronouns)
    has_followup = any(phrase in q_lower for phrase in followup_phrases)

    return (is_short and has_pronoun) or has_followup


def log_query(q: str, intent: str, score: float, outcome: str):
    """Append JSONL line for dataset growth."""
    try:
        logs = BASE_DIR / "logs"
        logs.mkdir(exist_ok=True, parents=True)
        path = logs / "intent_infer.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "t": int(time.time()),
                "q": q,
                "intent": intent,
                "score": round(score, 4),
                "outcome": outcome
            }) + "\n")
    except Exception:
        pass


# ============================================================
# Routes
# ============================================================
@app.route("/")
def home():
    # Clear the session so the bot "forgets" previous turns on refresh
    session.pop("pf", None)
    projects = load_yaml("projects.yml")
    skills = load_yaml("skills.yml")

    chat_mode = (request.args.get("chat") == "1")

    return render_template(
        "index.html",
        name="Kareena",
        projects=projects,
        skills=skills,
        chat_mode=chat_mode
    )


@app.route("/chat")
def chat_page():
    projects = load_yaml("projects.yml")
    skills = load_yaml("skills.yml")
    return render_template("index.html", name="Kareena", projects=projects, skills=skills)


# ---------- Chat: non-stream ----------
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    question = auto_align_words(question)

    if not question:
        return jsonify({"html": "<p>Please type a question.</p>"}), 400

    # 1) Handle special cases first (small talk, identity, rude, etc.)
    handled, edge_html = handle_edge_case(question, qa_system=qa)
    if handled:
        log_query(question, "edge_case", 1.0, "edge_case")
        return jsonify({"html": edge_html, "ok": True})

    res = qa.answer(question, top_k=3)
    if res["ok"]:
        return jsonify({"html": res["html"], "ok": True})

    # Then normal flow
    s = _sess()
    s["conversation_turn"] += 1

    intent, score = route_intent(question)

    # GREETING → ABOUT on first turn
    if intent == "GREETING" and s["conversation_turn"] <= 1:
        intent = "ABOUT"
        score = 0.9

    # Smart follow-up detection
    if intent == "FOLLOWUP" or is_followup_question(question, s["last_query"], s["last_intent"]):
        if score < INTENT_THRESH:
            score = 0.75
        intent = "FOLLOWUP"

    if score >= INTENT_THRESH:
        intent_lower = intent.lower()
        print(f"DEBUG STREAM passed: {intent_lower}")

        if intent_lower == "offtopic":
            html_response = get_smart_refusal(question)
            log_query(question, intent, score, "offtopic_refusal")
            return jsonify({"html": html_response, "ok": False})

        is_repeat = s["shown"].get(intent_lower, False)

        # For skills intent, try TF-IDF first, fall back to widget
        if intent_lower == "skills":
            skills_res = qa.answer(question, top_k=1)
            if skills_res["ok"]:
                html_response = skills_res["html"]
            else:
                # lower threshold retry for skills
                qa.thresh = 0.05
                skills_res = qa.answer(question, top_k=1)
                qa.thresh = 0.20  # restore original
                if skills_res["ok"]:
                    html_response = skills_res["html"]
                else:
                    html_response = get_response(intent_lower, is_repeat, s.get("last_intent"))
        else:
            html_response = get_response(intent_lower, is_repeat, s.get("last_intent"))

        if intent_lower in s["shown"]:
            s["shown"][intent_lower] = True
        s["last_intent"] = intent_lower
        s["last_query"] = question
        s["last_time"] = time.time()
        session.modified = True

        log_query(question, intent, score, intent_lower)
        return jsonify({"html": html_response, "ok": True})

    # Low confidence → try retrieval again
    res = qa.answer(question)
    outcome = "retrieval_ok" if res["ok"] else "refuse_lowconf"
    log_query(question, f"{intent}@{score:.2f}", score, outcome)


    s["last_query"] = question
    s["last_time"] = time.time()
    session.modified = True

    return jsonify({"html": res["html"], "ok": res["ok"], "sources": res.get("sources", [])})


# ---------- Chat: "stream" version ----------
@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    question = auto_align_words(question)


    if not question:
        return Response("Please type a question.", mimetype="text/plain")

    # 1) Edge-cases first, same as /api/chat
    handled, edge_html = handle_edge_case(question, qa_system=qa)
    if handled:
        log_query(question, "edge_case", 1.0, "edge_case")
        return Response(edge_html, mimetype="text/plain")

    res = qa.answer(question, top_k=3)
    print(f"DEBUG qa: ok={res['ok']} html={res['html'][:80]}")
    if res["ok"]:
        return Response(res["html"], mimetype="text/plain")

    s = _sess()
    s["conversation_turn"] += 1

    intent, score = route_intent(question)

    if intent == "GREETING" and s["conversation_turn"] <= 1:
        intent = "ABOUT"
        score = 0.9

    if intent == "FOLLOWUP" or is_followup_question(question, s["last_query"], s["last_intent"]):
        if score < INTENT_THRESH:
            score = 0.75
        intent = "FOLLOWUP"

    if score >= INTENT_THRESH:
        intent_lower = intent.lower()

        if intent_lower == "offtopic":
            html_response = get_smart_refusal(question)
            log_query(question, intent, score, "offtopic_refusal")
            return Response(html_response, mimetype="text/plain")

        is_repeat = s["shown"].get(intent_lower, False)

        # For skills intent, try TF-IDF first, fall back to widget
        if intent_lower == "skills":
            skills_res = qa.answer(question, top_k=1)
            if skills_res["ok"]:
                html_response = skills_res["html"]
            else:
                # lower threshold retry for skills
                qa.thresh = 0.05
                skills_res = qa.answer(question, top_k=1)
                qa.thresh = 0.20  # restore original
                if skills_res["ok"]:
                    html_response = skills_res["html"]
                else:
                    html_response = get_response(intent_lower, is_repeat, s.get("last_intent"))
        else:
            html_response = get_response(intent_lower, is_repeat, s.get("last_intent"))

        if intent_lower in s["shown"]:
            s["shown"][intent_lower] = True
        s["last_intent"] = intent_lower
        s["last_query"] = question
        s["last_time"] = time.time()
        session.modified = True

        log_query(question, intent, score, intent_lower)
        return Response(html_response, mimetype="text/plain")

    res = qa.answer(question)
    outcome = "retrieval_ok" if res["ok"] else "refuse_lowconf"
    log_query(question, f"{intent}@{score:.2f}", score, outcome)

    s["last_query"] = question
    s["last_time"] = time.time()
    session.modified = True

    return Response(res["html"], mimetype="text/plain")


# ---------- Reindex retrieval corpus ----------
@app.route("/api/reindex", methods=["POST"])
def api_reindex():
    qa.reload()
    return jsonify({"ok": True, "count": len(qa.docs)})


# ---------- Dev server ----------
if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))

    print("=== Startup ===")
    print("BASE_DIR:", BASE_DIR)
    try:
        print("INTENTS:", list(INTENT_PIPE.classes_))
    except Exception:
        print("INTENTS: (unavailable)")
    print("KB docs:", len(qa.docs))
    app.run(host=host, port=port, debug=True, use_reloader=False)