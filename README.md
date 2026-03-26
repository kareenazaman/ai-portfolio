# 🤖 AI-Powered Personal Portfolio

An intelligent, interactive personal portfolio website featuring a custom-built AI chatbot that speaks in the first person. 

Instead of a static resume, visitors and recruiters can have a natural conversation with my "AI Twin" to learn about my projects, technical skills, and experience.

**[👉 View the Live Demo Here](https://kareenazaman.com)**

---

## ✨ Key Features

* **Custom NLP Pipeline:** Built from scratch without relying on external LLM APIs (like OpenAI). 
* **Intent Classification:** Uses a custom-trained Machine Learning model (`scikit-learn` Logistic Regression) with character n-grams to detect user intent and gracefully handle typos or internet shorthand.
* **Context-Aware Memory:** The bot remembers the current conversational context, allowing users to ask natural follow-up questions (e.g., *"tell me more about that"*).
* **TF-IDF Search Engine:** Fallback retrieval system using Cosine Similarity to scan my markdown-based knowledge base and answer hyper-specific questions about my tech stack and background.
* **Dynamic UI/UX:** Features a custom Vanilla JS frontend with Apple-inspired FLIP animations, responsive glassmorphism design, and streaming "typing" effects.

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **Machine Learning:** `scikit-learn`, `joblib`, TF-IDF Vectorization, Logistic Regression
* **Frontend:** JavaScript (Vanilla), HTML5, CSS3 
* **Data & Content:** YAML, Markdown
* **Deployment:** Ubuntu VPS, Nginx, Gunicorn

## 🧠 How the AI Works

1.  **Input Sanitization:** User input is normalized to catch internet slang (e.g., "u" -> "you").
2.  **Intent Routing:** The input is passed through a trained ML pipeline. If the confidence score is high, it routes to a specific conversational flow (e.g., greetings, business inquiries, or specific section triggers).
3.  **Vector Retrieval:** If the query is highly specific (e.g., *"What did you use for your Android app?"*), it bypasses the intent model and is vectorized against a TF-IDF matrix of my resume/projects, returning the most relevant markdown chunk via cosine similarity.

## 🚀 Running Locally
