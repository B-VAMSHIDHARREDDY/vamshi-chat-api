# Vamshi Portfolio Chat API 🤖

> AI-powered chat API for [vamshi.site](https://vamshi.site) — built with FastAPI, with a smart fallback chain across Gemini AI → ChatGPT → HuggingFace.

---

## ✨ Features

- **FastAPI** — modern, async, production-ready Python web framework
- **Gemini AI** as the primary provider (`gemini-1.5-flash`)
- **ChatGPT (OpenAI)** as the first fallback
- **HuggingFace Inference API** as a free second fallback
- **Automatic fallback chain** — if a provider hits its rate limit or errors, the next one is tried seamlessly
- **CORS configured** for `vamshi.site`
- **No database required**
- **Render-ready** (`render.yaml` + `Procfile` included)
- **OpenAPI docs** at `/docs`

---

## 🗂 Project Structure

```
vamshi-chat-api/
├── app/
│   ├── main.py                  # FastAPI app, CORS, routes
│   ├── config.py                # Settings from env vars
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   ├── routers/
│   │   └── chat.py              # /api/v1/chat endpoints
│   └── services/
│       ├── ai_orchestrator.py   # Fallback chain logic
│       ├── gemini_service.py    # Gemini AI integration
│       ├── openai_service.py    # OpenAI ChatGPT integration
│       └── huggingface_service.py # HuggingFace free fallback
├── requirements.txt
├── render.yaml                  # Render deployment config
├── Procfile
├── .env.example
└── .gitignore
```

---

## 🚀 Local Development

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/vamshi-chat-api.git
cd vamshi-chat-api
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your real API keys
```

### 3. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

Visit:
- API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

---

## 📡 API Endpoints

### `POST /api/v1/chat`

Send a message to the AI assistant.

**Request body:**
```json
{
  "message": "Tell me about your projects",
  "conversation_history": [
    { "role": "user", "content": "Hi!" },
    { "role": "assistant", "content": "Hello! How can I help?" }
  ]
}
```

**Response:**
```json
{
  "reply": "I have worked on several exciting projects...",
  "provider": "gemini",
  "model": "gemini-1.5-flash",
  "success": true
}
```

### `GET /api/v1/chat/providers`

Lists all configured AI providers and their status.

### `GET /health`

Health check endpoint (used by Render).

---

## 🌐 Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Deploy**
5. In **Environment** tab, add your secret API keys:
   - `GEMINI_API_KEY`
   - `OPENAI_API_KEY`
   - `HUGGINGFACE_API_KEY`

Your API will be live at `https://vamshi-chat-api.onrender.com` (or your custom domain).

---

## 🔑 Getting API Keys

| Provider | Free Tier | Link |
|----------|-----------|------|
| **Gemini AI** | Yes (generous) | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| **OpenAI** | $5 credit on signup | [platform.openai.com](https://platform.openai.com/api-keys) |
| **HuggingFace** | Yes (free inference) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

---

## 🔗 Using in Your Frontend (vamshi.site)

```javascript
const response = await fetch("https://vamshi-chat-api.onrender.com/api/v1/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    message: userInput,
    conversation_history: chatHistory,
  }),
});
const data = await response.json();
console.log(data.reply); // AI response
```

---

## 🛠 Fallback Logic

```
User request
    │
    ▼
Gemini AI ──quota/error──► ChatGPT ──quota/error──► HuggingFace
    │                           │                        │
    └──── success ──────────────┴──── success ───────────┘
                                                    │
                                              All failed?
                                            HTTP 503 returned
```

---

## 📄 License

MIT — feel free to use and adapt for your own portfolio.
