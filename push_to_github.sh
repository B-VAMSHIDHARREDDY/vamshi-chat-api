#!/bin/bash
# ============================================================
# Push vamshi-chat-api to GitHub
# Run this script ONCE after cloning this project locally.
# ============================================================

set -e

REPO_NAME="vamshi-chat-api"
GITHUB_USERNAME="YOUR_GITHUB_USERNAME"  # <-- Change this!

echo "🔧 Creating GitHub repo and pushing..."

# 1. Create the repo on GitHub (requires gh CLI installed locally)
gh repo create "$REPO_NAME" \
  --public \
  --description "AI-powered chat API for vamshi.site — FastAPI + Gemini + ChatGPT + HuggingFace" \
  --source=. \
  --remote=origin \
  --push

echo ""
echo "✅ Done! Repo pushed to: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo ""
echo "👉 Next steps:"
echo "  1. Go to https://render.com"
echo "  2. New → Web Service → connect your GitHub repo"
echo "  3. Add your API keys in the Environment tab:"
echo "     - GEMINI_API_KEY"
echo "     - OPENAI_API_KEY"
echo "     - HUGGINGFACE_API_KEY"
echo "  4. Deploy!"
