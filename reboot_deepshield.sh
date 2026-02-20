#!/bin/bash
echo "🚀 DeepShield: Initiating Hard Reboot (Stable Forensic V5)..."

# Stopping existing instances
echo "🛑 Stopping existing instances..."
pkill -f "uvicorn main:app" || true

# Moving to backend directory
cd "$(dirname "$0")/backend"

# Starting in background
echo "🔥 Starting Backend (Universal Image Compatibility)..."
source venv/bin/activate
nohup python -m uvicorn main:app --host 0.0.0.0 --port 8000 > deepshield_backend.log 2>&1 &

echo "✅ Backend is starting in the background (see deepshield_backend.log)"
echo "⚠️ IMPORTANT: Now go to chrome://extensions and RELOAD DeepShield to sync."
