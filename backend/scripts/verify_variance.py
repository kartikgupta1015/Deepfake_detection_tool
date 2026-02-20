import requests
import os
import time

BACKEND_URL = "http://localhost:8000"

def test_image_variance():
    # Find 3 different images
    sample_images = [
        "/home/yashbansal/deepfake/backend/venv/lib/python3.12/site-packages/matplotlib/mpl-data/sample_data/Minduka_Present_Blue_Pack.png",
        "/home/yashbansal/deepfake/backend/venv/lib/python3.12/site-packages/matplotlib/mpl-data/images/matplotlib.png",
        "/home/yashbansal/deepfake/backend/venv/lib/python3.12/site-packages/scipy/ndimage/tests/dots.png"
    ]
    
    scores = []
    
    print("🧪 Starting Variance Test...")
    for img_path in sample_images:
        if not os.path.exists(img_path):
            print(f"❌ Missing test image: {img_path}")
            continue
            
        with open(img_path, "rb") as f:
            files = {"file": (os.path.basename(img_path), f, "image/png")}
            resp = requests.post(f"{BACKEND_URL}/detect-image", files=files)
            
            if resp.status_code == 200:
                data = resp.json()
                score = data["authenticity_score"]
                scores.append(score)
                print(f"✅ Image: {os.path.basename(img_path)} -> Score: {score}%")
            else:
                print(f"❌ Failed to scan {img_path}: {resp.text}")

    if len(scores) < 2:
        print("❌ Not enough scores to compare variance.")
        return

    # Check for variance
    distinct_scores = len(set(scores))
    if distinct_scores > 1:
        print(f"🎉 SUCCESS! Found {distinct_scores} distinct scores for {len(scores)} images.")
        # Print the difference
        diff = max(scores) - min(scores)
        print(f"📊 Range of scores: {diff:.4f}%")
    else:
        print("🚨 FAILURE: All images returned the EXACT SAME score!")
        print(f"💀 Constant Score: {scores[0]}%")

if __name__ == "__main__":
    # Wait for backend to be ready
    health_ok = False
    for _ in range(5):
        try:
            r = requests.get(f"{BACKEND_URL}/health")
            if r.status_code == 200:
                health_ok = True
                break
        except:
            pass
        time.sleep(2)
        
    if health_ok:
        test_image_variance()
    else:
        print("❌ Backend not reachable on port 8000")
