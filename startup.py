from image_embedder import warmup

def start_ai():
    print("🚀 Starting AI system...")

    try:
        warmup()
        print("✅ Image AI Ready")
    except Exception as e:
        print("❌ AI startup failed:", e)

    print("✅ Server ready")