from image_embedder import warmup

def startup_event():
    print("🚀 Starting AI system...")

    try:
        warmup()
        print("✅ Image AI Ready")
    except Exception as e:
        print("❌ AI startup failed:", e)

    print("✅ Server ready")