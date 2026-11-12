#!/usr/bin/env bash
set -e

PLATFORM="/Users/sivanesanbalu/Library/Android/sdk/platforms/android-35/android.jar"
SDK_BT="/Users/sivanesanbalu/Library/Android/sdk/build-tools/35.0.0"

cd /Users/sivanesanbalu/.gemini/antigravity/scratch/Book_AI_Backend/android
rm -rf bin && mkdir -p bin

echo "Step 1: Generating R.java"
"${SDK_BT}/aapt" package -m -J src -M AndroidManifest.xml -S res -I "${PLATFORM}"

echo "Step 2: Compiling Java source (target 17)"
javac -source 17 -target 17 -d bin -cp "${PLATFORM}:src" src/com/booknest/ai/*.java

echo "Step 3: DEXing bytecode"
"${SDK_BT}/d8" --lib "${PLATFORM}" --output bin/ $(find bin/com/booknest/ai -name "*.class")

echo "Step 4: Packaging APK"
"${SDK_BT}/aapt" package -f -M AndroidManifest.xml -S res -I "${PLATFORM}" -F bin/unaligned.apk bin/

echo "Step 5: Zipalign"
"${SDK_BT}/zipalign" -f -p 4 bin/unaligned.apk bin/aligned.apk

echo "Step 6: Signing APK"
"${SDK_BT}/apksigner" sign --ks ~/.android/debug.keystore --ks-pass pass:android --key-pass pass:android --out ../static/BookNestAI-release.apk bin/aligned.apk

cp ../static/BookNestAI-release.apk ../static/BookNestAI.apk
echo "Build complete! APK size:"
ls -lh ../static/BookNestAI-release.apk
