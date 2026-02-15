#!/bin/bash
# Automatic Installation Script for Length Penalty Modification

echo "=========================================="
echo "🚀 Automatic Length Penalty Installer"
echo "=========================================="
echo ""

# Check if Python script exists
if [ ! -f "apply_patch.py" ]; then
    echo "❌ ERROR: apply_patch.py not found!"
    echo "Please run this script from the directory containing all downloaded files."
    exit 1
fi

echo "Starting automatic installation..."
echo ""

# Run the Python patcher
python3 apply_patch.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ INSTALLATION SUCCESSFUL!"
    echo "=========================================="
    echo ""
    echo "🎉 All modifications applied automatically!"
    echo ""
    echo "📋 What was done:"
    echo "  1. ✅ Copied length_penalty_processor.py to qwen_tts/inference/"
    echo "  2. ✅ Added import statement to qwen3_tts_model.py"
    echo "  3. ✅ Modified generate_voice_clone() signature"
    echo "  4. ✅ Added length penalty logic"
    echo "  5. ✅ Created backup of original file"
    echo ""
    echo "🧪 Next steps:"
    echo "  1. Edit test_length_penalty.py:"
    echo "     Set REF_AUDIO = \"/path/to/your/audio.wav\""
    echo ""
    echo "  2. Run the test:"
    echo "     python3 test_length_penalty.py"
    echo ""
    echo "  3. Listen to generated clips and choose optimal alpha value"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ INSTALLATION FAILED"
    echo "=========================================="
    echo ""
    echo "The automatic patcher encountered errors."
    echo ""
    echo "🔧 Fallback options:"
    echo ""
    echo "Option 1 - Review error messages above and:"
    echo "  • Ensure you're running from the correct directory"
    echo "  • Check that all required files are present"
    echo "  • Try running: python3 apply_patch.py"
    echo ""
    echo "Option 2 - Manual installation:"
    echo "  1. Read: qwen3_tts_model_PATCH.txt"
    echo "  2. See: PATCH_EXAMPLE.txt for detailed examples"
    echo "  3. Apply changes manually with a text editor"
    echo ""
    exit 1
fi
