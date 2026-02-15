#!/bin/bash
# Automatic Installation Script for Working Length Penalty
set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     LENGTH PENALTY INSTALLATION (BUG FIXED VERSION)          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Find qwen_tts installation
echo "Step 1: Locating qwen_tts installation..."
QWEN_PATH=$(python3 -c "import qwen_tts; import os; print(os.path.dirname(qwen_tts.__file__))" 2>/dev/null)

if [ -z "$QWEN_PATH" ]; then
    echo -e "${RED}❌ ERROR: qwen_tts not installed!${NC}"
    echo "Install it with: pip install qwen-tts"
    exit 1
fi

echo -e "${GREEN}✅ Found at: $QWEN_PATH${NC}"
echo ""

# Check if files exist
echo "Step 2: Checking required files..."
REQUIRED_FILES=("length_penalty_processor.py" "qwen3_tts_model_MODIFIED.py" "modeling_qwen3_tts_MODIFIED.py")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}❌ ERROR: $file not found!${NC}"
        echo "Please run this script from the directory containing all downloaded files."
        exit 1
    fi
    echo -e "${GREEN}  ✓${NC} $file"
done
echo ""

# Create backups
echo "Step 3: Creating backups..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cp "$QWEN_PATH/inference/qwen3_tts_model.py" \
   "$QWEN_PATH/inference/qwen3_tts_model.py.backup_$TIMESTAMP"
echo -e "${GREEN}  ✓${NC} qwen3_tts_model.py.backup_$TIMESTAMP"

cp "$QWEN_PATH/core/models/modeling_qwen3_tts.py" \
   "$QWEN_PATH/core/models/modeling_qwen3_tts.py.backup_$TIMESTAMP"
echo -e "${GREEN}  ✓${NC} modeling_qwen3_tts.py.backup_$TIMESTAMP"
echo ""

# Install files
echo "Step 4: Installing modified files..."

# 1. length_penalty_processor.py
cp length_penalty_processor.py "$QWEN_PATH/inference/"
echo -e "${GREEN}  ✓${NC} length_penalty_processor.py → inference/"

# 2. qwen3_tts_model.py
cp qwen3_tts_model_MODIFIED.py "$QWEN_PATH/inference/qwen3_tts_model.py"
echo -e "${GREEN}  ✓${NC} qwen3_tts_model.py (wrapper)"

# 3. modeling_qwen3_tts.py (THE CRITICAL FIX!)
cp modeling_qwen3_tts_MODIFIED.py "$QWEN_PATH/core/models/modeling_qwen3_tts.py"
echo -e "${GREEN}  ✓${NC} modeling_qwen3_tts.py (CORE FIX)"
echo ""

# Verify installation
echo "Step 5: Verifying installation..."

python3 << 'PYEOF'
import sys

# Check 1: Wrapper parameters
try:
    from qwen_tts import Qwen3TTSModel
    import inspect
    sig = inspect.signature(Qwen3TTSModel.generate_voice_clone)
    params = list(sig.parameters.keys())
    
    if 'length_penalty_alpha' in params and 'frames_per_text_token' in params:
        print("  ✅ Wrapper parameters OK")
    else:
        print("  ❌ Wrapper parameters missing")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ Wrapper check failed: {e}")
    sys.exit(1)

# Check 2: Processor import
try:
    from qwen_tts.inference.length_penalty_processor import LengthPenaltyLogitsProcessor
    print("  ✅ Processor importable")
except ImportError as e:
    print(f"  ❌ Processor import failed: {e}")
    sys.exit(1)

# Check 3: Core model fix (THE CRITICAL CHECK!)
try:
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    import inspect
    gen_sig = inspect.signature(Qwen3TTSForConditionalGeneration.generate)
    gen_params = list(gen_sig.parameters.keys())
    
    if 'logits_processor' in gen_params:
        print("  ✅ Core model fix applied (logits_processor parameter present)")
    else:
        print("  ❌ Core model fix NOT applied - this is critical!")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ Core model check failed: {e}")
    sys.exit(1)

print("\n🎉 All checks passed!")
PYEOF

VERIFY_RESULT=$?
echo ""

if [ $VERIFY_RESULT -eq 0 ]; then
    echo "══════════════════════════════════════════════════════════════"
    echo -e "${GREEN}✅ INSTALLATION SUCCESSFUL!${NC}"
    echo "══════════════════════════════════════════════════════════════"
    echo ""
    echo "🎯 The bug is fixed! Length penalty will now work!"
    echo ""
    echo "Next steps:"
    echo "  1. Edit test_length_penalty.py to set your REF_AUDIO path"
    echo "  2. Add debug prints to see the penalty in action:"
    echo "     Edit length_penalty_processor.py, add in __call__:"
    echo "     if self.step_count % 10 == 0:"
    echo "         print(f'[PENALTY] Step {self.step_count}/{self.target_frames}')"
    echo ""
    echo "  3. Run: python3 test_length_penalty.py"
    echo "  4. You should see [PENALTY] output and different durations!"
    echo ""
    echo "🔄 To rollback:"
    echo "  cp $QWEN_PATH/inference/qwen3_tts_model.py.backup_$TIMESTAMP \\"
    echo "     $QWEN_PATH/inference/qwen3_tts_model.py"
    echo "  cp $QWEN_PATH/core/models/modeling_qwen3_tts.py.backup_$TIMESTAMP \\"
    echo "     $QWEN_PATH/core/models/modeling_qwen3_tts.py"
    echo ""
else
    echo "══════════════════════════════════════════════════════════════"
    echo -e "${RED}❌ INSTALLATION FAILED${NC}"
    echo "══════════════════════════════════════════════════════════════"
    echo ""
    echo "Check the error messages above."
    echo "Backups are available:"
    echo "  - qwen3_tts_model.py.backup_$TIMESTAMP"
    echo "  - modeling_qwen3_tts.py.backup_$TIMESTAMP"
    echo ""
    exit 1
fi
