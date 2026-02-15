#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic Patch Script for Qwen3-TTS Length Penalty
Applies all modifications to qwen3_tts_model.py without manual editing.
"""

import os
import sys
import shutil
from pathlib import Path
import re
from datetime import datetime


def find_qwen_tts_path():
    """Find the qwen_tts installation directory."""
    try:
        import qwen_tts
        qwen_path = Path(qwen_tts.__file__).parent
        return qwen_path
    except ImportError:
        print("❌ ERROR: qwen_tts not installed!")
        print("Install it with: pip install qwen-tts")
        sys.exit(1)


def backup_file(filepath):
    """Create a timestamped backup of the file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath.parent / f"{filepath.name}.backup_{timestamp}"
    shutil.copy2(filepath, backup_path)
    return backup_path


def apply_import_modification(content):
    """Add the length_penalty_processor import."""
    # Find the line with "from ..core.models import"
    pattern = r'(from \.\.core\.models import.*?\n)'
    match = re.search(pattern, content)
    
    if not match:
        print("⚠️  Warning: Could not find import location")
        return content, False
    
    # Check if already imported
    if 'from .length_penalty_processor import' in content:
        print("  ℹ️  Import already present, skipping")
        return content, True
    
    # Insert after the core.models import
    insert_pos = match.end()
    new_import = "from .length_penalty_processor import LengthPenaltyLogitsProcessor\n"
    
    modified_content = content[:insert_pos] + new_import + content[insert_pos:]
    print("  ✅ Added import statement")
    return modified_content, True


def apply_signature_modification(content):
    """Modify generate_voice_clone signature to add new parameters."""
    # Find the function signature
    pattern = r'(@torch\.inference_mode\(\)\s+def generate_voice_clone\([^)]*?non_streaming_mode: bool = True,)\s*(\*\*kwargs,)'
    
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print("⚠️  Warning: Could not find generate_voice_clone signature")
        return content, False
    
    # Check if already modified
    if 'length_penalty_alpha' in content[match.start():match.start()+2000]:
        print("  ℹ️  Signature already modified, skipping")
        return content, True
    
    # Insert new parameters
    new_params = (
        match.group(1) + "\n" +
        "        length_penalty_alpha: float = 0.0,\n" +
        "        frames_per_text_token: float = 8.0,\n" +
        "        " + match.group(2)
    )
    
    modified_content = content[:match.start()] + new_params + content[match.end():]
    print("  ✅ Modified function signature")
    return modified_content, True


def apply_docstring_modification(content):
    """Add documentation for new parameters in docstring."""
    # Find the max_new_tokens documentation
    pattern = r'(max_new_tokens:\s+Maximum number of new codec tokens to generate\.\s+)(\*\*kwargs:)'
    
    match = re.search(pattern, content)
    if not match:
        print("  ℹ️  Docstring location not found, skipping (optional)")
        return content, True
    
    # Check if already documented
    if 'length_penalty_alpha:' in content[match.start():match.start()+1000]:
        print("  ℹ️  Docstring already updated, skipping")
        return content, True
    
    # Insert documentation
    new_docs = (
        match.group(1) +
        "length_penalty_alpha:\n" +
        "                Strength of length penalty (0.0 = disabled, 0.15 = medium, 0.20 = strong).\n" +
        "                Prevents generation from being excessively long by boosting EOS probability.\n" +
        "            frames_per_text_token:\n" +
        "                Expected codec frames per text token (default: 8.0).\n" +
        "                Lower values = expects faster speech. Adjust based on speaking rate.\n" +
        "            " + match.group(2)
    )
    
    modified_content = content[:match.start()] + new_docs + content[match.end():]
    print("  ✅ Added parameter documentation")
    return modified_content, True


def apply_logic_modification(content):
    """Add length penalty logic before model.generate() call."""
    # Find the gen_kwargs line and model.generate call in generate_voice_clone
    # We need to be careful to find the right occurrence (in generate_voice_clone, not other methods)
    
    # Strategy: Find generate_voice_clone method, then find the gen_kwargs and model.generate within it
    voice_clone_pattern = r'def generate_voice_clone\([^)]*?\).*?(?=\n    def |\nclass |\Z)'
    voice_clone_match = re.search(voice_clone_pattern, content, re.DOTALL)
    
    if not voice_clone_match:
        print("⚠️  Warning: Could not find generate_voice_clone method")
        return content, False
    
    method_content = voice_clone_match.group(0)
    method_start = voice_clone_match.start()
    
    # Within this method, find: gen_kwargs = ... followed by talker_codes_list
    pattern = r'(gen_kwargs = self\._merge_generate_kwargs\(\*\*kwargs\)\s*\n)\s*(talker_codes_list, _ = self\.model\.generate\()'
    
    match = re.search(pattern, method_content)
    if not match:
        print("⚠️  Warning: Could not find gen_kwargs/model.generate location")
        return content, False
    
    # Check if already modified
    if 'Apply length penalty' in method_content[max(0, match.start()-500):match.end()]:
        print("  ℹ️  Length penalty logic already present, skipping")
        return content, True
    
    # Insert the length penalty logic
    new_logic = match.group(1) + """
        # Apply length penalty if enabled
        if length_penalty_alpha > 0:
            try:
                # Get EOS token ID from model config
                eos_id = self.model.config.talker_config.codec_eos_token_id
                
                # Create processor for each input sequence
                # NOTE: Currently HuggingFace LogitsProcessorList applies same processor
                # to all batch items. For different text lengths in batch, this is
                # approximate. For best results, use batch_size=1 in inference.
                processors = []
                for ids in input_ids:
                    processor = LengthPenaltyLogitsProcessor(
                        text_length=ids.shape[1],
                        frames_per_text_token=frames_per_text_token,
                        penalty_alpha=length_penalty_alpha,
                        eos_token_id=eos_id,
                    )
                    processors.append(processor)
                
                # HuggingFace limitation: logits_processor applies to entire batch
                # We use the first processor as a reasonable approximation
                # For variable-length batches, generate one at a time
                if 'logits_processor' not in gen_kwargs:
                    gen_kwargs['logits_processor'] = []
                gen_kwargs['logits_processor'].extend(processors[:1])
                
            except Exception as e:
                # If anything fails, continue without length penalty
                print(f"Warning: Could not apply length penalty: {e}")

        """ + match.group(2)
    
    # Replace in the method content
    new_method_content = method_content[:match.start()] + new_logic + method_content[match.end():]
    
    # Replace in full content
    modified_content = content[:method_start] + new_method_content + content[method_start + len(method_content):]
    
    print("  ✅ Added length penalty logic")
    return modified_content, True


def verify_modifications(content):
    """Verify all modifications were applied."""
    checks = [
        ("Import", "from .length_penalty_processor import LengthPenaltyLogitsProcessor"),
        ("Signature param 1", "length_penalty_alpha: float = 0.0"),
        ("Signature param 2", "frames_per_text_token: float = 8.0"),
        ("Logic", "Apply length penalty if enabled"),
    ]
    
    all_passed = True
    for name, pattern in checks:
        if pattern in content:
            print(f"  ✅ {name} verified")
        else:
            print(f"  ❌ {name} NOT FOUND")
            all_passed = False
    
    return all_passed


def main():
    print("=" * 70)
    print("🔧 Automatic Length Penalty Patcher")
    print("=" * 70)
    print()
    
    # Find qwen_tts
    print("Step 1: Locating qwen_tts installation...")
    qwen_path = find_qwen_tts_path()
    print(f"  ✅ Found at: {qwen_path}")
    print()
    
    # Check processor file
    print("Step 2: Checking length_penalty_processor.py...")
    processor_src = Path("length_penalty_processor.py")
    processor_dst = qwen_path / "inference" / "length_penalty_processor.py"
    
    if not processor_src.exists():
        print(f"  ❌ ERROR: {processor_src} not found in current directory!")
        print("  Please run this script from the directory containing the downloaded files.")
        sys.exit(1)
    
    # Copy processor
    processor_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(processor_src, processor_dst)
    print(f"  ✅ Copied to: {processor_dst}")
    print()
    
    # Target file
    print("Step 3: Preparing to patch qwen3_tts_model.py...")
    model_file = qwen_path / "inference" / "qwen3_tts_model.py"
    
    if not model_file.exists():
        print(f"  ❌ ERROR: {model_file} not found!")
        sys.exit(1)
    
    print(f"  Target: {model_file}")
    
    # Create backup
    print("  Creating backup...", end=" ")
    backup_path = backup_file(model_file)
    print(f"✅ {backup_path.name}")
    print()
    
    # Read content
    print("Step 4: Applying modifications...")
    with open(model_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    success = True
    
    # Apply modifications
    modifications = [
        ("Import", apply_import_modification),
        ("Signature", apply_signature_modification),
        ("Docstring", apply_docstring_modification),
        ("Logic", apply_logic_modification),
    ]
    
    for name, func in modifications:
        print(f"  Applying {name}...", end=" ")
        content, ok = func(content)
        if not ok:
            success = False
            print(f"❌ FAILED")
        print()
    
    if not success:
        print("⚠️  Some modifications failed. File not written.")
        print("You may need to apply the patch manually.")
        sys.exit(1)
    
    # Write modified content
    print("Step 5: Writing modified file...")
    with open(model_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print("  ✅ File written")
    print()
    
    # Verify
    print("Step 6: Verifying modifications...")
    if verify_modifications(content):
        print("  ✅ All checks passed!")
    else:
        print("  ⚠️  Some checks failed. Please review manually.")
        print(f"  Backup available at: {backup_path}")
    print()
    
    # Final test
    print("Step 7: Testing import...")
    try:
        # Reimport to test
        import importlib
        import qwen_tts
        importlib.reload(qwen_tts)
        
        from qwen_tts import Qwen3TTSModel
        import inspect
        
        sig = inspect.signature(Qwen3TTSModel.generate_voice_clone)
        params = list(sig.parameters.keys())
        
        if 'length_penalty_alpha' in params and 'frames_per_text_token' in params:
            print("  ✅ Parameters detected in function signature!")
        else:
            print("  ⚠️  Parameters not detected. May need to restart Python.")
    except Exception as e:
        print(f"  ⚠️  Could not test: {e}")
        print("  Try restarting Python and testing manually.")
    print()
    
    # Summary
    print("=" * 70)
    print("✅ PATCH COMPLETE!")
    print("=" * 70)
    print()
    print("📝 Summary:")
    print(f"  • Original file backed up to: {backup_path.name}")
    print(f"  • Modified file: {model_file}")
    print()
    print("🧪 Next steps:")
    print("  1. Edit test_length_penalty.py to set your REF_AUDIO path")
    print("  2. Run: python3 test_length_penalty.py")
    print("  3. Listen to generated clips and choose optimal alpha")
    print()
    print("🔄 To revert changes:")
    print(f"  cp {backup_path} {model_file}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
