# signature-sdk Design Partner Deliverable

This package contains everything you need to get started with the signature-sdk for efficient similarity search and duplicate detection.

## What's Included

```
signature-sdk-deliverable-v0.1.1/
├── install.sh              # Automated installation script
├── README.md               # This file
├── INSTALL.md              # Detailed installation guide
├── verify.py               # Post-install verification script
├── sdk/
│   ├── signature_sdk-0.1.1-py3-none-any.whl
│   ├── signature_sdk_native-0.1.1-*.whl    # Platform-specific (if included)
│   └── requirements.txt
├── model/
│   └── v1/
│       ├── onnx/
│       │   └── model_O4.onnx       # ~235MB (optimized ONNX model)
│       ├── tokenizer.json
│       └── config.json
├── signatures/
│   └── threat-feed-v1.json         # Example signature pack
└── examples/
    ├── basic_signature.py
    ├── similarity_comparison.py
    ├── duplicate_detection.py
    ├── sign_text.py
    └── cm_lsh_example.py
```

## Quick Start

### Prerequisites

- **Python 3.10 or newer** (check with `python3 --version`)
- **~500MB disk space** for model and SDK
- **Internet connection** (for online mode) OR bundled model files (for offline mode)

### Installation

**Option 1: Online Mode** (recommended if you have internet)
```bash
./install.sh --online
```

This will:
1. Create a Python virtual environment at `./venv/`
2. Install the signature-sdk package
3. Download the ONNX model from HuggingFace (~235MB)
4. Copy signature pack to `~/.signature-sdk/signatures/`
5. Run verification tests

**Option 2: Offline Mode** (for air-gapped environments)
```bash
./install.sh --offline
```

Requirements:
- Model files must be present in `model/v1/` directory
- No internet connection needed
- Same installation steps as online mode, but uses bundled model

### Verification

After installation completes, you should see:
```
✓ signature_sdk imported successfully (v0.1.1)
✓ Native Rust acceleration is ENABLED (653× faster)
✓ Signature generation works correctly
✓ Signature format validation passed
✓ Hamming distance calculation works correctly

All 4 checks passed!
Installation verified successfully.
```

If you see warnings about native acceleration, it means the pure Python implementation will be used (slower but fully functional).

## Platform Support

### Supported Platforms

Native Rust acceleration (653× faster) is available for:

| Platform | Architectures | Status |
|----------|---------------|--------|
| **Linux** | x86_64, aarch64 | ✅ Full support |
| **macOS** | x86_64 (Intel), arm64 (Apple Silicon) | ✅ Full support |
| **Windows** | x86_64 | ✅ Full support |

### Platform Detection

The installer automatically detects your platform and installs the appropriate native wheel if available. If no native wheel is found for your platform, the pure Python implementation is used (slower but identical results).

## Usage

### Activate the Virtual Environment

```bash
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### Run Examples

```bash
# Basic signature generation
python examples/basic_signature.py

# Compare multiple texts for similarity
python examples/similarity_comparison.py

# Batch duplicate detection
python examples/duplicate_detection.py

# High-level API with text input
python examples/sign_text.py
```

### Generate Your First Signature

```python
from signature_sdk import sign_text
from signature_sdk.providers import OnnxProvider

# Initialize embedding provider
provider = OnnxProvider.from_pretrained()

# Generate signature for a text
result = provider.sign_text(
    text="This is a test message",
    version="v1"
)

print(f"Signature: {result.signature_string}")
# Output: 0din-v1:abc123...
```

## Configuration

### Environment Variables

- `SIGNATURE_SDK_MODEL_CACHE`: Override model cache directory (default: `~/.cache/signature-sdk/`)
- `SIGNATURE_SDK_NO_NATIVE`: Set to `1` to force pure Python mode (disables native acceleration)
- `HF_TOKEN`: HuggingFace API token (required for private model downloads in online mode)

### Model Location

By default, models are cached at:
- Linux/macOS: `~/.cache/signature-sdk/models/`
- Windows: `%LOCALAPPDATA%\signature-sdk\models\`

You can override this with the `SIGNATURE_SDK_MODEL_CACHE` environment variable.

### Signature Packs

Signature packs are JSON files containing pre-computed signatures for known content (e.g., threat feeds, known duplicates). They are stored at:
- Linux/macOS: `~/.signature-sdk/signatures/`
- Windows: `%USERPROFILE%\.signature-sdk\signatures\`

## Troubleshooting

### Installation Issues

**Problem**: `install.sh` fails with "Python 3.10 or newer required"
- **Solution**: Install Python 3.10+ from [python.org](https://www.python.org/downloads/)

**Problem**: Native wheel not found for my platform
- **Solution**: The installer will automatically fall back to pure Python. You'll see a warning but installation will succeed.

**Problem**: Model download fails in online mode
- **Solution**: 
  1. Check internet connection
  2. Verify HuggingFace access (model repo: `0dinai/jailbreak-embeddings-small`)
  3. Set `HF_TOKEN` environment variable if repo is private
  4. Use offline mode with bundled model files

**Problem**: `verify.py` reports "signature generation failed"
- **Solution**:
  1. Check Python version (must be 3.10+)
  2. Try reinstalling: `./install.sh --online --force` (if available)
  3. Check disk space (~500MB required)
  4. Review error traceback for specific issue

### Runtime Issues

**Problem**: Import error `ModuleNotFoundError: No module named 'signature_sdk'`
- **Solution**: Activate the virtual environment first: `source venv/bin/activate`

**Problem**: Slow signature generation (~115ms per signature)
- **Solution**: Native acceleration is not installed. Reinstall with platform-specific wheel or accept slower performance (pure Python is still functional).

**Problem**: Model not found error
- **Solution**:
  1. Check model cache directory exists: `ls ~/.cache/signature-sdk/models/v1/`
  2. Re-run installer in online mode to download model
  3. Or manually copy model files to cache directory

## Performance

### With Native Acceleration (Rust)
- **Signature generation**: ~0.18ms per signature (~5,600 sigs/sec)
- **Recommended for**: Production deployments, real-time systems, batch processing

### Pure Python
- **Signature generation**: ~115ms per signature (~9 sigs/sec)
- **Recommended for**: Development, testing, platforms without native wheels

**Note**: Both implementations produce **bit-identical results**. The choice is purely about performance.

## What's Next?

### 1. Explore the Examples

The `examples/` directory contains working code demonstrating:
- Basic signature generation
- Similarity comparison between texts
- Batch duplicate detection (O(n) vs O(n²))
- High-level API usage
- CM-LSH (Confidence-Matrix LSH) for improved accuracy

### 2. Read the Documentation

Comprehensive documentation is included in `INSTALL.md` covering:
- Installation options and requirements
- Platform-specific instructions
- Air-gapped deployment
- Troubleshooting guide

### 3. Integrate Into Your Application

Key integration points:
- **Embedding provider**: Use `OnnxProvider` (included) or `OpenAIProvider` (requires API key)
- **Signature generation**: Call `sign_text()` or `simhash_lsh_multi()` depending on your needs
- **Similarity search**: Store signatures in database with band-based indexing
- **Duplicate detection**: Compare signatures using Hamming distance

See `examples/duplicate_detection.py` for a complete similarity search pipeline.

### 4. Review the Signature Pack

The included `signatures/threat-feed-v1.json` demonstrates the signature pack format:
```json
{
  "version": "v1",
  "created_at": "2024-03-09T00:00:00Z",
  "signatures": [
    {
      "id": "example-1",
      "signature": "0din-v1:abc123...",
      "metadata": {
        "category": "threat",
        "severity": "high"
      }
    }
  ]
}
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Application                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ sign_text()
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  signature-sdk (Python)                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  High-Level API: sign_text(), compare_signatures()    │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Embedding Providers: OnnxProvider, OpenAIProvider    │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  LSH Core: simhash_lsh_multi() [Rust or Python]       │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                       │
                       │ if native available
                       │
┌──────────────────────▼──────────────────────────────────────┐
│            signature-sdk-native (Rust/PyO3)                  │
│              ~653× faster signature generation               │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Signatures vs Embeddings

**Embeddings** (384-dim vectors):
- Capture semantic meaning
- High storage cost (~1.5KB per item)
- Slow similarity search at scale (must compare all items)

**Signatures** (256-bit hashes):
- Derived from embeddings via LSH (Locality-Sensitive Hashing)
- Low storage cost (~32 bytes per item)
- Fast similarity search (band-based indexing reduces candidates by ~44×)

### LSH Families

The SDK generates **3 independent hash families** by default for improved recall:
- Each family uses 256 bits split into 16 bands
- More families = better recall, but higher storage cost
- Similarity search checks candidates from ANY family match

### Versions

- **V0**: OpenAI embeddings (1536 dimensions) - legacy
- **V1**: ONNX embeddings (384 dimensions) - current, included in this package

**Important**: V0 and V1 signatures are NOT comparable (different embedding spaces).

## Support

For questions or issues with this deliverable package:

1. **Check troubleshooting section** above
2. **Review examples** in `examples/` directory  
3. **Read detailed installation guide** in `INSTALL.md`
4. **Contact your design partner representative**

## Version

**signature-sdk v0.1.1** (2024-03-09)

- Rust SDK: `signature-sdk` v0.1.1
- Python SDK: `signature-sdk` v0.1.1
- Python Native: `signature-sdk-native` v0.1.1
- Model: `0dinai/jailbreak-embeddings-small` v1 (ONNX optimized, ~235MB)

## License

Proprietary - Design Partner Distribution Only

This package is provided for evaluation purposes under a design partner agreement. Not for redistribution.

---

**Questions?** Contact your design partner representative or refer to the included documentation.
