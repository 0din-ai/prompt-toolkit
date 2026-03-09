---
sidebar_position: 3
---

# Embedding Providers API

API reference for embedding providers. See [Embedding Providers Concept](../concepts/embedding-providers) for usage guidance and comparison.

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## EmbeddingProvider

Protocol/trait/interface that all embedding providers must implement.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    fn name(&self) -> &str;
    fn model(&self) -> &str;
    fn dimensions(&self) -> usize;
    async fn generate_embedding(&self, text: &str) -> Result<EmbeddingResult>;
    async fn close(&self) -> Result<()>;
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'onnx')."""
    
    def model(self) -> str:
        """Model identifier."""
    
    def dimensions(self) -> int:
        """Embedding dimensionality."""
    
    async def generate_embedding(self, text: str) -> EmbeddingResult:
        """Generate embedding from text."""
    
    async def close(self) -> None:
        """Release resources (sessions, file handles, etc.)."""
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
interface EmbeddingProvider {
  name(): string;              // Provider name
  model(): string;             // Model identifier
  dimensions(): number;        // Embedding dimensionality
  generateEmbedding(text: string): Promise<EmbeddingResult>;
  close(): Promise<void>;      // Release resources
}
```

</TabItem>
</Tabs>

**Methods:**
- `name()`: Returns provider identifier (`"openai"`, `"onnx"`, etc.)
- `model()`: Returns model name (e.g., `"text-embedding-3-large"`)
- `dimensions()`: Returns embedding vector size (1536 for OpenAI, 384 for ONNX)
- `generate_embedding(text)`: Generates normalized embedding from text
- `close()`: Cleanup method (closes HTTP sessions, ONNX runtime, etc.)

---

## OpenAIProvider

Embedding provider using OpenAI's API.

### Constructor

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
impl OpenAIProvider {
    pub fn new(
        api_key: String,
        model: Option<String>,      // Default: "text-embedding-3-large"
        dimensions: Option<usize>,  // Default: 1536
        name: Option<String>,       // Default: "openai"
    ) -> Self
}
```

**Example:**
```rust
use signature_sdk::providers::OpenAIProvider;

let provider = OpenAIProvider::new(
    std::env::var("OPENAI_API_KEY")?,
    Some("text-embedding-3-large".to_string()),
    Some(1536),
    None,
);
```

</TabItem>
<TabItem value="python" label="Python">

```python
class OpenAIProvider:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        dimensions: int = 1536,
        name: str = "openai",
        base_url: Optional[str] = None,
    )
```

**Parameters:**
- `api_key`: OpenAI API key (get from https://platform.openai.com/api-keys)
- `model`: OpenAI model name (default: `"text-embedding-3-large"`)
- `dimensions`: Embedding dimensions (default: `1536`)
- `name`: Provider name (default: `"openai"`)
- `base_url`: Optional custom API base URL

**Example:**
```python
from signature_sdk.providers import OpenAIProvider
import os

provider = OpenAIProvider(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-large",
    dimensions=1536,
)
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
class OpenAIProvider implements EmbeddingProvider {
  constructor(config: {
    apiKey: string;
    model?: string;       // Default: "text-embedding-3-large"
    dimensions?: number;  // Default: 1536
    name?: string;        // Default: "openai"
    baseURL?: string;     // Optional custom API URL
  })
}
```

**Example:**
```typescript
import { OpenAIProvider } from '@0din/signature-sdk/providers';

const provider = new OpenAIProvider({
  apiKey: process.env.OPENAI_API_KEY!,
  model: 'text-embedding-3-large',
  dimensions: 1536,
});
```

</TabItem>
</Tabs>

### Configuration

**Environment Variables:**
- `OPENAI_API_KEY`: API key (can be passed via constructor instead)
- `OPENAI_BASE_URL`: Custom API base URL (optional)

**Cost:** ~$0.13 per 1M tokens (~$0.000013 per prompt)

**Latency:** ~100-200ms (network + API processing)

**Feature Requirement:**

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```toml
[dependencies]
signature-sdk = { version = "0.1", features = ["openai"] }
```

</TabItem>
<TabItem value="python" label="Python">

```bash
pip install 'signature-sdk[openai]'
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```bash
npm install @0din/signature-sdk openai
```

</TabItem>
</Tabs>

---

## OnnxProvider

Local ONNX-based embedding provider (no API key required).

### Factory Method

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
impl OnnxProvider {
    pub async fn new(
        cache: &ModelCache,
        model_name: Option<String>,  // Default: "intfloat/multilingual-e5-small"
        name: Option<String>,        // Default: "onnx"
    ) -> Result<Self>
}
```

**Example:**
```rust
use signature_sdk::providers::{ModelCache, OnnxProvider};

let cache = ModelCache::new()?;
let provider = OnnxProvider::new(&cache, None, None).await?;

// Provider auto-downloads model to cache directory
```

</TabItem>
<TabItem value="python" label="Python">

```python
class OnnxProvider:
    @classmethod
    async def new(
        cls,
        cache: ModelCache,
        model_name: str = "intfloat/multilingual-e5-small",
        name: str = "onnx",
    ) -> "OnnxProvider"
```

**Example:**
```python
from signature_sdk.providers import ModelCache, OnnxProvider

cache = ModelCache()
provider = await OnnxProvider.new(cache)

# Provider auto-downloads model to cache directory
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
class OnnxProvider implements EmbeddingProvider {
  static async create(
    cache: ModelCache,
    modelName?: string,  // Default: "intfloat/multilingual-e5-small"
    name?: string        // Default: "onnx"
  ): Promise<OnnxProvider>
}
```

**Example:**
```typescript
import { ModelCache, OnnxProvider } from '@0din/signature-sdk/providers';

const cache = new ModelCache();
const provider = await OnnxProvider.create(cache);

// Provider auto-downloads model to cache directory
```

</TabItem>
</Tabs>

### Configuration

**Model:** intfloat/multilingual-e5-small (custom fine-tuned variant for prompt similarity)

**Dimensions:** 384

**Cost:** Free (local CPU inference)

**Latency:** ~50-100ms on M1 Mac (CPU), ~10-20ms on GPU

**Model Download:** First run auto-downloads ~150MB model to cache directory

**Feature Requirement:**

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```toml
[dependencies]
signature-sdk = { version = "0.1", features = ["onnx"] }
```

</TabItem>
<TabItem value="python" label="Python">

```bash
pip install 'signature-sdk[onnx]'
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```bash
npm install @0din/signature-sdk onnxruntime-node
```

</TabItem>
</Tabs>

---

## ModelCache

Model cache manager for ONNX provider.

### Constructor

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
impl ModelCache {
    pub fn new() -> Result<Self>
    pub fn with_dir(cache_dir: PathBuf) -> Result<Self>
    pub fn cache_dir(&self) -> &Path
}
```

**Example:**
```rust
use signature_sdk::providers::ModelCache;

// Use default cache directory
let cache = ModelCache::new()?;

// Or specify custom directory
let cache = ModelCache::with_dir("/path/to/cache".into())?;
```

</TabItem>
<TabItem value="python" label="Python">

```python
class ModelCache:
    def __init__(self, cache_dir: Optional[Path] = None)
    
    @property
    def cache_dir(self) -> Path
```

**Example:**
```python
from signature_sdk.providers import ModelCache

# Use default cache directory
cache = ModelCache()

# Or specify custom directory
cache = ModelCache(cache_dir=Path("/path/to/cache"))
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
class ModelCache {
  constructor(cacheDir?: string)
  
  get cacheDir(): string
}
```

**Example:**
```typescript
import { ModelCache } from '@0din/signature-sdk/providers';

// Use default cache directory
const cache = new ModelCache();

// Or specify custom directory
const cache = new ModelCache('/path/to/cache');
```

</TabItem>
</Tabs>

### Cache Directory

**Default Location:**
- Linux/macOS: `~/.cache/signature-sdk/models/`
- Windows: `%LOCALAPPDATA%\signature-sdk\models\`

**Override via Environment Variable:**
```bash
export SIGNATURE_SDK_MODEL_CACHE=/path/to/cache
```

**Directory Structure:**
```
~/.cache/signature-sdk/models/
├── v1/
│   ├── config.json          # Model metadata
│   ├── model.onnx           # ONNX model (~150MB)
│   ├── tokenizer.json       # Tokenizer config
│   └── special_tokens_map.json
└── .locks/                  # Download lock files
```

**Storage Requirements:** ~150MB per model version

**Thread Safety:** ModelCache handles concurrent access via file locks

---

## Custom Providers

You can implement custom providers for any embedding source:

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
use async_trait::async_trait;
use signature_sdk::{EmbeddingProvider, EmbeddingResult, SigError};

pub struct CustomProvider {
    // Your provider fields
}

#[async_trait]
impl EmbeddingProvider for CustomProvider {
    fn name(&self) -> &str {
        "my-custom-provider"
    }
    
    fn model(&self) -> &str {
        "my-model-v1"
    }
    
    fn dimensions(&self) -> usize {
        384  // Your embedding size
    }
    
    async fn generate_embedding(&self, text: &str) -> Result<EmbeddingResult, SigError> {
        // 1. Generate raw embedding
        let embedding = your_embedding_function(text)?;
        
        // 2. Normalize
        let normalized = normalize_vector(&embedding);
        
        // 3. Compute SHA256
        let sha256 = compute_embedding_sha256(&normalized);
        
        Ok(EmbeddingResult {
            embedding,
            normalized_embedding: normalized,
            normalized_embedding_sha256: sha256,
            model: self.model().to_string(),
            dimensions: self.dimensions(),
            token_count: None,
            timing_ms: None,
        })
    }
    
    async fn close(&self) -> Result<(), SigError> {
        // Cleanup logic
        Ok(())
    }
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
from signature_sdk import EmbeddingProvider, EmbeddingResult
from signature_sdk import normalize_vector, compute_embedding_sha256

class CustomProvider:
    def name(self) -> str:
        return "my-custom-provider"
    
    def model(self) -> str:
        return "my-model-v1"
    
    def dimensions(self) -> int:
        return 384  # Your embedding size
    
    async def generate_embedding(self, text: str) -> EmbeddingResult:
        # 1. Generate raw embedding
        embedding = your_embedding_function(text)
        
        # 2. Normalize
        normalized = normalize_vector(embedding)
        
        # 3. Compute SHA256
        sha256 = compute_embedding_sha256(normalized)
        
        return EmbeddingResult(
            embedding=embedding,
            normalized_embedding=normalized,
            normalized_embedding_sha256=sha256,
            model=self.model(),
            dimensions=self.dimensions(),
        )
    
    async def close(self) -> None:
        # Cleanup logic
        pass
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
import { 
  EmbeddingProvider, 
  EmbeddingResult,
  normalizeVector,
  computeEmbeddingSha256
} from '@0din/signature-sdk';

class CustomProvider implements EmbeddingProvider {
  name(): string {
    return 'my-custom-provider';
  }
  
  model(): string {
    return 'my-model-v1';
  }
  
  dimensions(): number {
    return 384;  // Your embedding size
  }
  
  async generateEmbedding(text: string): Promise<EmbeddingResult> {
    // 1. Generate raw embedding
    const embedding = yourEmbeddingFunction(text);
    
    // 2. Normalize
    const normalized = normalizeVector(embedding);
    
    // 3. Compute SHA256
    const sha256 = computeEmbeddingSha256(normalized);
    
    return {
      embedding,
      normalizedEmbedding: normalized,
      normalizedEmbeddingSha256: sha256,
      model: this.model(),
      dimensions: this.dimensions(),
    };
  }
  
  async close(): Promise<void> {
    // Cleanup logic
  }
}
```

</TabItem>
</Tabs>

**Requirements for Custom Providers:**
1. Return normalized embeddings (L2 norm = 1)
2. Compute SHA256 hash in canonical JSON format
3. Implement `close()` for resource cleanup
4. Handle errors gracefully (network, model loading, etc.)

---

## See Also

- [Embedding Providers Concept](../concepts/embedding-providers) - Usage guide and comparison
- [Signature Versions](../concepts/signature-versions) - V0 vs V1 compatibility
- [Types](./types) - EmbeddingResult structure
- [Errors](./errors) - Provider error handling
