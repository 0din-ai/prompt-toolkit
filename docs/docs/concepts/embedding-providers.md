---
sidebar_position: 3
---

# Embedding Providers

Embedding providers generate vector embeddings from text. The signature-sdk SDK supports two built-in providers: OpenAI (API-based) and ONNX (local inference).

## Provider Interface

All providers implement the same interface:

```rust
trait EmbeddingProvider {
    fn name(&self) -> &str;
    fn model(&self) -> &str;
    fn dimensions(&self) -> usize;
    async fn generate_embedding(&self, text: &str) -> Result<EmbeddingResult>;
    async fn close(&self) -> Result<()>;
}
```

This allows you to switch between providers without changing your signature generation code.

## OpenAI Provider (V0)

**Best for:** Production applications requiring state-of-the-art embeddings

- **Model**: text-embedding-3-large
- **Dimensions**: 1536
- **Cost**: $0.13 per 1M tokens (~$0.000013 per prompt)
- **Setup**: Requires OpenAI API key
- **Latency**: ~100-200ms (network + API)
- **Quality**: Industry-leading semantic understanding

### Usage

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs groupId="language">
  <TabItem value="rust" label="Rust">

```rust
use signature_sdk::{sign_text, SignatureVersion};
use signature_sdk::providers::OpenAIProvider;

let provider = OpenAIProvider::new(
    std::env::var("OPENAI_API_KEY")?,
    None, // model (defaults to text-embedding-3-large)
    None, // dimensions (defaults to 1536)
    None, // name
);

let result = sign_text("Your text here", &provider, SignatureVersion::V0, None).await?;
```

  </TabItem>
  <TabItem value="python" label="Python">

```python
from signature_sdk import sign_text, SignatureVersion
from signature_sdk.providers import OpenAIProvider
import os

provider = OpenAIProvider(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-large",  # optional
    dimensions=1536,  # optional
)

result = await sign_text("Your text here", provider, SignatureVersion.V0)
```

  </TabItem>
  <TabItem value="typescript" label="TypeScript">

```typescript
import { signText, SignatureVersion } from '@0din/signature-sdk';
import { OpenAIProvider } from '@0din/signature-sdk/providers';

const provider = new OpenAIProvider({
  apiKey: process.env.OPENAI_API_KEY!,
  model: 'text-embedding-3-large',  // optional
  dimensions: 1536,  // optional
});

const result = await signText("Your text here", provider, SignatureVersion.V0);
```

  </TabItem>
</Tabs>

**Installation:**

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

## ONNX Provider (V1)

**Best for:** Local development, API-free deployment, cost-sensitive applications

- **Model**: intfloat/multilingual-e5-large (custom fine-tuned variant)
- **Dimensions**: 384
- **Cost**: Free (local inference)
- **Setup**: Model auto-downloads to `~/.cache/signature-sdk/models/v1/`
- **Latency**: ~50-100ms (CPU inference on M1 Mac)
- **Quality**: Good multilingual performance, optimized for prompt similarity

### Usage

<Tabs groupId="language">
  <TabItem value="rust" label="Rust">

```rust
use signature_sdk::{sign_text, SignatureVersion};
use signature_sdk::providers::{ModelCache, OnnxProvider};

let cache = ModelCache::new()?;
let provider = OnnxProvider::new(&cache, None, None).await?;

// Uses latest model (V1) by default
let result = sign_text("Your text here", &provider, SignatureVersion::Latest, None).await?;
```

  </TabItem>
  <TabItem value="python" label="Python">

```python
from signature_sdk import sign_text, SignatureVersion
from signature_sdk.providers import ModelCache, OnnxProvider

cache = ModelCache()
provider = await OnnxProvider.new(cache)

# Uses latest model (V1) by default
result = await sign_text("Your text here", provider)
```

  </TabItem>
  <TabItem value="typescript" label="TypeScript">

```typescript
import { signText, SignatureVersion } from '@0din/signature-sdk';
import { ModelCache, OnnxProvider } from '@0din/signature-sdk/providers';

const cache = new ModelCache();
const provider = await OnnxProvider.create(cache);

// Uses latest model (V1) by default
const result = await signText("Your text here", provider);
```

  </TabItem>
</Tabs>

**Installation:**

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

## Comparison

| Feature | OpenAI (V0) | ONNX (V1) |
|---------|-------------|-----------|
| **Dimensions** | 1536 | 1024 |
| **Cost** | $0.13 per 1M tokens | Free |
| **Latency** | ~100-200ms | ~50-100ms (CPU) |
| **API Key** | Required | Not required |
| **Network** | Required | Not required |
| **Quality** | Highest | Good |
| **Storage** | ~1.5KB per signature | ~384 bytes per signature |
| **Use Case** | Production apps | Local dev, cost-sensitive |

## Custom Providers

You can implement your own embedding provider by implementing the `EmbeddingProvider` trait/interface:

<Tabs groupId="language">
  <TabItem value="rust" label="Rust">

```rust
use async_trait::async_trait;
use signature_sdk::{EmbeddingProvider, EmbeddingResult};

struct MyProvider;

#[async_trait]
impl EmbeddingProvider for MyProvider {
    fn name(&self) -> &str { "my-provider" }
    fn model(&self) -> &str { "my-model" }
    fn dimensions(&self) -> usize { 384 }
    
    async fn generate_embedding(&self, text: &str) -> Result<EmbeddingResult> {
        // Your embedding logic here
    }
    
    async fn close(&self) -> Result<()> { Ok(()) }
}
```

  </TabItem>
  <TabItem value="python" label="Python">

```python
from signature_sdk import EmbeddingProvider, EmbeddingResult

class MyProvider:
    def name(self) -> str:
        return "my-provider"
    
    def model(self) -> str:
        return "my-model"
    
    def dimensions(self) -> int:
        return 384
    
    async def generate_embedding(self, text: str) -> EmbeddingResult:
        # Your embedding logic here
        pass
    
    async def close(self) -> None:
        pass
```

  </TabItem>
  <TabItem value="typescript" label="TypeScript">

```typescript
import { EmbeddingProvider, EmbeddingResult } from '@0din/signature-sdk';

class MyProvider implements EmbeddingProvider {
  name(): string { return 'my-provider'; }
  model(): string { return 'my-model'; }
  dimensions(): number { return 384; }
  
  async generateEmbedding(text: string): Promise<EmbeddingResult> {
    // Your embedding logic here
  }
  
  async close(): Promise<void> {}
}
```

  </TabItem>
</Tabs>

See [Signature Versions](./signature-versions) for more details on V0 vs V1 compatibility.
