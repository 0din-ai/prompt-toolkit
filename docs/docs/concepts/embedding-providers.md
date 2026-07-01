---
sidebar_position: 3
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Embedding Providers

Embedding providers generate vector embeddings from text. The odin-prompt-toolkit SDK uses an ONNX-based local inference provider by default, with support for custom providers.

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

## ONNX Provider (V1)

**Best for:** Local development, API-free deployment, cost-sensitive applications

- **Model**: 0dinai/0din-jailbreak-embeddings-small (custom fine-tuned variant)
- **Dimensions**: 1024
- **Cost**: Free (local inference)
- **Setup**: Model auto-downloads to `~/.cache/odin-prompt-toolkit/models/v1/`
- **Latency**: ~50-100ms (CPU inference, Apple M4 Pro)
- **Quality**: Good multilingual performance, optimized for prompt similarity

### Usage

<Tabs groupId="language">
  <TabItem value="rust" label="Rust">

```rust
use odin_prompt_toolkit::{sign_text, SignatureVersion};
use odin_prompt_toolkit::providers::{ModelCache, OnnxProvider};

let cache = ModelCache::new()?;
let provider = OnnxProvider::new(&cache, None, None, 0, 0).await?;

// Uses latest model (V1) by default
let result = sign_text("Your text here", &provider, SignatureVersion::Latest, None).await?;
```

  </TabItem>
  <TabItem value="python" label="Python">

```python
from odin_prompt_toolkit import sign_text, SignatureVersion
from odin_prompt_toolkit.providers import ModelCache, OnnxProvider

cache = ModelCache()
provider = await OnnxProvider.new(cache)

# Uses latest model (V1) by default
result = await sign_text("Your text here", provider)
```

  </TabItem>
  <TabItem value="typescript" label="TypeScript">

```typescript
import { signText, SignatureVersion } from '@0din/odin-prompt-toolkit';
import { ModelCache, OnnxProvider } from '@0din/odin-prompt-toolkit/providers';

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
odin-prompt-toolkit = { version = "0.1", features = ["onnx"] }
```

  </TabItem>
  <TabItem value="python" label="Python">

```bash
pip install 'odin-prompt-toolkit[onnx]'
```

  </TabItem>
  <TabItem value="typescript" label="TypeScript">

```bash
npm install @0din/odin-prompt-toolkit onnxruntime-node
```

  </TabItem>
</Tabs>

## Custom Providers

You can implement your own embedding provider by implementing the `EmbeddingProvider` trait/interface:

<Tabs groupId="language">
  <TabItem value="rust" label="Rust">

```rust
use async_trait::async_trait;
use odin_prompt_toolkit::{EmbeddingProvider, EmbeddingResult};

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
from odin_prompt_toolkit import EmbeddingProvider, EmbeddingResult

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
import { EmbeddingProvider, EmbeddingResult } from '@0din/odin-prompt-toolkit';

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

See [Signature Versions](./signature-versions) for details on versioning and backward compatibility.

## See Also

- [SusFactor Backend Selection](./susfactor#backend-selection-v080) — the `SusFactorProvider` trait follows the same abstraction pattern as `EmbeddingProvider`, applied to classifier backends.
