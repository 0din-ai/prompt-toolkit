---
sidebar_position: 6
---

# Error Handling

Error types and handling patterns across all three language implementations.

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## Error Hierarchy

All library operations use a unified error system with specific error types for different failure modes.

### SigError

Base error type for all library operations.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub enum SigError {
    Config(String),
    Provider(String),
    Model(String),
    Io(#[from] std::io::Error),
    Serialization(#[from] serde_json::Error),
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, SigError>;
```

**Usage:**
```rust
use odin_prompt_toolkit::{sign_text, SigError};

match sign_text(text, &provider, version, None).await {
    Ok(result) => println!("Success: {}", result.signature_string),
    Err(SigError::Provider(msg)) => eprintln!("Provider error: {}", msg),
    Err(SigError::InvalidInput(msg)) => eprintln!("Invalid input: {}", msg),
    Err(e) => eprintln!("Other error: {}", e),
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
class SigError(Exception):
    """Base exception for all odin-prompt-toolkit operations."""

class ConfigError(SigError):
    """Invalid LSH configuration."""

class ProviderError(SigError):
    """Embedding provider failure."""

class ModelError(SigError):
    """Model loading or inference error."""

class InvalidInputError(SigError):
    """Invalid input data."""
```

**Usage:**
```python
from odin_prompt_toolkit import sign_text, SigError, ProviderError, InvalidInputError

try:
    result = await sign_text(text, provider=provider)
    print(f"Success: {result.signature_string}")
except ProviderError as e:
    print(f"Provider error: {e}")
except InvalidInputError as e:
    print(f"Invalid input: {e}")
except SigError as e:
    print(f"Other error: {e}")
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
class SigError extends Error {
  name = 'SigError';
}

class ConfigError extends SigError {
  name = 'ConfigError';
}

class ProviderError extends SigError {
  name = 'ProviderError';
}

class ModelError extends SigError {
  name = 'ModelError';
}

class InvalidInputError extends SigError {
  name = 'InvalidInputError';
}
```

**Usage:**
```typescript
import { signText, SigError, ProviderError, InvalidInputError } from '@0din/odin-prompt-toolkit';

try {
  const result = await signText(text, provider);
  console.log(`Success: ${result.signatureString}`);
} catch (error) {
  if (error instanceof ProviderError) {
    console.error(`Provider error: ${error.message}`);
  } else if (error instanceof InvalidInputError) {
    console.error(`Invalid input: ${error.message}`);
  } else if (error instanceof SigError) {
    console.error(`Other error: ${error.message}`);
  }
}
```

</TabItem>
</Tabs>

---

## Error Types

### ConfigError

Raised when LSH configuration parameters are invalid or incompatible.

**Common Causes:**
- Invalid `families`, `bits`, or `bands` values
- Configuration mismatch (e.g., bands don't divide bits evenly)
- Negative or zero values

**Examples:**

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
// Invalid configuration
let config = LshConfig {
    families: 0,  // Must be > 0
    bits: 256,
    bands: 16,
};
// Returns: Err(SigError::Config("families must be > 0"))
```

</TabItem>
<TabItem value="python" label="Python">

```python
from odin_prompt_toolkit import LshConfig, ConfigError

try:
    config = LshConfig(families=0, bits=256, bands=16)  # Invalid
    # Would raise ConfigError if validated
except ConfigError as e:
    print(f"Config error: {e}")
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
import { ConfigError } from '@0din/odin-prompt-toolkit';

// Invalid configuration would throw ConfigError
// if validated in constructor
const config = { families: 0, bits: 256, bands: 16 };
```

</TabItem>
</Tabs>

---

### ProviderError

Raised when embedding provider fails to generate embeddings.

**Common Causes:**
- OpenAI API errors (authentication, rate limits, network)
- ONNX runtime errors
- Model file not found
- Network timeouts

**Examples:**

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
// Missing API key
let provider = OpenAIProvider::new("".to_string(), None, None, None);
// API call fails: Err(SigError::Provider("Authentication failed"))

// Network error
// Returns: Err(SigError::Provider("Network timeout after 30s"))
```

</TabItem>
<TabItem value="python" label="Python">

```python
from odin_prompt_toolkit.providers import OpenAIProvider
from odin_prompt_toolkit import ProviderError

try:
    provider = OpenAIProvider(api_key="invalid_key")
    result = await provider.generate_embedding("test")
except ProviderError as e:
    print(f"Provider failed: {e}")
    # e.g., "OpenAI API error: Invalid API key"
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
import { OpenAIProvider } from '@0din/odin-prompt-toolkit/providers';
import { ProviderError } from '@0din/odin-prompt-toolkit';

try {
  const provider = new OpenAIProvider({ apiKey: 'invalid' });
  await provider.generateEmbedding('test');
} catch (error) {
  if (error instanceof ProviderError) {
    console.error(`Provider failed: ${error.message}`);
    // e.g., "OpenAI API error: Incorrect API key provided"
  }
}
```

</TabItem>
</Tabs>

---

### ModelError

Raised when ONNX model loading or inference fails.

**Common Causes:**
- Model file corrupted or missing
- ONNX runtime initialization failure
- Incompatible model format
- Out of memory during inference

**Examples:**

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
// Model file not found
let cache = ModelCache::new()?;
let provider = OnnxProvider::new(&cache, Some("nonexistent/model".to_string()), None, 0, 0).await;
// Returns: Err(SigError::Model("Model file not found: nonexistent/model"))
```

</TabItem>
<TabItem value="python" label="Python">

```python
from odin_prompt_toolkit.providers import ModelCache, OnnxProvider
from odin_prompt_toolkit import ModelError

try:
    cache = ModelCache()
    provider = await OnnxProvider.new(cache, model_name="invalid/model")
except ModelError as e:
    print(f"Model error: {e}")
    # e.g., "Failed to load ONNX model: File not found"
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
import { ModelCache, OnnxProvider } from '@0din/odin-prompt-toolkit/providers';
import { ModelError } from '@0din/odin-prompt-toolkit';

try {
  const cache = new ModelCache();
  const provider = await OnnxProvider.create(cache, 'invalid/model');
} catch (error) {
  if (error instanceof ModelError) {
    console.error(`Model error: ${error.message}`);
  }
}
```

</TabItem>
</Tabs>

---

### InvalidInputError

Raised when input data doesn't meet requirements.

**Common Causes:**
- Empty text input
- Malformed signature string
- Invalid hex characters in signature
- Wrong embedding dimensions
- Zero-length or NaN vectors

**Examples:**

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
use odin_prompt_toolkit::{parse_signature_string, SigError};

// Invalid signature format
match parse_signature_string("invalid") {
    Err(SigError::InvalidInput(msg)) => {
        println!("Invalid: {}", msg);
        // "Invalid signature prefix: invalid"
    },
    _ => {}
}

// Non-hex characters
match parse_signature_string("0din-v1:xyz123") {
    Err(SigError::InvalidInput(msg)) => {
        println!("Invalid: {}", msg);
        // "Invalid hex signature: xyz123"
    },
    _ => {}
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
from odin_prompt_toolkit import parse_signature_string, InvalidInputError

try:
    # Invalid signature format
    parse_signature_string("invalid")
except InvalidInputError as e:
    print(f"Invalid: {e}")
    # "Invalid signature prefix: invalid"

try:
    # Non-hex characters
    parse_signature_string("0din-v1:xyz123")
except InvalidInputError as e:
    print(f"Invalid: {e}")
    # "Invalid hex signature: xyz123"
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
import { parseSignatureString, InvalidInputError } from '@0din/odin-prompt-toolkit';

try {
  // Invalid signature format
  parseSignatureString('invalid');
} catch (error) {
  if (error instanceof InvalidInputError) {
    console.error(`Invalid: ${error.message}`);
    // "Invalid signature prefix: invalid"
  }
}

try {
  // Non-hex characters
  parseSignatureString('0din-v1:xyz123');
} catch (error) {
  if (error instanceof InvalidInputError) {
    console.error(`Invalid: ${error.message}`);
    // "Invalid hex signature: xyz123"
  }
}
```

</TabItem>
</Tabs>

---

## Error Handling Patterns

### Retry with Exponential Backoff

For transient provider errors (rate limits, network issues):

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
use tokio::time::{sleep, Duration};

async fn sign_with_retry(
    text: &str,
    provider: &dyn EmbeddingProvider,
    max_retries: u32,
) -> Result<SignatureResult, SigError> {
    let mut delay_ms = 1000;
    
    for attempt in 0..max_retries {
        match sign_text(text, provider, SignatureVersion::Latest, None).await {
            Ok(result) => return Ok(result),
            Err(SigError::Provider(msg)) if attempt < max_retries - 1 => {
                eprintln!("Retry {}/{}: {}", attempt + 1, max_retries, msg);
                sleep(Duration::from_millis(delay_ms)).await;
                delay_ms *= 2;  // Exponential backoff
            },
            Err(e) => return Err(e),
        }
    }
    
    Err(SigError::Provider("Max retries exceeded".to_string()))
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
import asyncio
from odin_prompt_toolkit import sign_text, SignatureVersion, ProviderError

async def sign_with_retry(
    text: str,
    provider,
    max_retries: int = 3,
):
    delay_ms = 1000
    
    for attempt in range(max_retries):
        try:
            return await sign_text(text, provider=provider)
        except ProviderError as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1}/{max_retries}: {e}")
                await asyncio.sleep(delay_ms / 1000)
                delay_ms *= 2  # Exponential backoff
            else:
                raise
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
import { signText, SignatureVersion, ProviderError } from '@0din/odin-prompt-toolkit';

async function signWithRetry(
  text: string,
  provider: EmbeddingProvider,
  maxRetries: number = 3
) {
  let delayMs = 1000;
  
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await signText(text, provider);
    } catch (error) {
      if (error instanceof ProviderError && attempt < maxRetries - 1) {
        console.error(`Retry ${attempt + 1}/${maxRetries}: ${error.message}`);
        await new Promise(resolve => setTimeout(resolve, delayMs));
        delayMs *= 2;  // Exponential backoff
      } else {
        throw error;
      }
    }
  }
}
```

</TabItem>
</Tabs>

---

### Graceful Degradation

Fall back to alternative behavior on errors:

<Tabs groupId="language">
<TabItem value="python" label="Python">

```python
from odin_prompt_toolkit import sign_text, ProviderError, ModelError
from odin_prompt_toolkit.providers import OpenAIProvider, OnnxProvider, ModelCache

async def sign_with_fallback(text: str):
    """Try OpenAI, fall back to ONNX on error."""
    try:
        # Try OpenAI first (higher quality)
        provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
        return await sign_text(text, provider=provider, version=SignatureVersion.V0)
    except (ProviderError, ModelError) as e:
        print(f"OpenAI failed ({e}), falling back to ONNX...")
        
        # Fall back to ONNX (local, no API key needed)
        cache = ModelCache()
        provider = await OnnxProvider.new(cache)
        return await sign_text(text, provider=provider, version=SignatureVersion.V1)
```

</TabItem>
</Tabs>

---

### Input Validation

Validate inputs before expensive operations:

<Tabs groupId="language">
<TabItem value="typescript" label="TypeScript">

```typescript
import { parseSignatureString, InvalidInputError } from '@0din/odin-prompt-toolkit';

function validateAndCompare(sig1: string, sig2: string): number | null {
  try {
    const parsed1 = parseSignatureString(sig1);
    const parsed2 = parseSignatureString(sig2);
    
    // Check version compatibility
    if (parsed1.version !== parsed2.version) {
      console.error('Cannot compare signatures from different versions');
      return null;
    }
    
    // Proceed with comparison
    return hammingDistanceHex(parsed1.signature, parsed2.signature);
    
  } catch (error) {
    if (error instanceof InvalidInputError) {
      console.error(`Validation failed: ${error.message}`);
      return null;
    }
    throw error;
  }
}
```

</TabItem>
</Tabs>

---

## Error Context

### Adding Context

Wrap errors with additional context:

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
use anyhow::{Context, Result};

async fn process_batch(texts: Vec<String>) -> Result<Vec<SignatureResult>> {
    let mut results = Vec::new();
    
    for (i, text) in texts.iter().enumerate() {
        let result = sign_text(text, &provider, version, None)
            .await
            .with_context(|| format!("Failed to sign text at index {}", i))?;
        results.push(result);
    }
    
    Ok(results)
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
async def process_batch(texts: list[str], provider):
    results = []
    
    for i, text in enumerate(texts):
        try:
            result = await sign_text(text, provider=provider)
            results.append(result)
        except SigError as e:
            # Add context before re-raising
            raise type(e)(f"Failed to sign text at index {i}: {e}") from e
    
    return results
```

</TabItem>
</Tabs>

---

## Best Practices

### 1. Catch Specific Errors First

Always catch specific error types before catching the base `SigError`:

```python
try:
    result = await sign_text(text, provider=provider)
except InvalidInputError:
    # Handle input validation errors
    pass
except ProviderError:
    # Handle provider errors (maybe retry)
    pass
except SigError:
    # Handle all other errors
    pass
```

### 2. Use Type Guards (TypeScript)

```typescript
if (error instanceof ProviderError) {
  // TypeScript knows error.message is available
  console.error(`Provider error: ${error.message}`);
}
```

### 3. Don't Swallow Errors

Always log or handle errors appropriately:

```python
# ❌ BAD: Silently ignoring errors
try:
    result = await sign_text(text, provider=provider)
except:
    pass  # Error is lost!

# ✅ GOOD: Log errors
try:
    result = await sign_text(text, provider=provider)
except SigError as e:
    logger.error(f"Signature generation failed: {e}")
    raise  # Re-raise if caller should handle it
```

### 4. Clean Up Resources

Always close providers in finally blocks or use context managers:

<Tabs groupId="language">
<TabItem value="python" label="Python">

```python
provider = await OnnxProvider.new(cache)
try:
    result = await sign_text(text, provider=provider)
finally:
    await provider.close()  # Always clean up

# Or use async context manager (if implemented)
async with OnnxProvider.new(cache) as provider:
    result = await sign_text(text, provider=provider)
```

</TabItem>
</Tabs>

---

## See Also

- [Types](./types) - Error type definitions
- [Core Functions](./core-functions) - Functions that may raise errors
- [Providers](./providers) - Provider-specific error handling
- [Signature Format](./signature-format) - Parsing validation errors
