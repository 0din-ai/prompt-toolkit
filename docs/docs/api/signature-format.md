---
sidebar_position: 5
---

# Signature Format

Specification and API for the canonical signature string format.

## Format Specification

All 0din signatures use a standardized string format that encodes version and hex signature:

```
0din-v{N}:{hex_signature}
```

**Structure:**
- **Prefix**: `0din-` (lowercase, fixed)
- **Version tag**: `v` + version number (e.g., `v0`, `v1`)
- **Separator**: `:` (colon)
- **Hex signature**: Lowercase hex string (64 chars for 256-bit signatures)

**Examples:**
```
0din-v0:abc1234567890def...  (OpenAI, 1536-dim, 64 hex chars)
0din-v1:8d000000ac854dae...  (ONNX, 1024-dim, 64 hex chars)
```

---

## EBNF Grammar

Formal grammar specification:

```ebnf
signature_string = prefix version_tag ":" hex_signature ;
prefix           = "0din-" ;
version_tag      = "v" version_number ;
version_number   = digit+ ;
hex_signature    = hex_char+ ;
hex_char         = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | 
                   "8" | "9" | "a" | "b" | "c" | "d" | "e" | "f" ;
digit            = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
```

**Constraints:**
- `prefix` is case-sensitive (must be lowercase `"0din-"`)
- `hex_char` must be lowercase (`a-f`, not `A-F`)
- `hex_signature` length depends on LSH configuration (default: 64 chars = 256 bits)

---

## API Functions

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

### signatureString / signature_string

Generate formatted signature string from version and hex signature.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub fn signature_string(version: SignatureVersion, signature: &str) -> String
```

**Example:**
```rust
use odin_prompt_toolkit::{signature_string, SignatureVersion};

let sig = "8d000000ac854dae7f3b9c1e...";
let formatted = signature_string(SignatureVersion::V1, sig);
// Result: "0din-v1:8d000000ac854dae7f3b9c1e..."
```

</TabItem>
<TabItem value="python" label="Python">

```python
def signature_string(version: SignatureVersion, signature: str) -> str
```

**Example:**
```python
from odin_prompt_toolkit import signature_string, SignatureVersion

sig = "8d000000ac854dae7f3b9c1e..."
formatted = signature_string(SignatureVersion.V1, sig)
# Result: "0din-v1:8d000000ac854dae7f3b9c1e..."
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
function signatureString(version: SignatureVersion, signature: string): string
```

**Example:**
```typescript
import { signatureString, SignatureVersion } from '@0din/odin-prompt-toolkit';

const sig = "8d000000ac854dae7f3b9c1e...";
const formatted = signatureString(SignatureVersion.V1, sig);
// Result: "0din-v1:8d000000ac854dae7f3b9c1e..."
```

</TabItem>
</Tabs>

**Parameters:**
- `version`: Signature version enum (`V0`, `V1`, or `LATEST`)
- `signature`: Hex-encoded signature string (lowercase, no prefix)

**Returns:**
- Formatted signature string with `0din-v{N}:` prefix

**Note:** `LATEST` resolves to the current latest version (`v1`)

---

### parseSignatureString / parse_signature_string

Parse formatted signature string into version and hex signature components.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub fn parse_signature_string(s: &str) -> Result<ParsedSignature, SigError>
```

**Example:**
```rust
use odin_prompt_toolkit::parse_signature_string;

let parsed = parse_signature_string("0din-v1:8d000000ac854dae...")?;
println!("Version: {:?}", parsed.version);  // V1
println!("Signature: {}", parsed.signature); // "8d000000ac854dae..."
```

**Errors:**
```rust
// Invalid prefix
parse_signature_string("invalid")?;
// Error: InvalidInput("Invalid signature prefix: invalid")

// Invalid version
parse_signature_string("0din-v99:abc123")?;
// Error: InvalidInput("Unsupported signature version: v99")

// Non-hex characters
parse_signature_string("0din-v1:xyz123")?;
// Error: InvalidInput("Invalid hex signature: xyz123")
```

</TabItem>
<TabItem value="python" label="Python">

```python
def parse_signature_string(s: str) -> ParsedSignature
```

**Example:**
```python
from odin_prompt_toolkit import parse_signature_string

parsed = parse_signature_string("0din-v1:8d000000ac854dae...")
print(f"Version: {parsed.version}")    # V1
print(f"Signature: {parsed.signature}") # "8d000000ac854dae..."
```

**Exceptions:**
```python
# Invalid prefix
parse_signature_string("invalid")
# Raises: InvalidInputError("Invalid signature prefix: invalid")

# Invalid version
parse_signature_string("0din-v99:abc123")
# Raises: InvalidInputError("Unsupported signature version: v99")

# Non-hex characters
parse_signature_string("0din-v1:xyz123")
# Raises: InvalidInputError("Invalid hex signature: xyz123")
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
function parseSignatureString(s: string): ParsedSignature
```

**Example:**
```typescript
import { parseSignatureString } from '@0din/odin-prompt-toolkit';

const parsed = parseSignatureString("0din-v1:8d000000ac854dae...");
console.log(`Version: ${parsed.version}`);    // "v1"
console.log(`Signature: ${parsed.signature}`); // "8d000000ac854dae..."
```

**Throws:**
```typescript
// Invalid prefix
parseSignatureString("invalid");
// Throws: InvalidInputError("Invalid signature prefix: invalid")

// Invalid version
parseSignatureString("0din-v99:abc123");
// Throws: InvalidInputError("Unsupported signature version: v99")

// Non-hex characters
parseSignatureString("0din-v1:xyz123");
// Throws: InvalidInputError("Invalid hex signature: xyz123")
```

</TabItem>
</Tabs>

**Returns:**
- `ParsedSignature` with `version` and `signature` fields

**Validation:**
1. Checks prefix is exactly `"0din-"`
2. Validates version tag matches known versions (`v0`, `v1`)
3. Validates hex signature contains only lowercase hex characters (`[0-9a-f]+`)

**Error Cases:**
- Missing or incorrect prefix → `InvalidInput`
- Unsupported version → `InvalidInput`
- Non-hex characters in signature → `InvalidInput`
- Missing colon separator → `InvalidInput`

---

## Version Compatibility

:::danger Version Incompatibility
Signatures from different versions are **NOT comparable** because they use different embedding models and dimensionalities.
:::

**Version Matrix:**

| Version | Model | Dimensions | Comparable With |
|---------|-------|------------|-----------------|
| V0 | OpenAI text-embedding-3-large | 1536 | V0 only |
| V1 | 0din-jailbreak-embeddings-small ONNX | 1024 | V1 only |

**Cross-Version Comparison:**
```typescript
// ❌ INCORRECT - will give meaningless results
const sig_v0 = "0din-v0:abc123...";
const sig_v1 = "0din-v1:def456...";
const distance = hammingDistanceHex(
  parseSignatureString(sig_v0).signature,
  parseSignatureString(sig_v1).signature
);
// Result is meaningless - different embedding spaces!
```

**Correct Usage:**
```typescript
// ✅ CORRECT - same version
const sig1 = "0din-v1:abc123...";
const sig2 = "0din-v1:def456...";
const distance = hammingDistanceHex(
  parseSignatureString(sig1).signature,
  parseSignatureString(sig2).signature
);
// Result is meaningful
```

---

## Storage Recommendations

### Database Storage

**Recommended Schema:**
```sql
CREATE TABLE prompts (
  id SERIAL PRIMARY KEY,
  text TEXT NOT NULL,
  signature TEXT NOT NULL,  -- Store full "0din-v1:..." format
  version VARCHAR(10) NOT NULL,  -- Extract for filtering (e.g., "v1")
  signature_hex VARCHAR(128) NOT NULL,  -- Extract for comparisons
  created_at TIMESTAMP DEFAULT NOW()
);

-- Index for version filtering
CREATE INDEX idx_prompts_version ON prompts(version);

-- Index for signature lookups
CREATE INDEX idx_prompts_signature ON prompts(signature);
```

**Benefits:**
- `signature` column: Human-readable, self-documenting
- `version` column: Fast version filtering
- `signature_hex` column: Optimized for Hamming distance queries

### Band-Based Indexing

For similarity search, extract and index band slices:

```sql
CREATE TABLE prompt_bands (
  prompt_id INTEGER REFERENCES prompts(id),
  family INTEGER NOT NULL,    -- 0-2 for default config
  band_index INTEGER NOT NULL, -- 0-15 for default config
  band_value VARCHAR(8) NOT NULL,  -- 4 hex chars per band
  PRIMARY KEY (prompt_id, family, band_index)
);

-- Index for LSH bucketing
CREATE INDEX idx_bands_lookup ON prompt_bands(family, band_index, band_value);
```

**Query Pattern:**
```sql
-- Find candidates with matching band 0 in family 0
SELECT DISTINCT p.id, p.signature_hex
FROM prompts p
JOIN prompt_bands b ON p.id = b.prompt_id
WHERE b.family = 0
  AND b.band_index = 0
  AND b.band_value = '8d00'  -- From query signature
  AND p.version = 'v1';      -- Same version only
```

---

## Validation Rules

### Format Validation

**Valid Signatures:**
```
✅ 0din-v1:8d000000ac854dae7f3b9c1e...
✅ 0din-v0:abc1234567890def1234567...
✅ 0din-v1:00000000000000000000000...  (all zeros is valid)
```

**Invalid Signatures:**
```
❌ 0DIN-v1:8d00...      (uppercase prefix)
❌ odin-v1:8d00...      (missing leading zero)
❌ 0din-V1:8d00...      (uppercase version)
❌ 0din-v1:8D00...      (uppercase hex)
❌ 0din-v1-8d00...      (wrong separator)
❌ 0din-v1:8d00xyz...   (non-hex characters)
❌ 0din-v99:8d00...     (unsupported version)
```

### Length Validation

**Default Configuration (256 bits):**
- Hex length: 64 characters (4 bits per hex char)
- Total signature string: 10 + 64 = 74 characters
  - `0din-v1:` = 8 chars (or 9 for `0din-v0:`)
  - Hex signature = 64 chars

**Custom Configurations:**
- 128 bits → 32 hex chars
- 512 bits → 128 hex chars

---

## Migration Guide

### From Legacy Formats

If migrating from legacy systems (heimdall, thor, research):

**Heimdall (Rust):**
```rust
// Old: Raw hex string
let old_sig = "8d000000ac854dae...";

// New: Prefixed format
let new_sig = format!("0din-v1:{}", old_sig);
```

**Thor (TypeScript):**
```typescript
// Old: { version: 1, signature: "8d00..." }
const old = { version: 1, signature: "8d000000..." };

// New: String format
const new_sig = `0din-v${old.version}:${old.signature}`;
```

### Version Migration (V0 → V1)

:::warning Requires Re-computation
Migrating from V0 to V1 requires **regenerating signatures** from original text. You cannot convert V0 signatures to V1.
:::

**Migration Process:**
1. Store original text alongside V0 signatures
2. Re-generate embeddings using V1 provider (ONNX)
3. Generate new V1 signatures
4. Maintain both versions during transition period
5. Deprecate V0 after full migration

---

## See Also

- [Types](./types) - `ParsedSignature` structure
- [Core Functions](./core-functions) - `signature_string()` and `parse_signature_string()`
- [Errors](./errors) - `InvalidInputError` details
- [Signature Versions](../concepts/signature-versions) - V0 vs V1 compatibility
- [VERSIONING.md](https://github.com/0din-ai/odin-prompt-toolkit/blob/main/spec/VERSIONING.md) - Complete specification
