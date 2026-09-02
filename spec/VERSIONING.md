# Signature Versioning Specification

This document defines the signature version scheme, version registry, and compatibility rules for odin-prompt-toolkit signatures.

## Version: 1.0.0

Last updated: 2026-02-24

---

## 1. Version String Format

Signatures are encoded as version-prefixed strings:

```
0din-v{N}:<hex_signature>
```

Where:
- `0din-` is the fixed prefix
- `v{N}` is the version identifier (`v0`, `v1`, etc.)
- `<hex_signature>` is the lowercase hex-encoded LSH signature

### Examples

```
0din-v0:a3f9c2e1b8d4f7a2c5e8b1d3f6a9c2e5b8d1f4a7c2e5b8d1f4a7c2e5b8d1f4a7c2
0din-v1:7f2c8a9d3e1b5f4c8a2d6e9b1f3c5a7d2e6b9c1f4a8d3e7b2f5c9a1d4e8b3f6c
```

---

## 2. Version Registry

| Version | Provider | Model                                 | Dimensions | Signature Bits | Algorithm | Status       |
|---------|----------|---------------------------------------|------------|----------------|-----------|--------------|
| V0      | OpenAI   | text-embedding-3-large                | 1536       | 256            | LSH       | Stable       |
| V1      | ONNX     | 0din-jailbreak-embeddings-small | 1024       | 256            | LSH       | Stable       |
| Latest  | →V1      | —                                     | —          | —              | —         | Alias        |

### Version Descriptions

#### V0: OpenAI text-embedding-3-large

- **Provider**: OpenAI API
- **Model**: `text-embedding-3-large`
- **Dimensions**: 1536
- **Signature**: 256-bit LSH (3 families, 16 bands)
- **API Key**: Required
- **Cost**: Per-request API charges
- **Use Case**: High-quality embeddings, production workloads with API access

#### V1: ONNX 0din-jailbreak-embeddings-small

- **Provider**: Local ONNX inference
- **Model**: `0dinai/0din-jailbreak-embeddings-small` (custom 0din-threat-feed fine-tuned variant)
- **Dimensions**: 1024
- **Signature**: 256-bit LSH (3 families, 16 bands)
- **API Key**: Not required
- **Cost**: Free (local inference)
- **Use Case**: Local/offline deployments, cost-sensitive applications, privacy-first scenarios

#### Latest

- **Resolves to**: V1 (as of this specification version)
- **Usage**: Applications using `version: "latest"` will automatically get V1 signatures
- **Breaking Change**: Previously resolved to V0 (before V1 implementation)

---

## 3. Version Resolution

### Resolution Rules

```
resolve_version(v: Version) -> Version:
  if v == Latest:
    return V1  # Current resolution
  else:
    return v
```

### Resolution Table

| Input    | Resolved |
|----------|----------|
| V0       | V0       |
| V1       | V1       |
| Latest   | V1       |

---

## 4. Cross-Version Compatibility

### ⚠️ CRITICAL: V0 and V1 Signatures Are NOT Comparable

**V0 and V1 use different embedding spaces and dimensions:**

- V0: 1536-dimensional OpenAI space
- V1: 1024-dimensional 0din-jailbreak-embeddings-small space

**Implications:**

1. **Cannot compute meaningful similarity** between V0 and V1 signatures
2. **Cannot compare** V0 signature with V1 signature (even if you compute Hamming distance, the result is meaningless)
3. **Must regenerate** all signatures in the same version to compare them

### Valid Comparisons

| Signature A | Signature B | Can Compare? | Reason                            |
|-------------|-------------|--------------|-----------------------------------|
| V0          | V0          | ✅ Yes       | Same embedding space              |
| V1          | V1          | ✅ Yes       | Same embedding space              |
| V0          | V1          | ❌ No        | Different embedding spaces        |
| V1          | Latest      | ✅ Yes       | Latest resolves to V1             |
| V0          | Latest      | ❌ No        | Latest is V1, different from V0   |

---

## 5. Parsing Specification

### Grammar

```ebnf
signature_string = prefix version_tag ":" hex_signature ;
prefix           = "0din-" ;
version_tag      = "v" version_number ;
version_number   = digit+ ;
hex_signature    = hex_char+ ;
hex_char         = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | 
                   "8" | "9" | "a" | "b" | "c" | "d" | "e" | "f" ;
```

### Parsing Algorithm

```
parse_signature_string(s: string) -> ParsedSignature:
  # 1. Check prefix
  if !starts_with(s, "0din-"):
    return error("Invalid signature format: must start with '0din-'")
  
  # 2. Split into parts
  parts = split(s, ':', max_splits=2)
  if len(parts) != 2:
    return error("Invalid signature format: missing components")
  
  version_str = parts[0]  # "0din-v0" or "0din-v1"
  signature = parts[1]    # hex string
  
  # 3. Parse version
  if version_str == "0din-v0":
    return ParsedSignature { version: V0, signature: signature }
  elif version_str == "0din-v1":
    return ParsedSignature { version: V1, signature: signature }
  else:
    return error("Unsupported signature version: " + version_str)
```

### Validation Rules

1. **Prefix**: Must be exactly `"0din-"`
2. **Version**: Must be `v0` or `v1` (lowercase)
3. **Separator**: Must be exactly one `:`
4. **Signature**: Must be valid lowercase hex (0-9, a-f)
5. **Length**: For V0 and V1, signature should be 64 hex chars (256 bits), but implementations MAY accept other lengths

### Error Messages

| Error                          | Message                                                  |
|--------------------------------|----------------------------------------------------------|
| Invalid prefix                 | `Invalid signature format: must start with '0din-'`      |
| Missing components             | `Invalid signature format: missing components`           |
| Unsupported version            | `Unsupported signature version: 0din-v99`                |
| Invalid hex                    | `Invalid hex character in signature: 'g'`                |

---

## 6. Formatting Specification

### Formatting Algorithm

```
format_signature_string(version: Version, signature: string) -> string:
  resolved = resolve_version(version)
  
  if resolved == V0:
    return "0din-v0:" + signature
  elif resolved == V1:
    return "0din-v1:" + signature
  else:
    error("Cannot format unresolved version")
```

### Formatting Rules

1. **Version resolution**: `Latest` must be resolved before formatting
2. **Lowercase**: All hex characters must be lowercase
3. **No whitespace**: No spaces or newlines in the signature string
4. **Exact format**: Must match `0din-v{N}:<hex>`

---

## 7. Version Migration

### Migrating from V0 to V1

**When to migrate:**
- Cost reduction (eliminate OpenAI API charges)
- Privacy requirements (keep embeddings local)
- Offline deployment (no internet access)

**Migration process:**

1. **Regenerate signatures**: Generate V1 signatures for all existing content
2. **Update storage**: Replace V0 signature strings with V1 signature strings
3. **Update queries**: Ensure similarity search uses V1 signatures only
4. **Deprecate V0**: Remove V0 signatures once migration is complete

**⚠️ Cannot do gradual migration:** Because V0 and V1 are incompatible, you cannot have both versions in the same similarity index. Choose one version and regenerate all signatures.

### Migrating from Legacy Formats

If you have signatures in non-versioned formats (e.g., raw hex strings without `0din-` prefix):

1. **Determine version**: Identify which version the signature corresponds to (based on model/dimensions)
2. **Add prefix**: Prepend `0din-v0:` or `0din-v1:` as appropriate
3. **Validate**: Ensure the signature passes parsing validation

---

## 8. Future Version Extension

### Adding a New Version

When adding a new signature version (e.g., V2):

1. **Update registry**: Add V2 row to the version registry table
2. **Update resolution**: Decide if `Latest` should resolve to V2 (breaking change)
3. **Update parser**: Add V2 case to parsing algorithm
4. **Update formatter**: Add V2 case to formatting algorithm
5. **Add test vectors**: Generate V2 test vectors
6. **Document differences**: Clearly document what changed from V1 to V2

### Compatibility Considerations

- New versions SHOULD NOT break existing V0/V1 signature parsing
- New versions SHOULD use the same string format (`0din-v{N}:<hex>`)
- New versions MAY use different signature lengths (update grammar if needed)
- New versions SHOULD document compatibility with previous versions

### Example Future Version

```
V2:
  Provider: ONNX
  Model: 0din-jailbreak-embeddings-small
  Dimensions: 1024
  Signature: 512-bit CM-LSH (dual hash with confidence)
  Format: 0din-v2:<hash_a>:<hash_b>
  Compatible with: None (new embedding space)
```

---

## 9. API Integration

### Request Parameters

Most APIs accept a `version` parameter:

```json
{
  "prompt": "Hello world",
  "version": "v1"
}
```

Valid values: `"v0"`, `"v1"`, `"latest"`

### Response Format

Responses include the resolved version:

```json
{
  "signature": "0din-v1:7f2c8a9d...",
  "version": "v1",
  ...
}
```

### Version Defaults

If no `version` parameter is provided:
- Default: `"latest"` (currently resolves to V1)
- Clients SHOULD explicitly specify version for production use

---

## 10. Security Considerations

### Version Confusion Attacks

**Risk**: An attacker could craft a V0 signature that looks similar to a legitimate V1 signature, hoping to bypass deduplication.

**Mitigation**: 
- Always validate the version prefix before comparison
- Never compare signatures with different version tags
- Store version alongside signature in databases

### Version Downgrade Attacks

**Risk**: An attacker could force a system to use an older, potentially weaker version.

**Mitigation**:
- Pin version in production configurations
- Monitor for unexpected version changes
- Audit signature version distribution

---

## 11. References

- V0 Implementation: `odin-prompt-toolkit (formerly heimdall-core)` (Rust)
- V1 Implementation: `odin-prompt-toolkit (formerly heimdall-core)` (Rust)
- V1 Model: `models/v1/` directory
- Algorithm Specification: `spec/SPEC.md`
