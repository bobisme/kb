# Long Tokens & Identifiers

This document tests handling of very long tokens commonly found in technical documentation.

## URLs & Endpoints

API endpoint for user creation:
```
https://api.example.com/v2/users/management/create?api_key=sk_live_51H2Lr2CDBmqPHVPz1V7Q8xW9yZ0aB1cD2eF3gH4iJ5kL6mN7oP8qR9sT0uV1wX2yZ3&request_id=req_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p
```

## Cryptographic Hashes

SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

MD5: `5d41402abc4b2a76b9719d911017c592`

Blake3: `af1349b9f5f9a1a6a0404dea36dcc9499bcb25c0bacc216b570c09c282c2ff7`

## Package Identifiers

Cargo crate: `tokio = { version = "=1.35.0", default-features = false, features = ["full"] }`

NPM package: `@anthropic-ai/sdk@0.28.0-20240304003532-39c9e94e7f8c`

## Long Method Names

Java method: `java.lang.reflect.AccessibleObject.setAccessible(AccessibleObject.java:347)`

Rust generic: `HashMap<String, Vec<Result<Option<Box<dyn Fn(i32) -> i32>>>>>`

## File Paths

Long Unix path:
```
/home/user/.cargo/registry/cache/github.com-1ecc6299db8cc6/tokio-1.35.0/src/runtime/metrics/mod.rs
```
