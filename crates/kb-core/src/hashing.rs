use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A 256-bit BLAKE3 hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hash(blake3::Hash);

impl Hash {
    /// Returns the raw bytes of the hash.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        self.0.as_bytes()
    }

    /// Returns the hex representation of the hash.
    #[must_use]
    pub fn to_hex(&self) -> String {
        self.0.to_string()
    }
}

impl PartialOrd for Hash {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hash {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_bytes().cmp(other.as_bytes())
    }
}

impl From<blake3::Hash> for Hash {
    fn from(h: blake3::Hash) -> Self {
        Self(h)
    }
}

impl From<[u8; 32]> for Hash {
    fn from(bytes: [u8; 32]) -> Self {
        Self(blake3::Hash::from(bytes))
    }
}

impl fmt::Display for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for Hash {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(blake3::Hash::from_str(s)?))
    }
}

impl Serialize for Hash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Hash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::from_str(&s).map_err(serde::de::Error::custom)
    }
}

/// Hashes a single byte slice.
#[must_use]
pub fn hash_bytes(data: &[u8]) -> Hash {
    Hash(blake3::hash(data))
}

/// Hashes the contents of a file.
///
/// # Errors
/// Returns an error if the file cannot be opened or read.
pub fn hash_file(path: impl AsRef<Path>) -> io::Result<Hash> {
    let mut file = File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0; 8192];

    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    Ok(Hash(hasher.finalize()))
}

/// Hashes multiple byte slices as a single stream.
#[must_use]
pub fn hash_many(parts: &[&[u8]]) -> Hash {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        hasher.update(part);
    }
    Hash(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_hash_bytes() {
        let data = b"hello world";
        let h1 = hash_bytes(data);
        let h2 = hash_bytes(data);
        assert_eq!(h1, h2);
        assert_eq!(h1.to_hex(), blake3::hash(data).to_string());
    }

    #[test]
    fn test_hash_many() {
        let parts: &[&[u8]] = &[b"hello ", b"world"];
        let h1 = hash_many(parts);
        let h2 = hash_bytes(b"hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_file() -> io::Result<()> {
        let mut tmp = NamedTempFile::new()?;
        tmp.write_all(b"hello world")?;
        
        let h1 = hash_file(tmp.path())?;
        let h2 = hash_bytes(b"hello world");
        assert_eq!(h1, h2);
        Ok(())
    }

    #[test]
    fn test_serialization() {
        let data = b"hello world";
        let h = hash_bytes(data);
        let json = serde_json::to_string(&h).expect("serialize Hash to JSON");
        assert_eq!(json, format!("\"{}\"", h.to_hex()));
        
        let h2: Hash = serde_json::from_str(&json).expect("deserialize JSON to Hash");
        assert_eq!(h, h2);
    }
}
