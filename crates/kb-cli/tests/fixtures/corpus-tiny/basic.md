# Getting Started with Rust

This is a basic guide to help you begin your journey with the Rust programming language.

## Installation

Visit [rust-lang.org](https://www.rust-lang.org/) and download the Rust toolchain installer.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## First Program

Create a new project:

```bash
cargo new hello_world
cd hello_world
```

Your `main.rs` should contain:

```rust
fn main() {
    println!("Hello, world!");
}
```

## Key Concepts

- **Ownership** - Memory is managed through ownership rules
- **Borrowing** - Allows temporary access without taking ownership
- **Lifetimes** - Ensure references are valid
