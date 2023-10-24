Rust libraries and programs for solving [combination puzzles](https://en.wikipedia.org/wiki/Combination_puzzle), developed by [Piotr Beling](http://pbeling.w8.pl/).

Included libraries:
- `puzzles` ([crate](https://crates.io/crates/puzzles), [doc](https://docs.rs/puzzles)) - solves [combination puzzles](https://en.wikipedia.org/wiki/Combination_puzzle);

Included programs:
- `pdbs_benchmark` ([crate](https://crates.io/crates/pdbs_benchmark), [doc](https://docs.rs/pdbs_benchmark)) - a console program for testing the effectiveness of different pattern database implementations in solving [combination puzzles](https://en.wikipedia.org/wiki/Combination_puzzle).

# Installation
Programs can be compiled and installed from sources. To do this, a Rust compiler is needed.
The easiest way to obtain the compiler along with other necessary tools (like `cargo`) is
to use [rustup](https://www.rust-lang.org/tools/install).

Once Rust is installed, to compile and install a program with native optimizations, just execute:

```RUSTFLAGS="-C target-cpu=native" cargo install <program_name>```

for example

```RUSTFLAGS="-C target-cpu=native" cargo install pdbs_benchmark```