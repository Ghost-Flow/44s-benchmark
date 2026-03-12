# 44s Benchmark

**Don't believe the speedups? Run it yourself.**

This repo contains benchmark tools to independently verify [44s Cloud](https://44s.io) performance claims. No trust required — just math.

Everything runs **locally on your machine**. No network calls to our servers. No signup. No API key. Just `cargo run --release` and optionally your own Redis instance.

## Results (128-core AWS c6i.metal)

### Cache — 44s (in-process DashMap) vs Redis (localhost TCP)

| Threads | 44s Cache | Redis | Speedup |
|---------|-----------|-------|---------|
| 1 | 18.19M ops/s | 78K ops/s | **233×** |
| 8 | 18.35M ops/s | 78K ops/s | **235×** |
| 16 | 32.62M ops/s | 78K ops/s | **418×** |
| 32 | 58.47M ops/s | 78K ops/s | **749×** |
| 64 | 97.85M ops/s | 78K ops/s | **1,253×** |
| 128 | 149.23M ops/s | 78K ops/s | **1,910×** |

> **Methodology note:** The cache benchmark compares in-process concurrent hashmap operations against Redis over localhost TCP. This is intentional — it measures the combined overhead of network I/O, serialization, and protocol parsing that external services require. 44s eliminates that overhead by embedding the data structure in-process. If you want a pure data-structure comparison, see the KV cache benchmark below which compares two in-process implementations.

### Database — 44s (lock-free SkipMap) vs PostgreSQL

| Threads | 44s Database | PostgreSQL* | Speedup |
|---------|-------------|-------------|---------|
| 1 | 3.33M ops/s | 50K ops/s | **67×** |
| 16 | 13.06M ops/s | 50K ops/s | **261×** |
| 64 | 17.31M ops/s | 50K ops/s | **346×** |
| 128 | 14.77M ops/s | 50K ops/s | **295×** |

### Queue — 44s (lock-free SegQueue) vs RabbitMQ

| Threads | 44s Queue | RabbitMQ* | Speedup |
|---------|----------|-----------|---------|
| 1 | 6.08M msg/s | 50K msg/s | **122×** |
| 16 | 6.79M msg/s | 50K msg/s | **136×** |
| 64 | 5.43M msg/s | 50K msg/s | **109×** |

### AI Inference KV Cache — Lock-Free Atomics vs RwLock (apples-to-apples)

| Threads | Traditional (ms) | 44s Fractal (ms) | Speedup |
|---------|-----------------|-------------------|---------|
| 1 | 149 | 8 | **17×** |
| 8 | 2,845 | 11 | **242×** |
| 32 | 7,606 | 12 | **592×** |
| 64 | 20,160 | 21 | **930×** |
| 128 | 49,870 | 46 | **1,078×** |

> Traditional KV cache gets **334× slower** from 1→128 threads. 44s stays flat. This is a pure in-process comparison — no network overhead on either side.

*\*PostgreSQL baseline: ~50K ops/sec (pgbench, 16 clients, default config). RabbitMQ baseline: ~50K msgs/sec (tuned, persistent, acknowledged). Install them locally and run your own benchmarks to verify.*

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) (1.70+)
- Redis (optional, for live comparison — otherwise uses industry baseline)

### Install & Run

```bash
git clone https://github.com/Ghost-Flow/44s-benchmark.git
cd 44s-benchmark
cargo run --release
```

That's it. Takes ~60 seconds depending on core count.

### With Redis (recommended)

```bash
# macOS
brew install redis && redis-server --daemonize yes

# Ubuntu/Debian
sudo apt install redis-server && sudo systemctl start redis

# Then run the benchmark
cargo run --release
```

## Why the massive difference?

Traditional infrastructure has three bottlenecks that compound under load:

1. **Single-threaded design** — Redis processes all commands on a single core. On a 128-core server, that's 99%+ of your hardware sitting idle.
2. **Network + serialization overhead** — Every operation requires TCP roundtrips, protocol parsing, and serialization/deserialization. Even on localhost, this adds microseconds per operation.
3. **Lock contention** — Systems that do attempt concurrency (PostgreSQL row locks, RabbitMQ queue coordination) hit contention that degrades throughput as threads increase.

```
Traditional infrastructure:
  Thread 1: ████░░░░░░░░░░  (working, then blocked on I/O or locks...)
  Thread 2: ░░░░████░░░░░░  (waiting, then working...)
  Thread 3: ░░░░░░░░████░░  (waiting, waiting, working...)
  Thread 4: ░░░░░░░░░░░░██  (waiting, waiting, waiting...)
  
  More threads = diminishing returns

44s (concurrent, in-process):
  Thread 1: ██████████████  (always working)
  Thread 2: ██████████████  (always working)
  Thread 3: ██████████████  (always working)
  Thread 4: ██████████████  (always working)
  
  More threads = more throughput = LINEAR SCALING
```

44s eliminates all three bottlenecks by embedding concurrent data structures directly in-process — [DashMap](https://docs.rs/dashmap) (sharded concurrent hashmap with minimal locking), [SkipMap](https://docs.rs/crossbeam-skiplist) (lock-free sorted map), [SegQueue](https://docs.rs/crossbeam-queue) (lock-free MPMC queue) — no network hops, no serialization, and near-zero contention across all cores.

## Core count matters

The speedup scales with core count because that's where lock contention becomes catastrophic:

| Your Machine | Expected Cache Speedup vs Redis |
|-------------|-------------------------------|
| Laptop (4-8 cores) | ~50-200× |
| Cloud instance (16-32 cores) | ~400-750× |
| Dedicated server (64-96 cores) | ~1,000-1,500× |
| Metal instance (128+ cores) | ~1,900×+ |

## What's under the hood

This benchmark uses three open-source Rust crates:

- **[dashmap](https://crates.io/crates/dashmap)** — Sharded concurrent HashMap (Cache benchmark)
- **[crossbeam-skiplist](https://crates.io/crates/crossbeam-skiplist)** — Lock-free concurrent sorted map (Database benchmark)
- **[crossbeam-queue](https://crates.io/crates/crossbeam-queue)** — Lock-free MPMC queue (Queue benchmark)

The AI inference KV cache benchmark uses raw `AtomicU64` operations with cache-line-aligned slots.

44s Cloud is built on these same primitives, wrapped in production-grade services with persistence, replication, auth, billing, TLS, and clustering.

## Methodology

- All benchmarks run on the same machine (your machine)
- Tests measure **concurrent** operations (not sequential)
- Mix of reads and writes (50/50 for cache, 25/25/25/25 for database)
- Pre-populated data to avoid cold-start bias
- Release mode with LTO for accurate results

## FAQ

**Q: This seems too good to be true.**
A: We thought so too. That's why this repo exists. Run it yourself.

**Q: Why not compare against Dragonfly/KeyDB/other multi-threaded Redis alternatives?**
A: Great idea — feel free to add comparisons and open a PR. We benchmark against Redis because it's the industry standard that most teams use.

**Q: I got different numbers.**
A: Expected! Results scale with your core count. The ratio should be consistent — check the "Core count matters" table above.

**Q: Why is the benchmark code open source but not 44s itself?**
A: This benchmark proves the thesis (lock-free >> mutex-based). The production platform adds persistence, clustering, auth, billing, and 17 integrated services — that's the product.

## Verify Independently

Don't trust our benchmark code? Write your own:

```rust
use dashmap::DashMap;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

fn main() {
    let map: Arc<DashMap<u64, Vec<u8>>> = Arc::new(DashMap::new());
    let threads = num_cpus::get();
    let ops_per_thread = 1_000_000u64;
    
    let start = Instant::now();
    
    let handles: Vec<_> = (0..threads).map(|t| {
        let map = map.clone();
        thread::spawn(move || {
            for i in 0..ops_per_thread {
                let key = (t as u64 * ops_per_thread + i) % 10_000;
                if i % 2 == 0 { map.insert(key, vec![0u8; 100]); }
                else { map.get(&key); }
            }
        })
    }).collect();
    
    for h in handles { h.join().unwrap(); }
    
    let elapsed = start.elapsed();
    let total = threads as u64 * ops_per_thread;
    println!("{} ops in {:.2}s = {} ops/sec",
        total, elapsed.as_secs_f64(),
        (total as f64 / elapsed.as_secs_f64()) as u64
    );
    println!("Now compare that to `redis-benchmark -t set,get -n 100000`");
}
```

## License

MIT — do whatever you want with this benchmark code.

44s Cloud is proprietary and [patent-pending](https://44s.io).

## Contact

- **Twitter/X:** [@builtbyZach](https://twitter.com/builtbyZach)
- **Website:** [44s.io](https://44s.io)

---

Still skeptical? Good. **[Run the benchmark →](https://44s.io)**
