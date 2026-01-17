# 44s Benchmark

**Don't believe 450× faster? Run it yourself.**

This repo contains benchmark tools to independently verify 44s Cloud performance claims. No trust required — just math.

## What This Tests

| Benchmark | Comparison | Claimed Speedup |
|-----------|------------|-----------------|
| Cache | 44s vs Redis | 450× |
| Serverless | 44s vs AWS Lambda | 40,000× |
| Database | 44s vs PostgreSQL | 47× |

## Quick Start

### Prerequisites
- Python 3.8+
- Redis (local, for baseline comparison)
- A 44s API key (get one at [44s.io](https://44s.io))

### Install

```bash
git clone https://github.com/Ghost-Flow/44s-benchmark.git
cd 44s-benchmark
pip install -r requirements.txt
```

### Run the Benchmark

```bash
# Set your API key
export FORTY_FOURS_API_KEY="your_api_key_here"

# Run cache benchmark (Redis vs 44s)
python benchmark.py cache

# Run all benchmarks
python benchmark.py all

# Run with custom parameters
python benchmark.py cache --requests 100000 --concurrency 64
```

## Understanding the Results

The benchmark measures **operations per second under contention** — the scenario where traditional systems fall apart.

```
=== CACHE BENCHMARK RESULTS ===
Local Redis:     12,450 ops/sec
44s Cache:    5,602,500 ops/sec
Speedup:            450×
```

### Why the massive difference?

Traditional systems (Redis, PostgreSQL, etc.) use **mutex locks** for thread safety. Under high concurrency, threads spend most of their time waiting for locks instead of doing work.

44s uses **lock-free architecture** — no mutexes, no waiting, linear scaling with cores.

## Verify Independently

Don't trust our benchmark code? Write your own:

```python
import redis
import requests
import time

# Your local Redis
r = redis.Redis()

# Benchmark Redis
start = time.time()
for i in range(10000):
    r.set(f"key{i}", f"value{i}")
redis_time = time.time() - start

# Benchmark 44s (Redis-compatible API)
# Use your preferred HTTP client
start = time.time()
for i in range(10000):
    requests.post(
        "https://api.44s.io/SET",
        json={"key": f"key{i}", "value": f"value{i}"},
        headers={"X-API-Key": "your_key"}
    )
forty_fours_time = time.time() - start

print(f"Redis: {10000/redis_time:.0f} ops/sec")
print(f"44s: {10000/forty_fours_time:.0f} ops/sec")
```

## Methodology

- All benchmarks run against **the same hardware class** (c6a.24xlarge, 96 vCPUs)
- Tests measure **concurrent operations** (not sequential)
- Results are median of 5 runs
- Network latency is measured and subtracted for fair comparison

## FAQ

**Q: This seems too good to be true.**
A: We thought so too. Run the benchmark yourself.

**Q: Why is the benchmark code open source but not 44s itself?**
A: The benchmark measures performance — it doesn't reveal implementation. You can verify our claims without seeing proprietary code.

**Q: I got different numbers.**
A: Expected! Results vary by network conditions, hardware, and load. The ratio should remain consistent.

## License

MIT — do whatever you want with this benchmark code.

The 44s Cloud service itself is proprietary and protected by [patents](https://44s.io/verify.html).

---

**Still skeptical?** Good. [Run the benchmark](https://44s.io) →

