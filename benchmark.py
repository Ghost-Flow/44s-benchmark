#!/usr/bin/env python3
"""
44s Benchmark Suite
Verify 44s performance claims yourself. Don't believe us? Run it.
"""

import asyncio
import aiohttp
import redis
import time
import os
import statistics
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from concurrent.futures import ThreadPoolExecutor
import threading

console = Console()

# Configuration
FORTY_FOURS_API = "https://api.44s.io"
API_KEY = os.environ.get("FORTY_FOURS_API_KEY", "")

class BenchmarkResults:
    def __init__(self, name: str, ops: int, duration: float):
        self.name = name
        self.ops = ops
        self.duration = duration
        self.ops_per_sec = ops / duration if duration > 0 else 0

def print_banner():
    console.print("""
[cyan]╔═══════════════════════════════════════════════════════════╗
║                    44s BENCHMARK                          ║
║         Don't believe 450×? Run it yourself.              ║
╚═══════════════════════════════════════════════════════════╝[/cyan]
""")

# ============================================================================
# REDIS BENCHMARK (LOCAL)
# ============================================================================

def benchmark_redis_local(requests: int, concurrency: int) -> BenchmarkResults:
    """Benchmark local Redis with concurrent operations."""
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
    except redis.ConnectionError:
        console.print("[red]Error: Cannot connect to local Redis. Is it running?[/red]")
        console.print("  Install: brew install redis && redis-server")
        return BenchmarkResults("Redis (local)", 0, 1)
    
    counter = threading.atomic = 0
    lock = threading.Lock()
    completed = [0]
    
    def worker(thread_id: int, ops_per_thread: int):
        for i in range(ops_per_thread):
            key = f"bench:{thread_id}:{i}"
            r.set(key, f"value_{i}")
            r.get(key)
            with lock:
                completed[0] += 2  # SET + GET = 2 ops
    
    ops_per_thread = requests // concurrency
    
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, i, ops_per_thread) for i in range(concurrency)]
        for f in futures:
            f.result()
    duration = time.perf_counter() - start
    
    # Cleanup
    r.flushdb()
    
    return BenchmarkResults("Redis (local)", completed[0], duration)

# ============================================================================
# 44s BENCHMARK
# ============================================================================

async def benchmark_44s_cache(requests: int, concurrency: int) -> BenchmarkResults:
    """Benchmark 44s Cache with concurrent operations."""
    if not API_KEY:
        console.print("[red]Error: FORTY_FOURS_API_KEY environment variable not set[/red]")
        console.print("  Get your API key at https://44s.io/dashboard")
        return BenchmarkResults("44s Cache", 0, 1)
    
    completed = 0
    
    async def worker(session: aiohttp.ClientSession, worker_id: int, ops: int):
        nonlocal completed
        headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
        
        for i in range(ops):
            key = f"bench:{worker_id}:{i}"
            
            # SET
            async with session.post(
                f"{FORTY_FOURS_API}/cache/set",
                json={"key": key, "value": f"value_{i}"},
                headers=headers
            ) as resp:
                if resp.status == 200:
                    completed += 1
            
            # GET
            async with session.get(
                f"{FORTY_FOURS_API}/cache/get/{key}",
                headers=headers
            ) as resp:
                if resp.status == 200:
                    completed += 1
    
    ops_per_worker = requests // concurrency
    
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=300)
    
    start = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [worker(session, i, ops_per_worker) for i in range(concurrency)]
        await asyncio.gather(*tasks)
    duration = time.perf_counter() - start
    
    return BenchmarkResults("44s Cache", completed, duration)

# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

def run_cache_benchmark(requests: int, concurrency: int):
    """Run cache benchmark comparing Redis vs 44s."""
    console.print(f"\n[bold]Cache Benchmark[/bold]")
    console.print(f"  Requests: {requests:,}")
    console.print(f"  Concurrency: {concurrency}")
    console.print()
    
    # Benchmark Redis
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Benchmarking local Redis...", total=None)
        redis_results = benchmark_redis_local(requests, concurrency)
    
    console.print(f"  [green]✓[/green] Redis: {redis_results.ops_per_sec:,.0f} ops/sec")
    
    # Benchmark 44s
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Benchmarking 44s Cache...", total=None)
        fortyfours_results = asyncio.run(benchmark_44s_cache(requests, concurrency))
    
    console.print(f"  [green]✓[/green] 44s:   {fortyfours_results.ops_per_sec:,.0f} ops/sec")
    
    # Results
    if redis_results.ops_per_sec > 0 and fortyfours_results.ops_per_sec > 0:
        speedup = fortyfours_results.ops_per_sec / redis_results.ops_per_sec
        
        console.print()
        table = Table(title="Cache Benchmark Results", show_header=True)
        table.add_column("System", style="cyan")
        table.add_column("Ops/sec", justify="right", style="green")
        table.add_column("Speedup", justify="right", style="yellow")
        
        table.add_row("Redis (local)", f"{redis_results.ops_per_sec:,.0f}", "1×")
        table.add_row("44s Cache", f"{fortyfours_results.ops_per_sec:,.0f}", f"{speedup:.0f}×")
        
        console.print(table)
        
        if speedup >= 100:
            console.print(f"\n[bold green]✓ Verified: 44s is {speedup:.0f}× faster than Redis[/bold green]")
        else:
            console.print(f"\n[yellow]Note: Speedup of {speedup:.0f}× is lower than claimed 450×[/yellow]")
            console.print("[dim]This can happen due to network latency or low concurrency.[/dim]")
            console.print("[dim]Try increasing --concurrency for more realistic results.[/dim]")

# ============================================================================
# CLI
# ============================================================================

@click.group()
def cli():
    """44s Benchmark Suite - Verify performance claims yourself."""
    pass

@cli.command()
@click.option('--requests', '-r', default=10000, help='Number of requests')
@click.option('--concurrency', '-c', default=32, help='Concurrent connections')
def cache(requests, concurrency):
    """Benchmark 44s Cache vs Redis."""
    print_banner()
    run_cache_benchmark(requests, concurrency)

@cli.command()
@click.option('--requests', '-r', default=10000, help='Number of requests')
@click.option('--concurrency', '-c', default=32, help='Concurrent connections')
def all(requests, concurrency):
    """Run all benchmarks."""
    print_banner()
    run_cache_benchmark(requests, concurrency)
    # TODO: Add serverless, database benchmarks

@cli.command()
def check():
    """Check prerequisites."""
    print_banner()
    console.print("[bold]Checking prerequisites...[/bold]\n")
    
    # Check Redis
    try:
        r = redis.Redis()
        r.ping()
        console.print("[green]✓[/green] Redis: Connected")
    except:
        console.print("[red]✗[/red] Redis: Not running")
        console.print("  Install: brew install redis && redis-server")
    
    # Check API key
    if API_KEY:
        console.print("[green]✓[/green] API Key: Set")
    else:
        console.print("[red]✗[/red] API Key: Not set")
        console.print("  Set: export FORTY_FOURS_API_KEY='your_key'")
    
    # Check 44s API
    try:
        import requests as req
        resp = req.get(f"{FORTY_FOURS_API}/health", timeout=5)
        if resp.status_code == 200:
            console.print("[green]✓[/green] 44s API: Reachable")
        else:
            console.print("[yellow]![/yellow] 44s API: Unexpected status")
    except:
        console.print("[red]✗[/red] 44s API: Unreachable")

if __name__ == "__main__":
    cli()

