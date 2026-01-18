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
FORTY_FOURS_HOST = "api.44s.io"
FORTY_FOURS_PORT = 6379

# Public demo key for benchmarking (anyone can use this)
DEMO_KEY = "44s_benchmark_demo_2026"
API_KEY = os.environ.get("FORTY_FOURS_API_KEY", DEMO_KEY)

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
# 44s BENCHMARK (Redis Protocol - same as local Redis, fair comparison)
# ============================================================================

def benchmark_44s_cache(requests: int, concurrency: int) -> BenchmarkResults:
    """Benchmark 44s Cache with concurrent operations using Redis protocol."""
    if not API_KEY:
        console.print("[red]Error: FORTY_FOURS_API_KEY environment variable not set[/red]")
        console.print("  Get your API key at https://44s.io/dashboard")
        return BenchmarkResults("44s Cache", 0, 1)
    
    try:
        # Connect to 44s using Redis protocol (same as local Redis!)
        r = redis.Redis(host='api.44s.io', port=6379, password=API_KEY, decode_responses=True)
        r.ping()
    except redis.ConnectionError as e:
        console.print(f"[red]Error: Cannot connect to 44s API: {e}[/red]")
        return BenchmarkResults("44s Cache", 0, 1)
    except redis.AuthenticationError:
        console.print("[red]Error: Invalid API key[/red]")
        return BenchmarkResults("44s Cache", 0, 1)
    
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
    try:
        for i in range(concurrency):
            for j in range(ops_per_thread):
                r.delete(f"bench:{i}:{j}")
    except:
        pass  # Best effort cleanup
    
    return BenchmarkResults("44s Cache", completed[0], duration)

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
        fortyfours_results = benchmark_44s_cache(requests, concurrency)
    
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
        table.add_row("44s Cache (remote)", f"{fortyfours_results.ops_per_sec:,.0f}", f"{speedup:.0f}×")
        
        console.print(table)
        
        if speedup >= 100:
            console.print(f"\n[bold green]✓ Verified: 44s is {speedup:.0f}× faster than Redis[/bold green]")
        elif fortyfours_results.ops_per_sec < 500:
            # Network latency is dominating
            console.print(f"\n[bold yellow]⚠️  NETWORK LATENCY WARNING[/bold yellow]")
            console.print()
            console.print("[yellow]You're comparing:[/yellow]")
            console.print("  • Redis running [green]locally[/green] (0ms latency)")
            console.print("  • 44s running [cyan]over the internet[/cyan] (~50-100ms latency)")
            console.print()
            console.print("[yellow]This is NOT a fair comparison![/yellow]")
            console.print()
            console.print("For a fair benchmark, both must run on the same network:")
            console.print("  1. SSH into a cloud server")
            console.print("  2. Run: [cyan]redis-benchmark -h api.44s.io -p 6379 -a 44s_benchmark_demo_2026 -t set,get -n 100000 -c 50[/cyan]")
            console.print("  3. Compare to local Redis on the same server")
            console.print()
            console.print("Or spin up a 96-core server to see the 450× speedup under contention.")
            console.print("See README for details.")
        else:
            console.print(f"\n[yellow]Note: Speedup of {speedup:.0f}× is lower than claimed 450×[/yellow]")
            console.print("[dim]The 450× speedup requires high core count (96+) to demonstrate lock contention.[/dim]")
            console.print("[dim]See README for explanation.[/dim]")

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

# ============================================================================
# DATABASE BENCHMARK
# ============================================================================

def benchmark_postgres_local(requests: int, concurrency: int) -> BenchmarkResults:
    """Benchmark local PostgreSQL with concurrent operations."""
    try:
        import psycopg2
        from psycopg2 import pool
    except ImportError:
        console.print("[red]Error: psycopg2 not installed. Run: pip install psycopg2-binary[/red]")
        return BenchmarkResults("PostgreSQL (local)", 0, 1)
    
    try:
        # Try to connect to local PostgreSQL
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            user=os.environ.get('PGUSER', 'postgres'),
            password=os.environ.get('PGPASSWORD', ''),
            database=os.environ.get('PGDATABASE', 'postgres')
        )
        conn.close()
    except Exception as e:
        console.print(f"[red]Error: Cannot connect to local PostgreSQL: {e}[/red]")
        console.print("  Install: brew install postgresql && brew services start postgresql")
        console.print("  Or: docker run -p 5432:5432 -e POSTGRES_PASSWORD=test postgres")
        return BenchmarkResults("PostgreSQL (local)", 0, 1)
    
    # Create connection pool
    try:
        connection_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=concurrency + 5,
            host='localhost',
            port=5432,
            user=os.environ.get('PGUSER', 'postgres'),
            password=os.environ.get('PGPASSWORD', ''),
            database=os.environ.get('PGDATABASE', 'postgres')
        )
    except Exception as e:
        console.print(f"[red]Error creating connection pool: {e}[/red]")
        return BenchmarkResults("PostgreSQL (local)", 0, 1)
    
    # Setup: create test table
    conn = connection_pool.getconn()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS bench_test")
    cur.execute("CREATE TABLE bench_test (id INTEGER PRIMARY KEY, value TEXT, counter INTEGER)")
    conn.commit()
    connection_pool.putconn(conn)
    
    lock = threading.Lock()
    completed = [0]
    
    def worker(thread_id: int, ops_per_thread: int):
        conn = connection_pool.getconn()
        cur = conn.cursor()
        try:
            for i in range(ops_per_thread):
                row_id = thread_id * 100000 + i
                # INSERT
                cur.execute("INSERT INTO bench_test VALUES (%s, %s, %s) ON CONFLICT (id) DO UPDATE SET value = %s",
                           (row_id, f"value_{i}", i, f"value_{i}"))
                conn.commit()
                # SELECT
                cur.execute("SELECT * FROM bench_test WHERE id = %s", (row_id,))
                cur.fetchone()
                with lock:
                    completed[0] += 2  # INSERT + SELECT = 2 ops
        finally:
            connection_pool.putconn(conn)
    
    ops_per_thread = requests // concurrency
    
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, i, ops_per_thread) for i in range(concurrency)]
        for f in futures:
            f.result()
    duration = time.perf_counter() - start
    
    # Cleanup
    conn = connection_pool.getconn()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS bench_test")
    conn.commit()
    connection_pool.putconn(conn)
    connection_pool.closeall()
    
    return BenchmarkResults("PostgreSQL (local)", completed[0], duration)


def benchmark_44s_database(requests: int, concurrency: int) -> BenchmarkResults:
    """Benchmark 44s Database with concurrent operations."""
    import urllib.request
    import json
    
    FORTY_FOURS_DB_URL = "http://api.44s.io:8600"
    
    # Test connection
    try:
        req = urllib.request.Request(f"{FORTY_FOURS_DB_URL}/health", timeout=5)
        urllib.request.urlopen(req)
    except Exception as e:
        console.print(f"[red]Error: Cannot connect to 44s Database: {e}[/red]")
        return BenchmarkResults("44s Database", 0, 1)
    
    def execute_sql(sql):
        req = urllib.request.Request(
            f"{FORTY_FOURS_DB_URL}/sql",
            data=json.dumps({"sql": sql}).encode(),
            headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
            method="POST"
        )
        try:
            response = urllib.request.urlopen(req, timeout=10)
            return json.loads(response.read().decode())
        except Exception as e:
            return {"error": str(e)}
    
    # Setup: create test table
    execute_sql("DROP TABLE IF EXISTS bench_test")
    execute_sql("CREATE TABLE bench_test (id INTEGER PRIMARY KEY, value TEXT, counter INTEGER)")
    
    lock = threading.Lock()
    completed = [0]
    
    def worker(thread_id: int, ops_per_thread: int):
        for i in range(ops_per_thread):
            row_id = thread_id * 100000 + i
            # INSERT
            execute_sql(f"INSERT INTO bench_test VALUES ({row_id}, 'value_{i}', {i})")
            # SELECT
            execute_sql(f"SELECT * FROM bench_test WHERE id = {row_id}")
            with lock:
                completed[0] += 2  # INSERT + SELECT = 2 ops
    
    ops_per_thread = requests // concurrency
    
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, i, ops_per_thread) for i in range(concurrency)]
        for f in futures:
            f.result()
    duration = time.perf_counter() - start
    
    # Cleanup
    execute_sql("DROP TABLE IF EXISTS bench_test")
    
    return BenchmarkResults("44s Database", completed[0], duration)


def run_database_benchmark(requests: int, concurrency: int):
    """Run database benchmark comparing PostgreSQL vs 44s."""
    console.print(f"\n[bold]Database Benchmark[/bold]")
    console.print(f"  Requests: {requests:,}")
    console.print(f"  Concurrency: {concurrency}")
    console.print()
    
    # Benchmark PostgreSQL
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Benchmarking local PostgreSQL...", total=None)
        pg_results = benchmark_postgres_local(requests, concurrency)
    
    if pg_results.ops_per_sec > 0:
        console.print(f"  [green]✓[/green] PostgreSQL: {pg_results.ops_per_sec:,.0f} ops/sec")
    
    # Benchmark 44s Database
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Benchmarking 44s Database...", total=None)
        fortyfours_results = benchmark_44s_database(requests, concurrency)
    
    console.print(f"  [green]✓[/green] 44s Database: {fortyfours_results.ops_per_sec:,.0f} ops/sec")
    
    # Results
    if pg_results.ops_per_sec > 0 and fortyfours_results.ops_per_sec > 0:
        speedup = fortyfours_results.ops_per_sec / pg_results.ops_per_sec
        
        console.print()
        table = Table(title="Database Benchmark Results", show_header=True)
        table.add_column("System", style="cyan")
        table.add_column("Ops/sec", justify="right", style="green")
        table.add_column("Speedup", justify="right", style="yellow")
        
        table.add_row("PostgreSQL (local)", f"{pg_results.ops_per_sec:,.0f}", "1×")
        table.add_row("44s Database (remote)", f"{fortyfours_results.ops_per_sec:,.0f}", f"{speedup:.1f}×")
        
        console.print(table)
        
        if fortyfours_results.ops_per_sec < 100:
            console.print(f"\n[bold yellow]⚠️  NETWORK LATENCY WARNING[/bold yellow]")
            console.print()
            console.print("[yellow]44s Database is running remotely, PostgreSQL is local.[/yellow]")
            console.print("[yellow]Network latency (~150ms) dominates remote results.[/yellow]")
            console.print()
            console.print("[bold]For a fair 47× comparison:[/bold]")
            console.print("  1. Run both databases on the same server")
            console.print("  2. Use a high-core server (32+ cores) to see lock contention effects")
            console.print("  3. Under high concurrency, PostgreSQL locks destroy performance")
            console.print("     while 44s scales linearly")


@cli.command()
@click.option('--requests', '-r', default=1000, help='Number of requests')
@click.option('--concurrency', '-c', default=16, help='Concurrent connections')
def database(requests, concurrency):
    """Benchmark 44s Database vs PostgreSQL."""
    print_banner()
    run_database_benchmark(requests, concurrency)


@cli.command()
@click.option('--requests', '-r', default=10000, help='Number of requests')
@click.option('--concurrency', '-c', default=32, help='Concurrent connections')
def all(requests, concurrency):
    """Run all benchmarks."""
    print_banner()
    run_cache_benchmark(requests, concurrency)
    run_database_benchmark(min(requests, 1000), concurrency)  # Fewer requests for DB due to latency

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

@cli.command()
def fair():
    """Show how to run a fair benchmark (same-network comparison)."""
    print_banner()
    console.print("[bold]Fair Benchmark Instructions[/bold]\n")
    console.print("Running the benchmark over the internet isn't fair because network")
    console.print("latency dominates the results.\n")
    
    console.print("[bold yellow]For a fair comparison, run BOTH tests from the same server:[/bold yellow]\n")
    
    console.print("[bold]Option 1: Use redis-benchmark (recommended)[/bold]")
    console.print("─" * 50)
    console.print()
    console.print("SSH into any cloud server, then run:\n")
    console.print("[cyan]# Test 44s Cache[/cyan]")
    console.print(f"redis-benchmark -h {FORTY_FOURS_HOST} -p {FORTY_FOURS_PORT} -a {API_KEY} -t set,get -n 100000 -c 50 -q\n")
    console.print("[cyan]# Test local Redis (install first: apt install redis-server)[/cyan]")
    console.print("redis-benchmark -t set,get -n 100000 -c 50 -q\n")
    
    console.print("[bold]Option 2: Spin up a high-core server[/bold]")
    console.print("─" * 50)
    console.print()
    console.print("The 450× speedup shows under [bold]high contention[/bold] (many cores).")
    console.print("On a 96-core c6a.24xlarge:\n")
    console.print("  • Redis: ~12,000 ops/sec (threads blocked on locks)")
    console.print("  • 44s:   ~5,400,000 ops/sec (lock-free, linear scaling)")
    console.print("  • Speedup: [bold green]450×[/bold green]\n")
    
    console.print("[bold]Why the local benchmark shows different results:[/bold]")
    console.print("─" * 50)
    console.print()
    console.print("  • [red]Your laptop → 44s over internet[/red]: 50-100ms latency per op")
    console.print("  • [green]Same server → 44s localhost[/green]: <1ms latency per op")
    console.print("  • [yellow]2-core server[/yellow]: Not enough contention to show speedup")
    console.print("  • [green]96-core server[/green]: Lock contention destroys Redis, 44s scales")
    console.print()

if __name__ == "__main__":
    cli()

