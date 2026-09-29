import torch
from benchmarks.stats import median, percentile
from benchmarks.utilization import gpu_utilization, cpu_utilization
from benchmarks.prompts import make_prompt


def _run_once(fn, prompt, n):
    stats = {}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    fn(prompt, n, stats=stats)

    stats["peak_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    stats["gpu_util"] = gpu_utilization()
    stats["cpu_util"] = cpu_utilization()

    return stats


def _average(runs):
    return {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}


def compare(engine, prompt, lengths=(10, 50, 100, 200), repeats=3):
    engine.generate_naive(prompt, 5)
    engine.generate(prompt, 5)

    print(f"{'mode':<8}{'req':>5}{'gen':>5}{'TTFT ms':>10}{'total s':>9}{'tok/s':>8}{'MB':>8}")
    for n in lengths:
        for name, fn in (("naive", engine.generate_naive), ("cache", engine.generate)):
            avg = _average([_run_once(fn, prompt, n) for _ in range(repeats)])
            tps = avg["new_tokens"] / avg["total"]
            print(f"{name:<8}{n:>5}{avg['new_tokens']:>5.0f}"
                  f"{avg['ttft'] * 1000:>10.1f}{avg['total']:>9.3f}{tps:>8.1f}{avg['peak_mb']:>8.0f}")


def sweep(engine, prompt_lengths=(10, 50, 200), gen_lengths=(20, 100), repeats=10):
    print(f"{'mode':<8}{'plen':>6}{'glen':>6}{'TTFT p50':>10}{'TTFT p95':>10}"
          f"{'tok/s p50':>11}{'tok/s p95':>11}{'GPU%':>7}{'CPU%':>7}{'MB':>8}")

    for plen in prompt_lengths:
        prompt = make_prompt(engine, plen)
        actual_plen = engine.tokenizer.encode(prompt).shape[1]

        for glen in gen_lengths:
            engine.generate_naive(prompt, 5)
            engine.generate(prompt, 5)

            for name, fn in (("naive", engine.generate_naive), ("cache", engine.generate)):
                runs = [_run_once(fn, prompt, glen) for _ in range(repeats)]

                ttfts = [r["ttft"] * 1000 for r in runs]
                tokps = [r["new_tokens"] / r["total"] for r in runs]

                ttft_p50 = median(ttfts)
                ttft_p95 = percentile(ttfts, 95)
                tokps_p50 = median(tokps)
                tokps_p95 = percentile(tokps, 95)

                avg_gpu = sum(r["gpu_util"] for r in runs) / len(runs)
                avg_cpu = sum(r["cpu_util"] for r in runs) / len(runs)
                avg_mb = sum(r["peak_mb"] for r in runs) / len(runs)

                print(f"{name:<8}{actual_plen:>6}{glen:>6}{ttft_p50:>10.1f}{ttft_p95:>10.1f}"
                      f"{tokps_p50:>11.1f}{tokps_p95:>11.1f}{avg_gpu:>7.1f}{avg_cpu:>7.1f}{avg_mb:>8.0f}")