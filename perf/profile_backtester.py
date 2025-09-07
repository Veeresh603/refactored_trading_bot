# perf/profile_backtester.py
import cProfile, pstats, io
from perf.benchmark_backtester import main as run_main

def main():
    pr = cProfile.Profile()
    pr.enable()
    run_main()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(30)  # show top 30 by cumulative time
    print(s.getvalue())
    pr.dump_stats("profile_backtester.prof")
    print("✅ Profile saved to profile_backtester.prof")

if __name__ == "__main__":
    main()
