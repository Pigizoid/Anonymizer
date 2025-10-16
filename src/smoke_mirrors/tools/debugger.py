import sys
import time
import inspect
from functools import wraps
from collections import defaultdict


import re


class MiniLineProfiler:
    def __init__(self):
        # Store timings as: {(filename, lineno): [total_time, hits, locals_snapshots]}
        self.stats = defaultdict(lambda: [0.0, 0, []])
        self._source_map = {}  # Maps filename -> list of lines

    def _trace(self, frame, event, arg):
        code = frame.f_code
        lineno = frame.f_lineno
        filename = code.co_filename
        if (event not in ["line", "return"]) or (filename not in self._source_map):
            return self._trace
        key = (filename, lineno)

        now = time.perf_counter()

        # Attribute time to previous line
        if (
            getattr(self, "_last_time", None) is not None
            and self._last_key in self.stats
        ):
            elapsed = now - self._last_time
            self.stats[self._last_key][0] += elapsed
            self.stats[self._last_key][1] += 1

        # Record locals
        self.stats[key][2].append(dict(frame.f_locals))

        # Interactive mode: print and pause
        if getattr(self, "_interactive", False):
            if filename in self._source_map:
                # Pull line text from stored source lines
                source_lines = self._source_map.get(filename)
                if source_lines:
                    line_index = lineno - self._start_lineno_map[filename]
                    if 0 <= line_index < len(source_lines):
                        code_line = source_lines[line_index].rstrip()
                    else:
                        code_line = "<line unavailable>"
                else:
                    code_line = "<source not available>"

                total_line_time, hits, locals_snaps = self.stats.get(key, (0.0, 0, []))
                total_func_time = sum(v[0] for v in self.stats.values())
                self._print_line(
                    key,
                    code_line,
                    total_line_time,
                    hits,
                    locals_snaps,
                    total_func_time,
                    large_flag=True,
                )
                input("...")

        # Update state
        self._last_time = now
        self._last_key = key
        return self._trace

    def wrap(self, func=None, *, interactive=False):
        if func is None:
            return lambda f: self.wrap(f, interactive=interactive)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Reset state
            self.stats.clear()
            self._last_time = None
            self._last_key = None
            self._interactive = interactive

            source_lines, start_lineno = inspect.getsourcelines(func)
            filename = inspect.getsourcefile(func)
            if not hasattr(self, "_source_map"):
                self._source_map = {}
            self._source_map[filename] = source_lines
            if not hasattr(self, "_start_lineno_map"):
                self._start_lineno_map = {}
            self._start_lineno_map[filename] = start_lineno

            if interactive:
                print(
                    f"Interactive line profiler started for function '{func.__name__}'"
                )

            sys.settrace(self._trace)
            try:
                return func(*args, **kwargs)
            finally:
                sys.settrace(None)

        return wrapper

    def _print_line(
        self,
        key,
        code,
        total_time,
        hits,
        locals_snapshots,
        total_func_time,
        large_flag=False,
    ):
        time_ms = total_time * 1000
        pct_time = (total_time / total_func_time * 100) if total_func_time > 0 else 0.0

        var_names = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", code))
        locals_preview = ""
        if locals_snapshots and var_names:
            last_snapshot = locals_snapshots[-1]
            filtered_locals = {k: v for k, v in last_snapshot.items() if k in var_names}
            if filtered_locals:
                locals_preview = " | " + ", ".join(
                    f"{k}={repr(v)}" for k, v in filtered_locals.items()
                )
        if not large_flag:
            if len(locals_preview) > 60:
                locals_preview = locals_preview[:60] + "|..."

        print(
            f"{key[1]:5} {hits:5} {time_ms:10.3f} {pct_time:8.2f}  {code:40} {locals_preview}"
        )

    def report(self, func):
        filename = inspect.getsourcefile(func)
        source_lines = self._source_map[filename]
        start_lineno = self._start_lineno_map[filename]
        total_func_time = sum(v[0] for v in self.stats.values())

        print(f"\nLine-by-line profiling for {func.__name__} in {filename}:\n")
        print(
            f"{'Line':>5} {'Hits':>5} {'Time (ms)':>10} {'% Time':>8}  Code{' ' * 40} Locals"
        )
        print("-" * 140)

        for i, line in enumerate(source_lines, start=start_lineno):
            key = (filename, i)
            total_line_time, hits, locals_snaps = self.stats.get(key, (0.0, 0, []))
            self._print_line(
                key, line.rstrip(), total_line_time, hits, locals_snaps, total_func_time
            )


# ---------------- Example Usage ---------------- #

if __name__ == "__main__":
    profiler = MiniLineProfiler()

    @profiler.wrap(interactive=True)
    def my_function(n):
        total = 0
        for i in range(n):
            x = i * i
            total += x
        return total

    @profiler.wrap
    def my_function_2(n):
        total = 0
        for i in range(n):
            x = i * i
            total += x
        return total

    my_function(10)
    profiler.report(my_function)

    my_function_2(10)
    profiler.report(my_function_2)
