from time import perf_counter
from psutil import Process
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy
import os
import traceback


def _user_traceback(e):
    """
    Full traceback of an exception raised by my_solution, starting at the user's own code:
    the frames of this file (the evaluation code that called my_solution) are dropped.
    """
    here = os.path.abspath(__file__)
    tb = e.__traceback__
    while tb is not None and os.path.abspath(tb.tb_frame.f_code.co_filename) == here:
        tb = tb.tb_next
    return "".join(traceback.format_exception(type(e), e, tb))


def get_trss(proc):
    """Get current time and RSS memory usage."""
    return perf_counter(), proc.memory_info().rss


def _confirm_time(my_solution, test_input, exec_time, time_limit):
    """
    A call over the time limit is timed once more (on a copy of its input), so that a garbage
    collection pause or an OS hiccup cannot fail a fast solution. Returns the best of both timings.
    """
    if exec_time <= time_limit:
        return exec_time
    t_start = perf_counter()
    try:
        my_solution(*deepcopy(test_input))
    except Exception:
        return exec_time
    return min(exec_time, perf_counter() - t_start)


def _limit_violation(exec_time, mem_used, time_limit, memory_limit):
    """Message describing an exceeded limit, or None."""
    if exec_time > time_limit:
        return f"Time limit exceeded: {exec_time:.2f} s > {time_limit:g} s"
    if mem_used > memory_limit:
        return f"Memory limit exceeded: {mem_used / 1e6:.0f} MB > {memory_limit / 1e6:.0f} MB"
    return None


def evaluate_on_samples(samples, my_solution, check_solution, time_limit, memory_limit,
                        format_input=None, format_output=None):
    """
    Evaluate a solution on provided sample test cases with detailed output table.
    
    Parameters:
    -----------
    samples : list of tuples
        Each tuple contains the inputs for my_solution (e.g., (n, s) for challenge 1)
    my_solution : callable
        Function that takes unpacked sample inputs and returns the solution
    check_solution : callable
        Function with signature check_solution(sample, result, proc, tmax, rmax)
        Returns (is_accurate, is_time_efficient, is_memory_efficient, custom_message)
    time_limit : float
        Maximum time in seconds for one call of my_solution (enforced)
    memory_limit : float
        Maximum extra RSS memory in bytes for one call of my_solution (enforced)
    format_input : callable, optional
        Function to format input for display: format_input(sample) -> str
        If None, uses default repr with truncation
    format_output : callable, optional
        Function to format output for display: format_output(result) -> str
        If None, uses default repr with truncation
    
    Returns:
    --------
    None : the results are printed. If my_solution raised an exception, the full traceback of the
    first one (from the user's code onwards) is printed after the table. Returning None keeps a
    notebook cell that ends with this call from also displaying a value.
    """
    proc = Process(os.getpid())
    all_passed = True
    first_crash = None  # (sample number, traceback) of the first exception raised by my_solution

    # Collect results for table display
    results_table = []

    for i, sample in enumerate(samples, 1):
        # Measure execution
        t_start = perf_counter()
        mem_before = proc.memory_info().rss
        
        try:
            user_result = my_solution(*sample)
            runtime_error = None
        except Exception as e:
            user_result = None
            runtime_error = f"{type(e).__name__}: {e}"
            if first_crash is None:
                first_crash = (i, _user_traceback(e))

        t_end = perf_counter()
        mem_after = proc.memory_info().rss
        
        exec_time = t_end - t_start
        mem_used = max(0, mem_after - mem_before)
        if not runtime_error:
            exec_time = _confirm_time(my_solution, sample, exec_time, time_limit)
        violation = None if runtime_error else _limit_violation(exec_time, mem_used, time_limit, memory_limit)

        # Check solution
        if runtime_error:
            is_accurate = False
            custom_message = f"Runtime error: {runtime_error}"
        else:
            check_result = check_solution(
                sample, user_result, proc, time_limit, memory_limit
            )
            is_accurate, is_time_efficient, is_memory_efficient = check_result[:3]
            custom_message = check_result[3] if len(check_result) > 3 else None
        
        # Format input/output for display
        if format_input:
            input_str = format_input(sample)
        else:
            input_str = _truncate_repr(sample, 40)
        
        if format_output:
            output_str = format_output(user_result)
        else:
            output_str = _truncate_repr(user_result, 30)
        
        # Determine status
        if runtime_error:
            status = "❌ Error"
            status_detail = runtime_error[:30] + "..." if len(runtime_error) > 30 else runtime_error
        elif violation:
            status = "❌ TLE" if violation.startswith("Time") else "❌ MLE"
            status_detail = violation
        elif not is_accurate:
            status = "❌ Wrong"
            status_detail = custom_message if custom_message else "Invalid answer"
        else:
            status = "✅ Pass"
            status_detail = ""
        
        results_table.append({
            'index': i,
            'input': input_str,
            'output': output_str,
            'time_us': exec_time * 1e6,
            'status': status,
            'status_detail': status_detail
        })
        
        if status != "✅ Pass":
            all_passed = False

    # Display results table
    _print_results_table(results_table)
    
    # Summary
    passed_count = sum(1 for r in results_table if r['status'] == "✅ Pass")
    print(f"\n{'='*60}")
    if all_passed:
        print(f"✅  All {len(samples)} sample tests passed!")
    else:
        print(f"❌  {passed_count}/{len(samples)} sample tests passed")
        # Show first failure detail
        for r in results_table:
            if r['status'] != "✅ Pass" and r['status_detail']:
                print(f"    First failure: {r['status_detail']}")
                break
        if first_crash:
            index, trace = first_crash
            print(f"\nRuntime error on sample {index}:\n")
            print(trace, end="")


def _truncate_repr(obj, max_len):
    """Truncate string representation of an object."""
    s = repr(obj)
    if len(s) > max_len:
        return s[:max_len-3] + "..."
    return s


def _print_results_table(results):
    """Print a formatted results table."""
    # Calculate column widths
    col_widths = {
        'index': 4,
        'input': max(5, max(len(r['input']) for r in results)),
        'output': max(6, max(len(r['output']) for r in results)),
        'time': 12,
        'status': 10
    }
    
    # Cap widths to prevent overflow
    col_widths['input'] = min(col_widths['input'], 45)
    col_widths['output'] = min(col_widths['output'], 35)
    
    # Header
    header = (f"{'#':>{col_widths['index']}} │ "
              f"{'Input':<{col_widths['input']}} │ "
              f"{'Output':<{col_widths['output']}} │ "
              f"{'Time':>{col_widths['time']}} │ "
              f"{'Status':<{col_widths['status']}}")
    
    separator = "─" * (col_widths['index'] + 1) + "┼" + \
                "─" * (col_widths['input'] + 2) + "┼" + \
                "─" * (col_widths['output'] + 2) + "┼" + \
                "─" * (col_widths['time'] + 2) + "┼" + \
                "─" * (col_widths['status'] + 1)
    
    print(f"\n{'Sample Test Results':^{len(header)}}")
    print("=" * len(header))
    print(header)
    print(separator)
    
    # Rows
    for r in results:
        # Format time
        if r['time_us'] < 1000:
            time_str = f"{r['time_us']:.1f} μs"
        else:
            time_str = f"{r['time_us']/1000:.2f} ms"
        
        # Truncate if needed
        input_str = r['input'][:col_widths['input']]
        output_str = r['output'][:col_widths['output']]
        
        row = (f"{r['index']:>{col_widths['index']}} │ "
               f"{input_str:<{col_widths['input']}} │ "
               f"{output_str:<{col_widths['output']}} │ "
               f"{time_str:>{col_widths['time']}} │ "
               f"{r['status']:<{col_widths['status']}}")
        print(row)


def _pairs(m, limit=500_000):
    """Index pairs (i, j), i < j, for the Theil-Sen slopes; a fixed random sample when there are too many."""
    if m * (m - 1) // 2 <= limit:
        return np.triu_indices(m, 1)
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, m, limit), rng.integers(0, m, limit)
    keep = i != j
    return i[keep], j[keep]


def _fit_complexity(n_values, measurements, tolerance=0.05):
    """
    Estimate the complexity class of measurements taken at sizes n_values.

    measurements[i] is one value, or the list of every value measured at size n_values[i].
    Each class f is fitted as y = a * f(n) + b (growth a > 0, overhead b >= 0) with the Theil-Sen
    estimator (median of the pairwise slopes): unlike least squares, it is not dragged by single
    slow calls (garbage collection, OS scheduling). Fits are compared by their total absolute error,
    and the slowest-growing class within `tolerance` of the best error wins: over a limited range
    of n, n log n is almost a straight line, and noise alone must not turn O(n) into O(n log n).

    Returns:
    --------
    tuple : (best_name, best_func, best_params, all_results)
    """
    n = np.concatenate([np.full(np.size(m), float(k)) for k, m in zip(n_values, measurements)])
    y = np.concatenate([np.ravel(np.asarray(m, dtype=float)) for m in measurements])

    # Handle edge case: all measurements are zero or constant
    if len(y) < 2 or np.std(y) < 1e-10:
        return ("O(1)", lambda n: np.ones_like(n), (float(np.median(y)), 0), [])

    # Avoid division by zero and log of zero
    n_safe = np.maximum(n, 1)
    i, j = _pairs(len(y))

    # Define complexity functions: (name, transform_function), from the slowest growth to the fastest
    complexities = [
        ("O(1)", lambda n: np.ones_like(n)),
        ("O(log n)", lambda n: np.log2(np.maximum(n, 1))),
        ("O(n)", lambda n: n),
        ("O(n log n)", lambda n: n * np.log2(np.maximum(n, 1))),
        ("O(n²)", lambda n: n ** 2),
        ("O(n³)", lambda n: n ** 3),
        ("O(2ⁿ)", lambda n: 2 ** np.minimum(n, 30)),  # Cap to avoid overflow
    ]
    
    results = []  # (name, func, (a, b), total absolute error)

    for name, func in complexities:
        # For O(1), we fit y = constant
        if name == "O(1)":
            c = float(np.median(y))
            results.append((name, func, (c, 0), float(np.sum(np.abs(y - c)))))
            continue

        # Theil-Sen slope over the pairs of measurements taken at different sizes
        x = func(n_safe)
        dx = x[j] - x[i]
        keep = dx != 0
        if not keep.any():
            continue
        a = float(np.median((y[j] - y[i])[keep] / dx[keep]))

        # Only consider positive scaling factors for non-constant models
        if a <= 0:
            continue
        b = max(0.0, float(np.median(y - a * x)))
        results.append((name, func, (a, b), float(np.sum(np.abs(y - a * x - b)))))

    # The slowest-growing class that fits about as well as the best one
    best_error = min(r[3] for r in results)
    for r in results:
        if r[3] <= best_error * (1 + tolerance):
            return (r[0], r[1], r[2], results)


def _plot_complexity_analysis(time_by_n, memory_by_n, title_prefix="", show_estimation=True):
    """
    Create a beautiful plot showing time and memory complexity analysis.
    
    Parameters:
    -----------
    time_by_n : dict
        Dictionary mapping n -> list of execution times
    memory_by_n : dict
        Dictionary mapping n -> list of memory usages
    title_prefix : str
        Optional prefix for the plot title
    show_estimation : bool
        Whether to show complexity estimation annotations (default: True)
    """
    # Set up the style - use a style that's likely to be available
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            pass  # Use default style
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')
    
    # Color scheme
    color_primary = '#2E86AB'      # Blue
    color_secondary = '#A23B72'    # Pink/Magenta
    color_fit = '#F18F01'          # Orange
    color_fit2 = '#C73E1D'         # Red
    
    # Prepare data
    n_values = sorted(time_by_n.keys())
    
    # Calculate averages and std
    avg_times = [np.mean(time_by_n[n]) * 1e6 for n in n_values]  # Convert to microseconds
    std_times = [np.std(time_by_n[n]) * 1e6 for n in n_values]
    avg_memory = [np.mean(memory_by_n[n]) / 1024 for n in n_values]  # Convert to KB
    std_memory = [np.std(memory_by_n[n]) / 1024 for n in n_values]
    
    # Fit complexity curves on every measurement (same units as the plot)
    time_fit = _fit_complexity(n_values, [np.array(time_by_n[n]) * 1e6 for n in n_values])
    memory_fit = _fit_complexity(n_values, [np.array(memory_by_n[n]) / 1024 for n in n_values])
    
    # Generate smooth curve for plotting
    n_smooth = np.linspace(min(n_values), max(n_values), 100)
    
    # ==================== TIME PLOT ====================
    ax1.errorbar(n_values, avg_times, yerr=std_times, fmt='o', 
                 color=color_primary, markersize=8, capsize=5, capthick=2,
                 label='Measured time', ecolor=color_primary, alpha=0.7)
    
    # Plot fitted curve (only if show_estimation is True)
    if show_estimation:
        a, b = time_fit[2]
        if time_fit[0] == "O(1)":
            # Horizontal line for O(1)
            ax1.axhline(y=a, color=color_fit, linestyle='--', linewidth=2.5,
                        label=f'Fitted: {time_fit[0]}')
        elif a > 0:
            y_fit = a * time_fit[1](n_smooth) + b
            ax1.plot(n_smooth, y_fit, '--', color=color_fit, linewidth=2.5,
                     label=f'Fitted: {time_fit[0]}')
    
    ax1.set_xlabel('Input Size (n)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (μs)', fontsize=12, fontweight='bold')
    ax1.set_title('Time Complexity Analysis', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_ylim(bottom=0)  # Start from 0
    
    # Add complexity annotation (only if show_estimation is True)
    if show_estimation:
        ax1.annotate(f'Estimated: {time_fit[0]}', 
                     xy=(0.95, 0.05), xycoords='axes fraction',
                     fontsize=12, fontweight='bold', color=color_fit,
                     ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                              edgecolor=color_fit, alpha=0.9))
    
    # ==================== MEMORY PLOT ====================
    # Check if memory data is all zeros
    all_zero_memory = all(m == 0 for m in avg_memory)
    
    if all_zero_memory:
        if show_estimation:
            msg = 'Memory changes too small\nto measure accurately\n\nEstimated: O(1)'
        else:
            msg = 'Memory changes too small\nto measure accurately'
        ax2.text(0.5, 0.5, msg, 
                 transform=ax2.transAxes, fontsize=14, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                          edgecolor=color_secondary, alpha=0.9),
                 color=color_secondary)
        ax2.set_xlim(min(n_values) - 0.5, max(n_values) + 0.5)
        memory_fit = ("O(1)", lambda n: np.ones_like(n), (0, 0), [])
    else:
        ax2.errorbar(n_values, avg_memory, yerr=std_memory, fmt='s',
                     color=color_secondary, markersize=8, capsize=5, capthick=2,
                     label='Measured memory', ecolor=color_secondary, alpha=0.7)
        
        # Plot fitted curve (only if show_estimation is True)
        if show_estimation:
            a, b = memory_fit[2]
            if memory_fit[0] == "O(1)":
                ax2.axhline(y=a, color=color_fit2, linestyle='--', linewidth=2.5,
                            label=f'Fitted: {memory_fit[0]}')
            elif a > 0:
                y_fit = a * memory_fit[1](n_smooth) + b
                ax2.plot(n_smooth, y_fit, '--', color=color_fit2, linewidth=2.5,
                         label=f'Fitted: {memory_fit[0]}')
        
        ax2.legend(loc='upper left', fontsize=10)
        ax2.set_ylim(bottom=0)
        
        # Add complexity annotation (only if show_estimation is True)
        if show_estimation:
            ax2.annotate(f'Estimated: {memory_fit[0]}', 
                         xy=(0.95, 0.05), xycoords='axes fraction',
                         fontsize=12, fontweight='bold', color=color_fit2,
                         ha='right', va='bottom',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                  edgecolor=color_fit2, alpha=0.9))
    
    ax2.set_xlabel('Input Size (n)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Memory Usage (KB)', fontsize=12, fontweight='bold')
    ax2.set_title('Space Complexity Analysis', fontsize=14, fontweight='bold', pad=10)
    
    # Overall title
    main_title = "Complexity Analysis"
    if title_prefix:
        main_title = f"{title_prefix} - {main_title}"
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.show()
    
    return time_fit[0], memory_fit[0]


def internal_evaluation(test_file_path, my_solution, check_solution, parse_tests,
                        time_limit, memory_limit, get_input_size=None, plot=True,
                        plot_title="", show_estimation=True):
    """
    Evaluate a solution on hidden test cases from a file.
    
    Parameters:
    -----------
    test_file_path : str
        Path to the test file
    my_solution : callable
        Function that takes test inputs and returns the solution
    check_solution : callable
        Function with signature check_solution(test_input, result, proc, tmax, rmax)
        Returns (is_accurate, is_time_efficient, is_memory_efficient, custom_message)
    parse_tests : callable
        Function that parses the test file and returns a list of test inputs
        Signature: parse_tests(file_path) -> list of tuples
    time_limit : float
        Maximum time in seconds for one call of my_solution (enforced, like a Codeforces
        test file holding this single test case)
    memory_limit : float
        Maximum extra RSS memory in bytes for one call of my_solution (enforced)
    get_input_size : callable, optional
        Function that extracts the size parameter (n) from test_input for plotting
        Signature: get_input_size(test_input) -> int
        If None, plotting is disabled
    plot : bool
        Whether to plot complexity analysis (default: True)
        Requires get_input_size to be provided
    plot_title : str
        Optional title prefix for the complexity plot
    show_estimation : bool
        Whether to show complexity estimation on the plot (default: True)
        Set to False when n range is too small for reliable estimation
    
    Returns:
    --------
    bool : True if all tests passed
    """
    if not os.path.exists(test_file_path):
        print(f"❌  Test file not found: {test_file_path}")
        return False
    
    # Parse test cases using the challenge-specific parser
    tests = parse_tests(test_file_path)
    
    if not tests:
        print("❌  No test cases found in file")
        return False
    
    proc = Process(os.getpid())
    all_passed = True
    
    # For complexity analysis
    time_by_n = defaultdict(list)
    memory_by_n = defaultdict(list)
    
    for test_num, test_input in tqdm(enumerate(tests, 1), total=len(tests)):
        # Measure execution time
        t_start = perf_counter()
        mem_before = proc.memory_info().rss
        
        try:
            user_result = my_solution(*test_input)
        except Exception as e:
            print(f"\n❌  Runtime error at test {test_num}:\n")
            print(_user_traceback(e), end="")
            return False
        
        t_end = perf_counter()
        mem_after = proc.memory_info().rss
        exec_time = _confirm_time(my_solution, test_input, t_end - t_start, time_limit)
        mem_used = max(0, mem_after - mem_before)

        # Record metrics for complexity analysis
        if get_input_size is not None:
            n = get_input_size(test_input)
            time_by_n[n].append(exec_time)
            memory_by_n[n].append(mem_used)

        violation = _limit_violation(exec_time, mem_used, time_limit, memory_limit)
        if violation:
            size = f" (n = {get_input_size(test_input)})" if get_input_size is not None else ""
            print(f"\n❌  {violation} at test {test_num}{size}")
            all_passed = False
            break

        check_result = check_solution(
            test_input, user_result, proc, time_limit, memory_limit
        )
        
        # Unpack result (supports optional 4th element for custom error message)
        is_accurate, is_time_efficient, is_memory_efficient = check_result[:3]
        custom_message = check_result[3] if len(check_result) > 3 else None
        
        if not is_accurate:
            if custom_message:
                print(f"\n❌  {custom_message}")
            else:
                print(f"\n❌  Wrong answer at test {test_num}")
            all_passed = False
            break
        elif not is_time_efficient:
            print(f"\n❌  Maximum time exceeded at test {test_num}")
            all_passed = False
            break
        elif not is_memory_efficient:
            print(f"\n❌  Maximum memory exceeded at test {test_num}")
            all_passed = False
            break
    
    if all_passed:
        print(f"✅  All {len(tests)} tests passed!")
        
        # Plot complexity analysis if enabled and we have data
        if plot and get_input_size is not None and time_by_n:
            time_complexity, space_complexity = _plot_complexity_analysis(
                time_by_n, memory_by_n, title_prefix=plot_title,
                show_estimation=show_estimation
            )
            if show_estimation:
                print(f"\n📊 Estimated Time Complexity: {time_complexity}")
                print(f"📊 Estimated Space Complexity: {space_complexity}")
    
    return all_passed