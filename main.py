#!/usr/bin/env python3
"""
Sorting Algorithm Runtime Visualizer
------------------------------------
Run with: python main.py
Requires: see requirements.txt
"""

import matplotlib
matplotlib.use('TkAgg')

import time
import random
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# === Sorting Algorithms ===

def bubble_sort(arr):
    n = len(arr)
        
    for i in range(n):
        swapped = False
        
        for j in range(0, n - i - 1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
                
        if not swapped:
            break
            
    return arr

def heap_sort(arr):
    def sift_down(a, start, end):
        root = start
        while(left := 2 * root + 1) <= end:
            right = left + 1
            largest = root
            if a[left] > a[largest]:
                largest = left
            if right <= end and a[right] > a[largest]:
                largest = right
            if largest == root:
                break
            a[root], a[largest] = a[largest], a[root]
            root = largest

    def build_max_heap(a):
        n = len(a)
        for i in range(n // 2 - 1, -1, -1):
            sift_down(a, i, n - 1)

    a = arr
    n = len(a)
    build_max_heap(a)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift_down(a, 0, end - 1)
    return a

def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left_half = merge_sort(arr[:mid])
    right_half = merge_sort(arr[mid:])

    def merge(left, right):
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    return merge(left_half,right_half)

def counting_sort(arr):
    if not arr:
        return[]

    max_val = max(arr)
    min_val = min(arr)
    offset = -min_val if min_val < 0 else 0
    k = max_val - min_val + 1

    count = [0] * k
    for num in arr:
        count[num - min_val] += 1

    output = []
    for i, freq in enumerate(count):
        value = i + min_val
        output.extend([value] * freq)

    return output

def insertion_sort(arr):
    n = len(arr)

    for i in range(1, n):
        key = arr[i]
        j = i - 1

        while j >= 0 and arr[j] > key: #shifts elements that are greater than key value
            arr[j+1] = arr[j]
            j-= 1

        arr[j+1] = key

    return arr

def quick_sort(arr):
    def partition(low, high):
        pivot = arr[(low+high) // 2]
        i = low
        j = high
        while i <= j:
            while arr[i] < pivot:
                i += 1
            while arr[j] > pivot:
                j -= 1
            if i <= j:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
                j -= 1
        return i, j

    def sort(low, high):
        if low < high:
            i, j = partition(low, high)
            sort(low, j)
            sort(i, high)

    sort(0, len(arr) - 1)
    return arr

def quick_select_sort(arr: list[int], low: int = 0, high: int | None = None, k: int | None = None) -> int:

    # Use random k for sake of benchmark
    if high is None:
        high = len(arr) - 1
    if k is None:
        k = random.randint(0, len(arr) - 1)

    def partition(lo: int, hi: int) -> int:
        pivot = arr[hi]
        i = lo
        for j in range(lo, hi):
            if arr[j] <= pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[hi] = arr[hi], arr[i]
        return i

    pi = partition(low, high)
    if pi == k:
        return arr[pi]
    elif pi > k:
        return quick_select_sort(arr, low, pi - 1, k)
    else:
        return quick_select_sort(arr, pi + 1, high, k)


def radix_sort(arr, base=10):
    if not arr:
        return arr

    def _count_sort_by_digit(a, exp, base=10):
        n = len(a)
        output = [0] * n
        count = [0] * base
        # Counter for each digit
        for i in range(n):
            d = (a[i] // exp) % base
            count[d] += 1
        # Get positions
        for d in range(1, base):
            count[d] += count[d - 1]
        # Stable placement
        for i in range(n - 1, -1, -1):
            d = (a[i] // exp) % base
            output[count[d] - 1] = a[i]
            count[d] -= 1
        # Replace
        for i in range(n):
            a[i] = output[i]

    def _radix_sort_lsd_nonneg(a, base=10):
        if not a:
            return a
        max_val = max(a)
        exp = 1
        while max_val // exp > 0:
            _count_sort_by_digit(a, exp, base)
            exp *= base
        return a

    neg = [-x for x in arr if x < 0]
    pos = [x for x in arr if x >= 0]

    if neg:
        _radix_sort_lsd_nonneg(neg, base)
    if pos:
        _radix_sort_lsd_nonneg(pos, base)

    out = [-x for x in reversed(neg)] + pos

    for i, v in enumerate(out):
        arr[i] = v

    return arr

def bucket_sort(arr: list[float]) -> list[float]:
    n = len(arr)
    if n <= 1:
        return arr[:]
    
    mn = min(arr)
    mx = max(arr)
    rng = mx - mn
    if range == 0:
        return arr[:]

    buckets: list[list[float]] =[[] for _ in range(n)]

    for x in arr:
        idx = int(n * ((x-mn) / rng))
        if idx == n:
            idx = n - 1
        buckets[idx].append(x)

    result: list[float] = []
    for b in buckets:
        result.extend(b)

    return result

# === Benchmark Function ===

# Multiple runs: Average over several runs to smooth out small GC quirks.
# Warm-up run: Run each algorithm once before measuring so caches, interpreter overhead, and imports don’t skew results.

def measure_runtime(arr, algo):
    start = time.perf_counter()
    algo(arr[:])
    end = time.perf_counter()
    return (end - start) * 1000000  # seconds -> microseconds

# === Matplotlib Chart & Tkinter UI ===

class SortAlgoAnalyzer:
    def __init__(self, master):
        self.master = master
        self.master.title('SortAlgoAnalyzer')

        # Frame for Matplotlib graph
        self.plot_frame = tk.Frame(master)
        self.plot_frame.pack(fill='both', expand=True)

        # Textbox
        self.entry = tk.Text(master, width=90, height=4)
        self.entry.pack(side=tk.BOTTOM, padx=10, pady=20)
        self.entry.bind('<Return>',self.on_return_start)

        # Input Prompt
        self.input_label = tk.Label(master, text='Enter integers seaparated by spaces:')
        self.input_label.pack(side=tk.BOTTOM)

        # Frame for buttons
        button_frame = tk.Frame(self.master)
        button_frame.pack(side=tk.BOTTOM, pady=20)

        # Start / Stop toggle
        self.toggle_btn = tk.Button(
            master,
            text='Start / Stop',
            width=12,
            height=1,
            bg='beige',
            fg='darkblue',
            command=self.start_or_toggle
        )
        self.toggle_btn.pack(in_=button_frame, side=tk.LEFT, padx=5)

        # Restart Button
        self.restart_btn = tk.Button(
            master,
            text='Restart',
            width=12,
            height=1,
            bg='beige',
            fg='darkblue',
            command=self.restart_program
        )
        self.restart_btn.pack(in_=button_frame, side=tk.RIGHT, padx=5)

        # Initialize Matplotlib variables
        self.canvas = None
        self.fig = None
        self.ax = None
        self.ani = None
        self.is_paused = False

    def parse_input(self):
        # Take string of characters from text widget and removes spaces
        text = self.entry.get('1.0', 'end').strip()

        # Input error handling
        if not text:
            raise ValueError('Please enter at least one integer.')
        text = ' '.join(text.split())
        try:
            nums = list(map(int, text.split()))
        except ValueError:
            raise ValueError('Please enter only integers separated by spaces.')
        return nums
    
    def on_enter(self, event=None):
        # On <Return> or click of start toggle accept user input from entry box
        try:
            numbers = self.parse_input()
        except ValueError as e:
            messagebox.showerror('Invalid Input', str(e))
            return
        
        algorithms = {
            'Bubble':       bubble_sort,
            'Heap':         heap_sort,
            'Merge':        merge_sort,
            'Counting':     counting_sort,
            'Insertion':    insertion_sort,
            'Quick':        quick_sort,
            'Quick-Select': quick_select_sort,
            'Radix':        radix_sort,
            'Bucket':       bucket_sort
        }

        # Creates lists for algorithm title and corresponding runtime
        titles = list(algorithms.keys())
        runtimes = [measure_runtime(numbers,algo) for algo in algorithms.values()]

        # Initiates graph creation
        self.make_plot(titles,runtimes)

    def on_return_start(self, event=None):
        self.start_or_toggle()
        return 'break'

    def start_or_toggle(self):
        # Start program if initial start
        if self.ani is None or getattr(self.ani, 'event_source', None) is None:
            try:
                self.on_enter()
            except Exception as e:
                messagebox.showerror('Error while running', str(e))
                return
        # Toggle if program already started
        else:
            self.toggle_animation()

    def toggle_animation(self):
        try:
            # Do nothing if nothing to animate
            if self.ani is None or getattr(self.ani, 'event_source', None) is None:
                return
            # Start clock if paused
            if self.is_paused:
                self.ani.event_source.start()
            # Stop clock if running
            else:
                self.ani.event_source.stop()
            self.is_paused = not self.is_paused
        except Exception:
            pass

    def restart_program(self):
        # Clear textbox
        try:
            self.entry.delete('1.0', 'end')
        except Exception:
            pass

        # Clear plot
        if self.canvas is not None:
            try:
                self.canvas.get_tk_widget().destroy()
            except Exception:
                pass
            self.canvas = None
            self.fig = None
            self.ax = None
            self.ani = None

        # Reset clock flag
        self.is_paused = False

    def make_plot(self, titles, runtimes):
        # Ensure prior run was cleaned up
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

        # Bar plot layout and labels
        self.fig, self.ax = plt.subplots(figsize=(8, 4.8), dpi=100)
        self.ax.set_xlabel('Algorithm')
        self.ax.set_ylabel('Runtime (Microseconds)')
        self.ax.set_title('Time Complexity of Sorting Algorithms')

        # Prevent bars from passing top of graph and set colors
        colors = ['navy', 'cornflowerblue', 'lightskyblue', 'mediumseagreen', 'springgreen', 'gold', 'orange', 'coral', 'gainsboro']
        self.bars = self.ax.bar(titles, [0] * len(titles), color=colors)
        self.ax.set_ylim(0, max(runtimes) * 1.15)

        # Rotate algorithm names
        self.ax.tick_params(axis='x', pad=10)  
        plt.setp(self.ax.get_xticklabels(), rotation=30, ha='right')
        self.fig.subplots_adjust(bottom=0.25)

        # Embed bar plot into designated canvas / figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0,20))

        # Create bar growth animation
        frames_grow = list(range(0,101))
        frames_stop = [100] * 100
        frames = frames_grow + frames_stop

        def update(frame_pct):
            for bar, rt in zip(self.bars, runtimes):
                bar.set_height(rt * (frame_pct / 100.0))
            return self.bars
        
        self.ani = animation.FuncAnimation(self.fig, update, frames=frames, interval=5, blit=True, repeat=False)
        self.is_paused = False
        self.canvas.draw_idle()

def main():
    root = tk.Tk()
    app = SortAlgoAnalyzer(root)
    root.mainloop()

if __name__ == '__main__':
    main()

