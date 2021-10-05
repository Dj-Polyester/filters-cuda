# filters-cuda

**NOTE:** 2d functions have implementations, however benchmarking is only available for 
1d kernels, kernels that have 1d grid and block sizes.

Config with CMake - `./config.sh`

Compile - `./compile.sh`

Execute for each file in `img_in` folder with `block_size` - `./exec.sh block_size` 

Benchmark - `./bench.sh filterfunc [howmanytimes_to_run=1 new_file=filterfunc]`

Benchmarks are cached to be plotted later. List all benchmarks - `./list.sh`

Plotting file takes input from one (or more) of the names in the output of `./list.sh`.
Plot benchmarks - `./python3 plot.py list_of_benchmarks_to_plot`. 

Plot all available benchmarks - `./list.sh | xargs ./python3 plot.py`