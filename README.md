# filters-cuda

**NOTE:** 2d functions have implementations, however benchmarking is only available for 
1d kernels, kernels that have 1d grid and block sizes.

Config with CMake - `./config.sh`

Compile - `./compile.sh`

Execute for each file in `img_in` folder with `block_size` - `./exec.sh filterfunc block_size` 

Benchmark - `./bench.sh filterfunc [howmanytimes_to_run=1 new_file=filterfunc windowWidth=3]`

Benchmarks are cached to be plotted later in `$BENCHMARKFOLDER` folder with default value of `benchmarks`. 
`plot.py` takes input as series of file paths as command line arguments. 
Plot benchmarks - `./python3 plot.py file_paths_separated_by_space`. 

Plot all available benchmarks - `source bench.sh && ls $BENCHMARKFOLDER/* | xargs ./python3 plot.py`