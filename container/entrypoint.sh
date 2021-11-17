source setup.sh

function enable_kokkos_omp {
    PATH=/opt/kokkos-omp/bin:${PATH}
}

function enable_kokkos_cuda {
    PATH=/opt/cuda/bin:/opt/kokkos-cuda/bin:${PATH}
    LD_LIBRARY_PATH=/opt/cuda/lib64:${LD_LIBRARY_PATH}
}

WIRECELL_PATH=/opt/wire-cell-data:/products/wirecell/v0_14_0e/Linux64bit+3.10-2.17-e19-prof/wirecell-0.14.0/cfg:/products/wirecell/v0_14_0e/Linux64bit+3.10-2.17-e19-prof/share/wirecell

export -f enable_kokkos_omp enable_kokkos_cuda
/bin/bash "$@"
