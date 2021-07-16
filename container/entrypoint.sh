source setup.sh

function enable_kokkos_omp {
    PATH=/opt/kokkos-omp/bin:${PATH}
}

function enable_kokkos_cuda {
    PATH=/opt/cuda/bin:/opt/kokkos-cuda/bin:${PATH}
    LD_LIBRARY_PATH=/opt/cuda/lib64:${LD_LIBRARY_PATH}
}

WIRECELL_PATH=/opt/wire-cell-data:$WIRECELL_PATH

export -f enable_kokkos_omp enable_kokkos_cuda
/bin/bash "$@"
