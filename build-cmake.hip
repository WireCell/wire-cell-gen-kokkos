#!/bin/bash

if ! type -p cmake >& /dev/null
then
  echo "CMake not available."
  exit 1
fi

top_dir="$(cd $(dirname ${BASH_SOURCE[0]}) >&/dev/null && pwd)"
cmake_dir_kokkos=$top_dir/.cmake-kk-hip

if [ -z ${WC_GEN_KK_BUILD} ] ; then
	build_dir_kokkos=$top_dir/build/kokkos-hip
else
	build_dir_kokkos=${WC_GEN_KK_BUILD}
fi

mkdir -p  $build_dir_kokkos


build_cmds="$@"


echo
echo "=========================="
echo "Building Kokkos-hip "
echo "=========================="
echo
cd $build_dir_kokkos
[ ! -f Makefile ] && \
    (cmake $cmake_dir_kokkos -DCMAKE_CXX_EXTENSIONS=Off -DCMAKE_CXX_FLAGS="-fopenmp" -DCMAKE_CXX_COMPILER=hipcc || exit 4)
make $build_cmds || exit 5
