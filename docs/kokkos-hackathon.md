# Kokkos Hackathon

## prerequisites
 - Need to have access to a `Wire-Cell Toolkit` build and its dependencies.
 - Need to have access to a `Kokkos` build.

## build on cori with shifter

[Shifter usage for Cori](https://github.com/hep-cce2/PPSwork/blob/master/Wire-Cell/Shifter.md)

Within shifter container:

building script using cmake:
```bash
./build-cmake
```

## unit test

run the kokkos unit test manually:

In the source folder (one level above build)
```bash
./build/kokkos/test/test_fft 
```

Example output:
```bash
Space: N6Kokkos6OpenMPE
WIRECELL_KOKKOSARRAY_FFTW
A: init 3.049067 ms.
A dims: 2 3
WIRECELL_KOKKOSARRAY_FFTW
irep: 0 0.524948 ms.
A: nrep: 1 avg: 0.52 ms.
WIRECELL_KOKKOSARRAY_FFTW
B: init 11.097288 ms.
B dims: 4 3
WIRECELL_KOKKOSARRAY_FFTW
irep: 0 2.083998 ms.
B: nrep: 1 avg: 2.08 ms.
```

## full test running `wire-cell` as plugin of `LArSoft`


### download `wire-cell-data`

`wire-cell-data` contains needed data files, e.g. geometry files, running full tests.

```
git clone https://github.com/WireCell/wire-cell-data.git
```

### $WIRECELL_PATH

`wire-cell` searches pathes in this env var for configuration and data files.

for bash, run something like this below:

```
export WIRECELL_PATH=$WIRECELL_FQ_DIR/wirecell-0.14.0/cfg # main cfg
export WIRECELL_PATH=$WIRECELL_DATA_PATH:$WIRECELL_PATH # data
export WIRECELL_PATH=$WIRECELL_GEN_KOKKOS_INSTALL_PATH/share/wirecell/cfg:$WIRECELL_PATH # gen-kokkos

```
Variable meaning:
 - `$WIRECELL_FQ_DIR` is a variable defined developing in Kyle's container or `setup wirecell` in a Fermilab ups system, current version is `0.14.0`, may upgrade in the future.
 - `WIRECELL_DATA_PATH` refer to the git repository cloned from the previous step
 - `WIRECELL_GEN_KOKKOS_INSTALL_PATH` refer to the install path of the `wire-cell-gen-kokkos` standalone package.

### $LD_LIBRARY_PATH

```
export LD_LIBRARY_PATH=$WIRECELL_GEN_KOKKOS_INSTALL_PATH/lib64/:$LD_LIBRARY_PATH
```


### run

 - input: a root file (refered to as [`g4.root`](https://github.com/hep-cce2/PPSwork/blob/master/Wire-Cell/examples/g4.root) below) containing Geant4 energy depo (`sim::SimEnergyDeposits`)
 - in the example folder: `lar -n 1 -c sim.fcl g4.root`
