/**
 * Similar like the WireCell::Array with Eigen backend,
 * this KokkosArray provides interface for FFTs.
 */

#ifndef WIRECELL_KOKKOSARRAY_DUMMY
#define WIRECELL_KOKKOSARRAY_DUMMY

namespace WireCell {

    namespace KokkosArray {
        /**
         * Below are 4 functions for for the FFT operations.
         * The actual operation is just a placeholder, adding "1" for debugging purpose.
         */
        inline array_xxc dft_rc(const array_xxf& arr, int dim = 0)
        {
            Index N0 = arr.extent(0);
            Index N1 = arr.extent(1);
            auto ret = gen_2d_view<array_xxc>(N0, N1, 0);
            // TODO place holder for now, implement a real one!
            Kokkos::parallel_for("dft_rc",
                                 Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {N0, N1}),
                                 KOKKOS_LAMBDA(const Index& i0, const Index& i1) { ret(i0, i1) = arr(i0, i1)+1; });
            return ret;
        }
        inline array_xxc dft_cc(const array_xxc& arr, int dim = 0)
        {
            Index N0 = arr.extent(0);
            Index N1 = arr.extent(1);
            auto ret = gen_2d_view<array_xxc>(N0, N1, 0);
            // TODO place holder for now, implement a real one!
            Kokkos::parallel_for("dft_cc",
                                 Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {N0, N1}),
                                 KOKKOS_LAMBDA(const Index& i0, const Index& i1) { ret(i0, i1) = arr(i0, i1)+1; });
            return ret;
        }
        inline array_xxc idft_cc(const array_xxc& arr, int dim = 0)
        {
            Index N0 = arr.extent(0);
            Index N1 = arr.extent(1);
            auto ret = gen_2d_view<array_xxc>(N0, N1, 0);
            // TODO place holder for now, implement a real one!
            Kokkos::parallel_for("idft_cc",
                                 Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {N0, N1}),
                                 KOKKOS_LAMBDA(const Index& i0, const Index& i1) { ret(i0, i1) = arr(i0, i1)-1; });
            return ret;
        }
        inline array_xxf idft_cr(const array_xxc& arr, int dim = 0)
        {
            Index N0 = arr.extent(0);
            Index N1 = arr.extent(1);
            auto ret = gen_2d_view<array_xxf>(N0, N1, 0);
            // TODO place holder for now, implement a real one!
            Kokkos::parallel_for("idft_cr",
                                 Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {N0, N1}),
                                 KOKKOS_LAMBDA(const Index& i0, const Index& i1) { ret(i0, i1) = Kokkos::real(arr(i0, i1))-1; });
            return ret;
        }

    }  // namespace KokkosArray
}  // namespace WireCell

#endif
