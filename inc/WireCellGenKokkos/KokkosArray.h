/**
 * Similar like the WireCell::Array with Eigen backend,
 * this KokkosArray provides interface for FFTs.
 */

#ifndef WIRECELL_KOKKOSARRAY
#define WIRECELL_KOKKOSARRAY

#include "WireCellUtil/Waveform.h"

#include <Kokkos_Core.hpp>

#include <string>

namespace WireCell {

    namespace KokkosArray {
        using Scalar = float;
        using Index = int;
        using Layout = Kokkos::LayoutLeft;
        using Space = Kokkos::CudaSpace;

        /// A real, 2D array
        typedef Kokkos::View<Scalar**, Layout, Space> array_xxf;

        /// A complex, 2D array
        typedef Kokkos::View<Kokkos::complex<Scalar>**, Layout, Space> array_xxc;

        /// Generate a 2D view initialized with given value.
        template <class ViewType>
        inline ViewType gen_2d_view(const Index N0, const Index N1, const Scalar val = 0)
        {
            ViewType ret("ret", N0, N1);
            Kokkos::parallel_for("gen_2d_view",
                                 Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {N0, N1}),
                                 KOKKOS_LAMBDA(const Index& i0, const Index& i1) { ret(i0, i1) = val; });
            return ret;
        }

        /// Dump out a string for pinting for a 2D view.
        template <class ViewType>
        inline std::string dump_2d_view(const ViewType& A, const Index length_limit = 20)
        {
            std::stringstream ss;
            ss << "dump_2d_view: \n";

            auto h_A = Kokkos::create_mirror_view(A);
            Kokkos::deep_copy(h_A, A);

            Index N0 = A.extent(0);
            Index N1 = A.extent(1);
            bool print_dot0 = true;
            for (Index i = 0; i < N0; ++i) {
                if (i > length_limit && i < N0 - length_limit) {
                    if (print_dot0) {
                        ss << "... \n";
                        print_dot0 = false;
                    }
                    continue;
                }

                bool print_dot1 = true;
                for (Index j = 0; j < N1; ++j) {
                    if (j > length_limit && j < N1 - length_limit) {
                        if (print_dot1) {
                            ss << "... ";
                            print_dot1 = false;
                        }

                        continue;
                    }
                    ss << h_A(i, j) << " ";
                }
                ss << std::endl;
            }

            return ss.str();
        }

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
        array_xxc dft_cc(const array_xxc& arr, int dim = 1)
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
        array_xxc idft_cc(const array_xxc& arr, int dim = 1)
        {
            Index N0 = arr.extent(0);
            Index N1 = arr.extent(1);
            auto ret = gen_2d_view<array_xxc>(N0, N1, 0);
            // TODO place holder for now, implement a real one!
            Kokkos::parallel_for("idft_cc",
                                 Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {N0, N1}),
                                 KOKKOS_LAMBDA(const Index& i0, const Index& i1) { ret(i0, i1) = arr(i0, i1)+1; });
            return ret;
        }
        array_xxf idft_cr(const array_xxc& arr, int dim = 0)
        {
            Index N0 = arr.extent(0);
            Index N1 = arr.extent(1);
            auto ret = gen_2d_view<array_xxf>(N0, N1, 0);
            // TODO place holder for now, implement a real one!
            Kokkos::parallel_for("idft_cr",
                                 Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {N0, N1}),
                                 KOKKOS_LAMBDA(const Index& i0, const Index& i1) { ret(i0, i1) = Kokkos::real(arr(i0, i1))+1; });
            return ret;
        }

    }  // namespace KokkosArray
}  // namespace WireCell

#endif
