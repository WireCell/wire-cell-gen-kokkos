/**
 * Similar like the WireCell::Array with Eigen backend,
 * this KokkosArray provides interface for FFTs.
 */

#ifndef WIRECELL_KOKKOSARRAY
#define WIRECELL_KOKKOSARRAY

#include "WireCellUtil/Waveform.h"

#include <Kokkos_Core.hpp>

#include <string>
#include <typeinfo>

namespace WireCell {

    namespace KokkosArray {
        using Scalar = float;
        using Index = int;
        using Layout = Kokkos::LayoutLeft;
        using Space = Kokkos::DefaultExecutionSpace;

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
        template <class ViewType>
        inline ViewType Zero(const Index N0, const Index N1)
        {
            return gen_2d_view<ViewType>(N0, N1, 0);
        }

        /// Dump out a string for pinting for a 2D view.
        template <class ViewType>
        inline std::string dump_2d_view(const ViewType& A, const Index length_limit = 20)
        {
            std::stringstream ss;
            ss << typeid(ViewType).name() << ":\n";

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

            bool all_zero = true;
            for (Index i = 0; i < N0 && all_zero == true; ++i) {
                for (Index j = 0; j < N1 && all_zero == true; ++j) {
                    if (h_A(i, j) != 0) {
                        all_zero = false;
                        break;
                    }
                }
            }
            if (all_zero) {
                ss << "All Zero!\n";
            }

            return ss.str();
        }

        /// Dump out a string for pinting for a 1D view.
        template <class ViewType>
        inline std::string dump_1d_view(const ViewType& A, const Index length_limit = 20)
        {
            std::stringstream ss;
            ss << typeid(ViewType).name() << ":\n";

            auto h_A = Kokkos::create_mirror_view(A);
            Kokkos::deep_copy(h_A, A);

            Index N0 = A.extent(0);
            bool print_dot1 = true;
            for (Index j = 0; j < N0; ++j) {
                if (j > length_limit && j < N0 - length_limit) {
                    if (print_dot1) {
                        ss << "... ";
                        print_dot1 = false;
                    }
                    continue;
                }
                ss << h_A(j) << " ";
            }
            ss << std::endl;

            bool all_zero = true;
            for (Index j = 0; j < N0 && all_zero == true; ++j) {
                if (h_A(j) != 0) {
                    all_zero = false;
                    break;
                }
            }
            if (all_zero) {
                ss << "All Zero!\n";
            }

            return ss.str();
        }

    }  // namespace KokkosArray
}  // namespace WireCell

#ifdef KOKKOS_ENABLE_CUDA
#include "WireCellGenKokkos/KokkosArray_cuda.h"
#else
#include "WireCellGenKokkos/KokkosArray_fftw.h"
#endif

#endif
