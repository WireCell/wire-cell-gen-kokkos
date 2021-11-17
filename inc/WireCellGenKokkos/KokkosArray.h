/**
 * Similar like the WireCell::Array with Eigen backend,
 * this KokkosArray provides interface for FFTs.
 */

#ifndef WIRECELL_KOKKOSARRAY
#define WIRECELL_KOKKOSARRAY

#include <Kokkos_Core.hpp>

#include <string>
#include <typeinfo>

namespace WireCell {

    namespace KokkosArray {
        using Scalar = float;
        using Index = int;
        using Layout = Kokkos::LayoutLeft; /// left layout -> column major
        using Space = Kokkos::DefaultExecutionSpace;

        /// A real, 1D array
        typedef Kokkos::View<Scalar*, Layout, Space> array_xf;
        typedef Kokkos::View<Scalar*, Layout, Kokkos::HostSpace> array_xf_h;

        /// A complex, 1D array
        typedef Kokkos::View<Kokkos::complex<Scalar>*, Layout, Space> array_xc;
        typedef Kokkos::View<Kokkos::complex<Scalar>*, Layout, Kokkos::HostSpace> array_xc_h;

        /// A real, 2D array
        typedef Kokkos::View<Scalar**, Layout, Space> array_xxf;

        /// A complex, 2D array
        typedef Kokkos::View<Kokkos::complex<Scalar>**, Layout, Space> array_xxc;

        /// Generate a 1D view initialized with given value.
        template <class ViewType>
        inline ViewType gen_1d_view(const Index N0, const Scalar val = 0)
        {
            ViewType ret(Kokkos::view_alloc("ret", Kokkos::WithoutInitializing), N0);
            Kokkos::deep_copy(ret, val);
            return ret;
        }

        /// Generate a 2D view initialized with given value.
        template <class ViewType>
        inline ViewType gen_2d_view(const Index N0, const Index N1, const Scalar val = 0)
        {
            ViewType ret(Kokkos::view_alloc("ret", Kokkos::WithoutInitializing), N0, N1);
            Kokkos::deep_copy(ret, val);
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
            ss << typeid(ViewType).name() << ", shape: {" << A.extent(0) << ", " << A.extent(1) << "} :\n";

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
            ss << typeid(ViewType).name() << ", shape: {" << A.extent(0) << "} :\n";

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

#ifdef   KOKKOS_ENABLE_HIP
#include "WireCellGenKokkos/KokkosArray_hip.h"
#else

#include "WireCellGenKokkos/KokkosArray_fftw.h"
// #endif

#endif

#endif
