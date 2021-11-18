/**
 * Wrappers for cuFFT based FFT
 */

#ifndef WIRECELL_KOKKOSARRAY_CUDA
#define WIRECELL_KOKKOSARRAY_CUDA

#include <cufft.h>

namespace WireCell {

    namespace KokkosArray {
        inline array_xc dft(const array_xf& in)
        {
            Index N0 = in.extent(0);
            auto out = gen_1d_view<array_xc>(N0, 0);
            cufftHandle plan;
            cufftPlan1d(&plan, N0, CUFFT_R2C, 1);
            cufftExecR2C(plan, (cufftReal*) in.data(), (cufftComplex*) out.data());
            cufftDestroy(plan);
            
            return out;
        }
        inline array_xf idft(const array_xc& in)
        {
            Index N0 = in.extent(0);
            auto out = gen_1d_view<array_xf>(N0, 0);
            cufftHandle plan;
            cufftPlan1d(&plan, N0, CUFFT_C2R, 1);
            cufftExecC2R(plan, (cufftComplex*) in.data(), (cufftReal*) out.data());
            cufftDestroy(plan);
            Kokkos::parallel_for(N0, KOKKOS_LAMBDA(const KokkosArray::Index& i0) {
                out(i0) /= N0;
            });

            return out;
        }
        inline array_xxc dft_rc(const array_xxf& in, int dim = 0)
        {
            std::cout << "WIRECELL_KOKKOSARRAY_CUDA" << std::endl;
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
            auto out = gen_2d_view<array_xxc>(N0, N1, 0);

            cufftHandle plan;

            if (dim == 0) {
                int n[] = {(int) N1};
                int inembed[] = {(int) N1};
                int onembed[] = {(int) N1};
                cufftPlanMany(&plan, 1, n, inembed, (int) N0, 1, onembed, (int) N0, 1, CUFFT_R2C, (int) N0);
                cufftExecR2C(plan, (cufftReal*) in.data(), (cufftComplex*) out.data());
                cufftDestroy(plan);
            }

            if (dim == 1) {
                int n[] = {(int) N0};
                int inembed[] = {(int) N0};
                int onembed[] = {(int) N0};
                cufftPlanMany(&plan, 1, n, inembed, 1, (int) N0, onembed, 1, (int) N0, CUFFT_R2C, (int) N1);
                cufftExecR2C(plan, (cufftReal*) in.data(), (cufftComplex*) out.data());
                cufftDestroy(plan);
            }

            return out;
        }
        inline array_xxc dft_cc(const array_xxc& in, int dim = 0)
        {
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
            auto out = gen_2d_view<array_xxc>(N0, N1, 0);

            cufftHandle plan;

            if (dim == 0) {
                int n[] = {(int) N1};
                int inembed[] = {(int) N1};
                int onembed[] = {(int) N1};
                cufftPlanMany(&plan, 1, n, inembed, (int) N0, 1, onembed, (int) N0, 1, CUFFT_C2C, (int) N0);
                cufftExecC2C(plan, (cufftComplex*) in.data(), (cufftComplex*) out.data(), CUFFT_FORWARD);
                cufftDestroy(plan);
            }

            if (dim == 1) {
                int n[] = {(int) N0};
                int inembed[] = {(int) N0};
                int onembed[] = {(int) N0};
                cufftPlanMany(&plan, 1, n, inembed, 1, (int) N0, onembed, 1, (int) N0, CUFFT_C2C, (int) N1);
                cufftExecC2C(plan, (cufftComplex*) in.data(), (cufftComplex*) out.data(), CUFFT_FORWARD);
                cufftDestroy(plan);
            }

            return out;
        }
        inline array_xxc idft_cc(const array_xxc& in, int dim = 0)
        {
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
            auto out = gen_2d_view<array_xxc>(N0, N1, 0);

            cufftHandle plan;

            if (dim == 0) {
                int n[] = {(int) N1};
                int inembed[] = {(int) N1};
                int onembed[] = {(int) N1};
                cufftPlanMany(&plan, 1, n, inembed, (int) N0, 1, onembed, (int) N0, 1, CUFFT_C2C, (int) N0);
                cufftExecC2C(plan, (cufftComplex*) in.data(), (cufftComplex*) out.data(), CUFFT_INVERSE);
                cufftDestroy(plan);
                Kokkos::parallel_for(
                    Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {N0, N1}),
                    KOKKOS_LAMBDA(const KokkosArray::Index& i0, const KokkosArray::Index& i1) { out(i0, i1) /= N1; });
            }

            if (dim == 1) {
                int n[] = {(int) N0};
                int inembed[] = {(int) N0};
                int onembed[] = {(int) N0};
                cufftPlanMany(&plan, 1, n, inembed, 1, (int) N0, onembed, 1, (int) N0, CUFFT_C2C, (int) N1);
                cufftExecC2C(plan, (cufftComplex*) in.data(), (cufftComplex*) out.data(), CUFFT_INVERSE);
                cufftDestroy(plan);
                Kokkos::parallel_for(
                    Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {N0, N1}),
                    KOKKOS_LAMBDA(const KokkosArray::Index& i0, const KokkosArray::Index& i1) { out(i0, i1) /= N0; });
            }

            return out;
        }
        inline array_xxf idft_cr(const array_xxc& in, int dim = 0)
        {
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
            auto out = gen_2d_view<array_xxf>(N0, N1, 0);

            cufftHandle plan;

            if (dim == 0) {
                int n[] = {(int) N1};
                int inembed[] = {(int) N1};
                int onembed[] = {(int) N1};
                cufftPlanMany(&plan, 1, n, inembed, (int) N0, 1, onembed, (int) N0, 1, CUFFT_C2R, (int) N0);
                cufftExecC2R(plan, (cufftComplex*) in.data(), (cufftReal*) out.data());
                cufftDestroy(plan);
                Kokkos::parallel_for(
                    Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {N0, N1}),
                    KOKKOS_LAMBDA(const KokkosArray::Index& i0, const KokkosArray::Index& i1) { out(i0, i1) /= N1; });
            }

            if (dim == 1) {
                int n[] = {(int) N0};
                int inembed[] = {(int) N0};
                int onembed[] = {(int) N0};
                cufftPlanMany(&plan, 1, n, inembed, 1, (int) N0, onembed, 1, (int) N0, CUFFT_C2R, (int) N1);
                cufftExecC2R(plan, (cufftComplex*) in.data(), (cufftReal*) out.data());
                cufftDestroy(plan);
                Kokkos::parallel_for(
                    Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {N0, N1}),
                    KOKKOS_LAMBDA(const KokkosArray::Index& i0, const KokkosArray::Index& i1) { out(i0, i1) /= N0; });
            }

            return out;
        }

    }  // namespace KokkosArray
}  // namespace WireCell

#endif
