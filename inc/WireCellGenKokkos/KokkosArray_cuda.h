/**
 * Wrappers for cuFFT based FFT
 */

#ifndef WIRECELL_KOKKOSARRAY_CUDA
#define WIRECELL_KOKKOSARRAY_CUDA

#include <cufft.h>

namespace WireCell {

    namespace KokkosArray {
        /**
         * TODO: under developing now
         */
        inline array_xxc dft_rc(const array_xxf& in, int dim = 0)
        {
            std::cout << "WIRECELL_KOKKOSARRAY_CUDA" << std::endl;
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
            auto out = gen_2d_view<array_xxc>(N0, N1, 0);

            cufftHandle plan;

            // FIXME: neede to do 1D FFTs
            // cufftPlan2d(&plan, N0, N1, CUFFT_R2C);
            // cufftExecR2C(plan, (cufftReal*) in.data(), (cufftComplex*) out.data());
            // cufftDestroy(plan);

            if (dim==0) {
                cufftPlan1d(&plan, N1, CUFFT_R2C, 1);
                for(int i=0; i<N0;++i) {
                    auto in_sub = Kokkos::subview(in, (size_t) i, std::make_pair((size_t) 0, (size_t) N1));
                    auto out_sub = Kokkos::subview(out, (size_t) i, std::make_pair((size_t) 0, (size_t) N1));
                    // FIXME: raw pointer won't work if subview is with LayoutStride
                    // https://github.com/kokkos/kokkos/wiki/Subviews#1121-c11-type-deduction
                    cufftExecR2C(plan, (cufftReal*) in_sub.data(), (cufftComplex*) out_sub.data());
                }
                cufftDestroy(plan);
            }

            if (dim==1) {
                cufftPlan1d(&plan, N0, CUFFT_R2C, 1);
                for(int i=0; i<N1;++i) {
                    auto in_sub = Kokkos::subview(in, std::make_pair((size_t) 0, (size_t) N0), (size_t) i);
                    auto out_sub = Kokkos::subview(out, std::make_pair((size_t) 0, (size_t) N0), (size_t) i);
                    cufftExecR2C(plan, (cufftReal*) in_sub.data(), (cufftComplex*) out_sub.data());
                }
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

            // cufftPlan2d(&plan, N0, N1, CUFFT_C2C);
            // cufftExecC2C(plan, (cufftComplex*) in.data(), (cufftComplex*) out.data(), CUFFT_FORWARD);
            // cufftDestroy(plan);

            if (dim==0) {
                cufftPlan1d(&plan, N1, CUFFT_C2C, 1);
                for(int i=0; i<N0;++i) {
                    auto in_sub = Kokkos::subview(in, (size_t) i, std::make_pair((size_t) 0, (size_t) N1));
                    auto out_sub = Kokkos::subview(out, (size_t) i, std::make_pair((size_t) 0, (size_t) N1));
                    cufftExecC2C(plan, (cufftComplex*) in_sub.data(), (cufftComplex*) out_sub.data(), CUFFT_FORWARD);
                }
                cufftDestroy(plan);
            }

            if (dim==1) {
                cufftPlan1d(&plan, N0, CUFFT_C2C, 1);
                for(int i=0; i<N1;++i) {
                    auto in_sub = Kokkos::subview(in, std::make_pair((size_t) 0, (size_t) N0), (size_t) i);
                    auto out_sub = Kokkos::subview(out, std::make_pair((size_t) 0, (size_t) N0), (size_t) i);
                    cufftExecC2C(plan, (cufftComplex*) in_sub.data(), (cufftComplex*) out_sub.data(), CUFFT_FORWARD);
                }
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

            // cufftPlan2d(&plan, N0, N1, CUFFT_C2C);
            // cufftExecC2C(plan, (cufftComplex*) in.data(), (cufftComplex*) out.data(), CUFFT_INVERSE);
            // cufftDestroy(plan);

            if (dim==0) {
                cufftPlan1d(&plan, N1, CUFFT_C2C, 1);
                for(int i=0; i<N0;++i) {
                    auto in_sub = Kokkos::subview(in, (size_t) i, std::make_pair((size_t) 0, (size_t) N1));
                    auto out_sub = Kokkos::subview(out, (size_t) i, std::make_pair((size_t) 0, (size_t) N1));
                    cufftExecC2C(plan, (cufftComplex*) in_sub.data(), (cufftComplex*) out_sub.data(), CUFFT_INVERSE);
                }
                cufftDestroy(plan);
            }

            if (dim==1) {
                cufftPlan1d(&plan, N0, CUFFT_C2C, 1);
                for(int i=0; i<N1;++i) {
                    auto in_sub = Kokkos::subview(in, std::make_pair((size_t) 0, (size_t) N0), (size_t) i);
                    auto out_sub = Kokkos::subview(out, std::make_pair((size_t) 0, (size_t) N0), (size_t) i);
                    cufftExecC2C(plan, (cufftComplex*) in_sub.data(), (cufftComplex*) out_sub.data(), CUFFT_INVERSE);
                }
                cufftDestroy(plan);
            }

            return out;
        }
        inline array_xxf idft_cr(const array_xxc& in, int dim = 0)
        {
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
            auto out = gen_2d_view<array_xxf>(N0, N1, 0);
            
            cufftHandle plan;

            // cufftPlan2d(&plan, N0, N1, CUFFT_C2R);
            // cufftExecC2R(plan, (cufftComplex*) in.data(), (cufftReal*) out.data());
            // cufftDestroy(plan);

            if (dim==0) {
                cufftPlan1d(&plan, N1, CUFFT_C2R, 1);
                for(int i=0; i<N0;++i) {
                    auto in_sub = Kokkos::subview(in, (size_t) i, std::make_pair((size_t) 0, (size_t) N1));
                    auto out_sub = Kokkos::subview(out, (size_t) i, std::make_pair((size_t) 0, (size_t) N1));
                    cufftExecC2R(plan, (cufftComplex*) in_sub.data(), (cufftReal*) out_sub.data());
                }
                cufftDestroy(plan);
            }

            if (dim==1) {
                cufftPlan1d(&plan, N0, CUFFT_C2R, 1);
                for(int i=0; i<N1;++i) {
                    auto in_sub = Kokkos::subview(in, std::make_pair((size_t) 0, (size_t) N0), (size_t) i);
                    auto out_sub = Kokkos::subview(out, std::make_pair((size_t) 0, (size_t) N0), (size_t) i);
                    cufftExecC2R(plan, (cufftComplex*) in_sub.data(), (cufftReal*) out_sub.data());
                }
                cufftDestroy(plan);
            }

            return out;
        }

    }  // namespace KokkosArray
}  // namespace WireCell

#endif
