/**
 * Similar like the WireCell::Array with Eigen backend,
 * this KokkosArray provides interface for FFTs.
 */

#ifndef WIRECELL_KOKKOSARRAY_CUDA
#define WIRECELL_KOKKOSARRAY_CUDA

#include <cufft.h>

namespace WireCell {

    namespace KokkosArray {
        /**
         * Below are 4 functions for for the FFT operations.
         * The actual operation is just a placeholder, adding "1" for debugging purpose.
         */
        inline array_xxc dft_rc(const array_xxf& in, int dim = 0)
        {
            std::cout << "WIRECELL_KOKKOSARRAY_CUDA" << std::endl;
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
            auto out = gen_2d_view<array_xxc>(N0, N1, 0);

            cufftHandle plan;
            // FIXME neede to do 1D FFTs
            cufftPlan2d(&plan, N0, N1, CUFFT_R2C);
            cufftExecR2C(plan, (cufftReal*) in.data(), (cufftComplex*) out.data());
            cufftDestroy(plan);

            return out;
        }
        inline array_xxc dft_cc(const array_xxc& in, int dim = 0)
        {
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
            auto out = gen_2d_view<array_xxc>(N0, N1, 0);
            
            cufftHandle plan;
            cufftPlan2d(&plan, N0, N1, CUFFT_C2C);
            cufftExecC2C(plan, (cufftComplex*) in.data(), (cufftComplex*) out.data(), CUFFT_FORWARD);
            cufftDestroy(plan);

            return out;
        }
        inline array_xxc idft_cc(const array_xxc& in, int dim = 0)
        {
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
            auto out = gen_2d_view<array_xxc>(N0, N1, 0);
            
            cufftHandle plan;
            cufftPlan2d(&plan, N0, N1, CUFFT_C2C);
            cufftExecC2C(plan, (cufftComplex*) in.data(), (cufftComplex*) out.data(), CUFFT_INVERSE);
            cufftDestroy(plan);

            return out;
        }
        inline array_xxf idft_cr(const array_xxc& in, int dim = 0)
        {
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
            auto out = gen_2d_view<array_xxf>(N0, N1, 0);
            
            cufftHandle plan;
            cufftPlan2d(&plan, N0, N1, CUFFT_C2R);
            cufftExecC2R(plan, (cufftComplex*) in.data(), (cufftReal*) out.data());
            cufftDestroy(plan);

            return out;
        }

    }  // namespace KokkosArray
}  // namespace WireCell

#endif
