/**
 * Wrappers for FFTW based FFT
 */

#ifndef WIRECELL_KOKKOSARRAY_FFTW
#define WIRECELL_KOKKOSARRAY_FFTW

#include <WireCellUtil/Array.h>
#include <iostream> //debug

namespace WireCell {

    namespace KokkosArray {
        /**
         * using eigen based WireCell::Array for fast prototyping
         * this may introduce extra data copying, will investigate later
         * by default eigen is left layout (column major)
         * https://eigen.tuxfamily.org/dox/group__TopicStorageOrders.html
         */
        inline array_xxc dft_rc(const array_xxf& in, int dim = 0)
        {
            std::cout << "WIRECELL_KOKKOSARRAY_FFTW" << std::endl;
            Eigen::Map<Eigen::ArrayXXf> in_eigen((float*) in.data(), in.extent(0), in.extent(1));
            auto out_eigen = WireCell::Array::dft_rc(in_eigen, dim);
            auto out = gen_2d_view<array_xxc>(out_eigen.rows(), out_eigen.cols(), 0);
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar) * 2);

            return out;
        }
        inline array_xxc dft_cc(const array_xxc& in, int dim = 0)
        {
            Eigen::Map<Eigen::ArrayXXcf> in_eigen((std::complex<float>*) in.data(), in.extent(0), in.extent(1));
            auto out_eigen = WireCell::Array::dft_cc(in_eigen, dim);
            auto out = gen_2d_view<array_xxc>(out_eigen.rows(), out_eigen.cols(), 0);
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar) * 2);

            return out;
        }
        inline array_xxc idft_cc(const array_xxc& in, int dim = 0)
        {
            Eigen::Map<Eigen::ArrayXXcf> in_eigen((std::complex<float>*) in.data(), in.extent(0), in.extent(1));
            auto out_eigen = WireCell::Array::idft_cc(in_eigen, dim);
            auto out = gen_2d_view<array_xxc>(out_eigen.rows(), out_eigen.cols(), 0);
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar) * 2);

            return out;
        }
        inline array_xxf idft_cr(const array_xxc& in, int dim = 0)
        {
            Eigen::Map<Eigen::ArrayXXcf> in_eigen((std::complex<float>*) in.data(), in.extent(0), in.extent(1));
            auto out_eigen = WireCell::Array::idft_cr(in_eigen, dim);
            auto out = gen_2d_view<array_xxf>(out_eigen.rows(), out_eigen.cols(), 0);
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar));

            return out;
        }

    }  // namespace KokkosArray
}  // namespace WireCell

#endif
