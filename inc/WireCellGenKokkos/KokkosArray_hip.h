/**
 * Wrappers for hipFFT based FFT
 */

#ifndef WIRECELL_KOKKOSARRAY_HIP
#define WIRECELL_KOKKOSARRAY_HIP

#include <hipfft.h>
#include <assert.h>

namespace WireCell {

    namespace KokkosArray {
        /**
         * TODO:
         *  - round trip norm not included yet
         */
        inline array_xxc dft_rc(const array_xxf& in, int dim = 0)
        {
            std::cout << "WIRECELL_KOKKOSARRAY_HIP" << std::endl;
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
            auto out = gen_2d_view<array_xxc>(N0, N1, 0);

            hipfftHandle plan = NULL;
            
            hipfftResult status ;
            hipError_t sync_status ;

	    if (dim == 0) {
                int n[] = { (int) N1};
                int inembed[] = {(int) N1};
                int onembed[] = {(int) N1};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;	
	        // set to autoAllocate
	        status = hipfftSetAutoAllocation( plan, 1);
        	assert(status == HIPFFT_SUCCESS) ;
	
		//MakePlan
	        status = hipfftMakePlanMany( plan, 1, n, inembed, (int) N0, 1, onembed, N0, 1, HIPFFT_R2C, (int) N0, &worksize);
        	//std::cout<<"worksize= "<<worksize << std::endl ;
        	assert(status == HIPFFT_SUCCESS) ;
                
		//Excute
		status = hipfftExecR2C( plan,  (float * )in.data(), (float2 *) out.data() ) ;
		assert(status == HIPFFT_SUCCESS) ;

		// Wait for execution to finish
		sync_status=hipDeviceSynchronize() ;

		//Destroy plan
		hipfftDestroy(plan);
            }

	    if (dim == 1) {
                int n[] = {(int) N0};
		int inembed[] = {(int) N0};
                int onembed[] = {(int) N0};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed, 1 , (int) N0, onembed, 1, N0, HIPFFT_R2C, (int) N1, &worksize);
                //std::cout<<"worksize= "<<worksize << std::endl ;
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecR2C( plan,  (float * )in.data(), (float2 *) out.data() ) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
		sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);

            }

            return out;
        }
        inline array_xxc dft_cc(const array_xxc& in, int dim = 0)
        {
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
            auto out = gen_2d_view<array_xxc>(N0, N1, 0);

            hipfftHandle plan ;
            
            hipfftResult status ;
            hipError_t sync_status ;
            
	    if (dim == 0) {
                int n[] = { (int) N1};
                int inembed[] = {(int) N1};
                int onembed[] = {(int) N1};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;	
	        // set to autoAllocate
	        status = hipfftSetAutoAllocation( plan, 1);
        	assert(status == HIPFFT_SUCCESS) ;
	
		//MakePlan
	        status = hipfftMakePlanMany( plan, 1, n, inembed, (int) N0, 1, onembed, N0, 1, HIPFFT_C2C, (int) N0, &worksize);
        	//std::cout<<"worksize= "<<worksize << std::endl ;
        	assert(status == HIPFFT_SUCCESS) ;
                
		//Excute
		status = hipfftExecC2C( plan,  (hipfftComplex * )in.data(), (hipfftComplex *) out.data() , HIPFFT_FORWARD) ;
		assert(status == HIPFFT_SUCCESS) ;

		// Wait for execution to finish
		sync_status=hipDeviceSynchronize() ;

		//Destroy plan
		hipfftDestroy(plan);
            }

	    if (dim == 1) {
                int n[] = {(int) N0};
		int inembed[] = {(int) N0};
                int onembed[] = {(int) N0};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed,  1, N0, onembed, 1, N0, HIPFFT_C2C, (int) N1, &worksize);
                //std::cout<<"worksize= "<<worksize << std::endl ;
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecC2C( plan,  (hipfftComplex  * )in.data(), (hipfftComplex *) out.data() , HIPFFT_FORWARD) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
		sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);

            }

            return out;
        }
        inline array_xxc idft_cc(const array_xxc& in, int dim = 0)
        {
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
            auto out = gen_2d_view<array_xxc>(N0, N1, 0);

            hipfftHandle plan ;
            
            hipfftResult status ;
            hipError_t sync_status ;
            
	    if (dim == 0) {
                int n[] = { (int) N1};
                int inembed[] = {(int) N1};
                int onembed[] = {(int) N1};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;	
	        // set to autoAllocate
	        status = hipfftSetAutoAllocation( plan, 1);
        	assert(status == HIPFFT_SUCCESS) ;
	
		//MakePlan
	        status = hipfftMakePlanMany( plan, 1, n, inembed, (int) N0, 1, onembed, N0, 1, HIPFFT_C2C, (int) N0, &worksize);
        	//std::cout<<"worksize= "<<worksize << std::endl ;
        	assert(status == HIPFFT_SUCCESS) ;
                
		//Excute
		status = hipfftExecC2C( plan,  (float2 * )in.data(), (float2 *) out.data(), HIPFFT_BACKWARD ) ;
		assert(status == HIPFFT_SUCCESS) ;

		// Wait for execution to finish
		sync_status=hipDeviceSynchronize() ;

		//Destroy plan
		hipfftDestroy(plan);
            }

	    if (dim == 1) {
                int n[] = {(int) N0};
		int inembed[] = {(int) N0};
                int onembed[] = {(int) N0};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed, 1 ,N0, onembed, 1, N0, HIPFFT_C2C, (int) N1, &worksize);
                //std::cout<<"worksize= "<<worksize << std::endl ;
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecC2C( plan,  (float2 * )in.data(), (float2 *) out.data() ,HIPFFT_BACKWARD) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
		sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);

            }

            return out;
        }
        inline array_xxf idft_cr(const array_xxc& in, int dim = 0)
        {
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
            auto out = gen_2d_view<array_xxf>(N0, N1, 0);

            hipfftHandle plan = NULL;
            
            hipfftResult status ;
            hipError_t sync_status ;
            
	    if (dim == 0) {
                int n[] = { (int) N1};
                int inembed[] = {(int) N1};
                int onembed[] = {(int) N1};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;	
	        // set to autoAllocate
	        status = hipfftSetAutoAllocation( plan, 1);
        	assert(status == HIPFFT_SUCCESS) ;
	
		//MakePlan
	        status = hipfftMakePlanMany( plan, 1, n, inembed, (int) N0, 1, onembed, N0, 1, HIPFFT_C2R, (int) N0, &worksize);
        	assert(status == HIPFFT_SUCCESS) ;
                
		//Excute
		status = hipfftExecC2R( plan,  (float2 * )in.data(), (float *) out.data() ) ;
		assert(status == HIPFFT_SUCCESS) ;

		// Wait for execution to finish
		sync_status=hipDeviceSynchronize() ;

		//Destroy plan
		hipfftDestroy(plan);
            }

	    if (dim == 1) {
                int n[] = {(int) N0};
		int inembed[] = {(int) N0};
                int onembed[] = {(int) N0};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed, 1, N0, onembed, 1, N0, HIPFFT_C2R, (int) N1, &worksize);
                //std::cout<<"worksize= "<<worksize << std::endl ;
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecC2R( plan,  (float2 * )in.data(), (float *) out.data() ) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
		sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);

            }

            return out;
        }
    }  // namespace KokkosArray
}  // namespace WireCell

#endif
