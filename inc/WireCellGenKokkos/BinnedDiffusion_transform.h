#ifndef WIRECELL_GENKOKKOS_BINNEDDIFFUSION_TRANSFORM
#define WIRECELL_GENKOKKOS_BINNEDDIFFUSION_TRANSFORM

#include "WireCellUtil/Pimpos.h"
#include "WireCellUtil/Point.h"
#include "WireCellUtil/Units.h"
#include "WireCellIface/IDepo.h"

#include "WireCellGenKokkos/ImpactData.h"
#include "WireCellGenKokkos/KokkosArray.h"

#include <deque>
#include <tuple>
#include <Eigen/Sparse>

#include "config.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

#define MAX_NPSS_DEVICE 1000
#define MAX_NTSS_DEVICE 1000



namespace WireCell {
    namespace GenKokkos {

      /* struct GausDiffTimeCompare{ */
      /* 	bool operator()(const std::shared_ptr<GenKokkos::GaussianDiffusion>& lhs, const std::shared_ptr<GenKokkos::GaussianDiffusion>& rhs) const; */
      /* }; */
	/**  A BinnedDiffusion_transform maintains an association between impact
	 * positions along the pitch direction of a wire plane and
	 * the diffused depositions that drift to them.
         *
         * It covers a fixed and discretely sampled time and pitch
         * domain.
	 */
	class BinnedDiffusion_transform {
	public:

	    /** Create a BinnedDiffusion_transform. 

               Arguments are:
	      	
               - pimpos :: a Pimpos instance defining the wire and impact binning.

               - tbins :: a Binning instance defining the time sampling binning.

	       - nsigma :: number of sigma the 2D (transverse X
                 longitudinal) Gaussian extends.
              
	       - fluctuate :: set to an IRandom if charge-preserving
                 Poisson fluctuations are to be applied.

               - calcstrat :: set a calculation strategy that gives
                 how the microscopic distribution of charge between
                 two impacts will be interpolated toward either edge.

	     */

            /// Useful to client code to mark a calculation strategy. 
            enum ImpactDataCalculationStrategy { constant=1, linear=2 };

	    BinnedDiffusion_transform(const Pimpos& pimpos, const Binning& tbins,
			    double nsigma=3.0, IRandom::pointer fluctuate=nullptr,
                            ImpactDataCalculationStrategy calcstrat = linear);
            //#ifdef HAVE_CUDA_INC
            virtual ~BinnedDiffusion_transform();
            //#endif

            const Pimpos& pimpos() const { return m_pimpos; }
            const Binning& tbins() const { return m_tbins; }

	    /// Add a deposition and its associated diffusion sigmas.
	    /// Return false if no activity falls within the domain.
	    bool add(IDepo::pointer deposition, double sigma_time, double sigma_pitch);

	    /// Unconditionally associate an already built
	    /// GaussianDiffusion to one impact.  
	    //void add(std::shared_ptr<GaussianDiffusion> gd, int impact_index);

	    /// Drop any stored ImpactData within the half open
	    /// impact index range.
	    // void erase(int begin_impact_index, int end_impact_index);

	    /// Return the data in the given impact bin.  Note, this
	    /// bin represents drifted charge between two impact
	    /// positions.  Take care when using BinnedDiffusion_transform and
	    /// field responses because epsilon above or below the
	    /// impact position exactly in the middle of two wires
	    /// drastically different response.
	    //ImpactData::pointer impact_data(int bin) const;

	    // test ... 
	    void get_charge_vec(std::vector<std::vector<std::tuple<int,int, double> > >& vec_vec_charge, std::vector<int>& vec_impact);
	    void get_charge_matrix(std::vector<Eigen::SparseMatrix<float>* >& vec_spmatrix, std::vector<int>& vec_impact);
        void get_charge_matrix_kokkos(KokkosArray::array_xxf& out,
        std::vector<int>& vec_impact,
        const int start_pitch,
        const int start_tick);
	    
	    
            /// Return the range of pitch containing depos out to
            /// given nsigma and without bounds checking.
            std::pair<double,double> pitch_range(double nsigma=0.0) const;

            /// Return the half open bin range of impact bins,
            /// constrained so that either number is in [0,nimpacts].
            std::pair<int,int> impact_bin_range(double nsigma=0.0) const;

            /// Return the range of time containing depos out to given
            /// nsigma and without bounds checking.
            std::pair<double,double> time_range(double nsigma=0.0) const;

            /// Return the half open bin range for time bins
            /// constrained so that either number is in [0,nticks].
            std::pair<int,int> time_bin_range(double nsigma=0.0) const;

	    double get_nsigma() const {return m_nsigma;};

	    std::tuple< Kokkos::View<float*>, Kokkos::View<unsigned long*> >
        set_sampling_bat( const unsigned long npatch, int patch_size, const bool copy_DtoH = true) ;
	    
	private:
	    
            const Pimpos& m_pimpos;
            const Binning& m_tbins;

	    double m_nsigma;
            IRandom::pointer m_fluctuate;
            ImpactDataCalculationStrategy m_calcstrat;

	    // current window set by user.
	    std::pair<int,int> m_window;
	    // the content of the current window
	    std::map<int, GenKokkos::ImpactData::mutable_pointer> m_impacts;
            //std::vector<std::shared_ptr<GaussianDiffusion> > m_diffs;
	    //std::set<std::shared_ptr<GaussianDiffusion>, GausDiffTimeCompare> m_diffs;
	    std::vector<std::shared_ptr<GaussianDiffusion> > m_diffs;

            int m_outside_pitch;
            int m_outside_time;

            //double* m_pvec_D;
            //double* m_tvec_D;
            //float* m_patch_D;
            //float* m_rand_D;
            //curandGenerator_t m_Gen;

            //Kokkos::DualView<double[MAX_NPSS_DEVICE]> m_pvec;
            //Kokkos::DualView<double[MAX_NTSS_DEVICE]> m_tvec;
            Kokkos::View<float*> m_patch;
            Kokkos::View<double*> m_normals;
            Kokkos::DualView<double*> m_ptvecs;
	    //for batch
            Kokkos::View<double*> m_pvecs;
            Kokkos::View<double*> m_tvecs;
            Kokkos::View<double*> m_charges;
            Kokkos::View<unsigned long *> m_t_idx;
            Kokkos::View<unsigned long *> m_p_idx;
            Kokkos::View<unsigned long *> m_patch_idx;
	    void* m_ptvecs_h ;
	    double* m_pvecs_h ;
	    double* m_tvecs_h ;
	    double* m_charges_h ;
	    float * m_patch_h ;
	    unsigned long * m_t_idx_h;
	    unsigned long * m_p_idx_h ;
	    unsigned long * m_patch_idx_h;
	    

        private:
            //#ifdef HAVE_CUDA_INC
            void init_Device();
            void clear_Device();
            //#endif


	};


    }

}

#endif
