#include "WireCellGenKokkos/BinnedDiffusion_transform.h"
#include "WireCellGenKokkos/GaussianDiffusion.h"
#include "WireCellUtil/Units.h"

#include <iostream>             // debug
#include <omp.h>
#include <unordered_map>
#include <cmath>
#include <typeinfo>


#include <Kokkos_Core.hpp>

#include <Kokkos_Random.hpp>
#include <Kokkos_DualView.hpp>
#include <impl/Kokkos_Timer.hpp>


#define MAX_PATCH_SIZE 1024
#define P_BLOCK_SIZE  512
#define MAX_PATCHES 200000
#define MAX_NPSS_DEVICE 1000
#define MAX_NTSS_DEVICE 1000
#define FULL_MASK 0xffffffff
#define RANDOM_BLOCK_SIZE (1024*1024)
#define RANDOM_BLOCK_NUM 512
//#define MAX_RANDOM_LENGTH (RANDOM_BLOCK_NUM*RANDOM_BLOCK_SIZE)
#define MAX_RANDOM_LENGTH (MAX_PATCH_SIZE*MAX_PATCHES)
#define PI 3.14159265358979323846



using namespace std;

using namespace WireCell;
//namespace wc = WireCell::Kokkos;
//using namespace Kokkos;

double g_get_charge_vec_time_part1 = 0.0;
double g_get_charge_vec_time_part2 = 0.0;
double g_get_charge_vec_time_part3 = 0.0;
double g_get_charge_vec_time_part4 = 0.0;
double g_get_charge_vec_time_part5 = 0.0;

extern double g_set_sampling_part1;
extern double g_set_sampling_part2;
extern double g_set_sampling_part3;
extern double g_set_sampling_part4;
extern double g_set_sampling_part5;

extern size_t g_total_sample_size;



template <class GeneratorPool>
struct generate_random {
    Kokkos::View<double*> normals; // Normal distribution N(0,1)
    GeneratorPool rand_pool1;
    GeneratorPool rand_pool2;
    int samples;
    uint64_t range_min;
    uint64_t range_max1 = 0;
    uint64_t range_max2 = 0;


    generate_random(Kokkos::View<double*> normals_, GeneratorPool rand_pool1_, GeneratorPool rand_pool2_, int samples_)
        : normals(normals_), rand_pool1(rand_pool1_), rand_pool2(rand_pool2_), samples(samples_), range_min(1) {
        //range_max1 = rand_pool1.get_state().max();
        //range_max2 = rand_pool2.get_state().max();
        range_max1 = 0xffffffffffffffffULL-1;
        range_max2 = range_max1;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(int i) const {
        //*
        typename GeneratorPool::generator_type rand_gen1 = rand_pool1.get_state();
        typename GeneratorPool::generator_type rand_gen2 = rand_pool2.get_state();

        for (int k = 0; k < samples/2; k++) {
            double u1 = (double) rand_gen1.urand64(range_min, range_max1) / range_max1;
            double u2 = (double) rand_gen2.urand64(range_min, range_max2) / range_max2;
            normals(i * samples + 2*k)     = sqrt(-2*log(u1)) * cos(2*PI*u2);
            normals(i * samples + 2*k + 1) = sqrt(-2*log(u1)) * sin(2*PI*u2);
        }

        rand_pool1.free_state(rand_gen1);
        rand_pool2.free_state(rand_gen2);
        //*/
    }
   
};












// bool GenKokkos::GausDiffTimeCompare::operator()(const std::shared_ptr<GenKokkos::GaussianDiffusion>& lhs, const std::shared_ptr<GenKokkos::GaussianDiffusion>& rhs) const
// {
//   if (lhs->depo_time() == rhs->depo_time()) {
//     if (lhs->depo_x() == lhs->depo_x()) {
//       return lhs.get() < rhs.get(); // break tie by pointer
//     }
//     return lhs->depo_x() < lhs->depo_x();
//   }
//   return lhs->depo_time() < rhs->depo_time();
// }


GenKokkos::BinnedDiffusion_transform::BinnedDiffusion_transform(const Pimpos& pimpos, const Binning& tbins,
                                      double nsigma, IRandom::pointer fluctuate,
                                      ImpactDataCalculationStrategy calcstrat)
    : m_pimpos(pimpos)
    , m_tbins(tbins)
    , m_nsigma(nsigma)
    , m_fluctuate(fluctuate)
    , m_calcstrat(calcstrat)
    , m_window(0,0)
    , m_outside_pitch(0)
    , m_outside_time(0)
{
    //Kokkos::realloc(m_patch, MAX_PATCH_SIZE*MAX_PATCHES);
    //Kokkos::realloc(m_ptvecs, MAX_NPSS_DEVICE+MAX_NTSS_DEVICE);
    m_ptvecs_h = (void*)malloc((MAX_NPSS_DEVICE+MAX_NTSS_DEVICE)*sizeof(double)) ;
    m_pvecs_h = (double*)malloc((MAX_NPSS_DEVICE*MAX_PATCHES)*sizeof(double)) ;
    m_tvecs_h = (double*)malloc((MAX_NTSS_DEVICE*MAX_PATCHES)*sizeof(double)) ;
    m_charges_h = (double*)malloc(MAX_PATCHES*sizeof(double)) ;
    m_patch_h = (float*)malloc(MAX_PATCHES*MAX_PATCH_SIZE*sizeof(float)) ;
    m_p_idx_h = (unsigned long *)malloc((MAX_PATCHES+1)*sizeof(unsigned long)) ;
    m_t_idx_h = (unsigned long*)malloc((MAX_PATCHES+1)*sizeof(unsigned long)) ;
    m_patch_idx_h = (unsigned long*)malloc((MAX_PATCHES+1)*sizeof(unsigned long)) ;
    init_Device();
}


GenKokkos::BinnedDiffusion_transform::~BinnedDiffusion_transform() {
    clear_Device();
    free(m_ptvecs_h);
    free(m_pvecs_h);
    free(m_tvecs_h);
    free(m_charges_h);
    free(m_patch_h);
    free(m_p_idx_h);
    free(m_t_idx_h);
    free(m_patch_idx_h);
}



void GenKokkos::BinnedDiffusion_transform::init_Device() {


    //size_t size = RANDOM_BLOCK_NUM;
    //size_t samples = RANDOM_BLOCK_SIZE;
    size_t size = MAX_PATCHES;
    size_t samples = MAX_PATCH_SIZE;
    int seed = 2020;

    Kokkos::Random_XorShift64_Pool<> rand_pool1(seed);
    Kokkos::Random_XorShift64_Pool<> rand_pool2(seed+1);
    Kokkos::resize(m_normals, size * samples);

    Kokkos::parallel_for(size*samples/256, generate_random<Kokkos::Random_XorShift64_Pool<> >(m_normals, rand_pool1, rand_pool2, 256));
}


void GenKokkos::BinnedDiffusion_transform::clear_Device() {
  //CUDA_SAFE_CALL(cudaFree(m_pvec_D));
  //CUDA_SAFE_CALL(cudaFree(m_tvec_D));
  //CUDA_SAFE_CALL(cudaFree(m_patch_D));
  //CUDA_SAFE_CALL(cudaFree(m_rand_D));

  //CURAND_SAFE_CALL(curandDestroyGenerator(m_Gen));

}




bool GenKokkos::BinnedDiffusion_transform::add(IDepo::pointer depo, double sigma_time, double sigma_pitch)
{

    const double center_time = depo->time();
    const double center_pitch = m_pimpos.distance(depo->pos());

    GenKokkos::GausDesc time_desc(center_time, sigma_time);
    {
        double nmin_sigma = time_desc.distance(m_tbins.min());
        double nmax_sigma = time_desc.distance(m_tbins.max());

        double eff_nsigma = sigma_time>0?m_nsigma:0;
        if (nmin_sigma > eff_nsigma || nmax_sigma < -eff_nsigma) {
            // std::cerr << "BinnedDiffusion_transform: depo too far away in time sigma:"
            //           << " t_depo=" << center_time/units::ms << "ms not in:"
            //           << " t_bounds=[" << m_tbins.min()/units::ms << ","
            //           << m_tbins.max()/units::ms << "]ms"
            //           << " in Nsigma: [" << nmin_sigma << "," << nmax_sigma << "]\n";
            ++m_outside_time;
            return false;
        }
    }

    auto ibins = m_pimpos.impact_binning();

    GenKokkos::GausDesc pitch_desc(center_pitch, sigma_pitch);
    {
        double nmin_sigma = pitch_desc.distance(ibins.min());
        double nmax_sigma = pitch_desc.distance(ibins.max());

        double eff_nsigma = sigma_pitch>0?m_nsigma:0;
        if (nmin_sigma > eff_nsigma || nmax_sigma < -eff_nsigma) {
            // std::cerr << "BinnedDiffusion_transform: depo too far away in pitch sigma: "
            //           << " p_depo=" << center_pitch/units::cm << "cm not in:"
            //           << " p_bounds=[" << ibins.min()/units::cm << ","
            //           << ibins.max()/units::cm << "]cm"
            //           << " in Nsigma:[" << nmin_sigma << "," << nmax_sigma << "]\n";
            ++m_outside_pitch;
            return false;
        }
    }

    // make GD and add to all covered impacts
    // int bin_beg = std::max(ibins.bin(center_pitch - sigma_pitch*m_nsigma), 0);
    // int bin_end = std::min(ibins.bin(center_pitch + sigma_pitch*m_nsigma)+1, ibins.nbins());
    // debug
    //int bin_center = ibins.bin(center_pitch);
    //cerr << "DEBUG center_pitch: "<<center_pitch/units::cm<<endl; 
    //cerr << "DEBUG bin_center: "<<bin_center<<endl;

    auto gd = std::make_shared<GaussianDiffusion>(depo, time_desc, pitch_desc);
    // for (int bin = bin_beg; bin < bin_end; ++bin) {
    //   //   if (bin == bin_beg)  m_diffs.insert(gd);
    //   this->add(gd, bin);
    // }
    m_diffs.push_back(gd);
    return true;
}

// void GenKokkos::BinnedDiffusion_transform::add(std::shared_ptr<GaussianDiffusion> gd, int bin)
// {
//     ImpactData::mutable_pointer idptr = nullptr;
//     auto it = m_impacts.find(bin);
//     if (it == m_impacts.end()) {
// 	idptr = std::make_shared<ImpactData>(bin);
// 	m_impacts[bin] = idptr;
//     }
//     else {
// 	idptr = it->second;
//     }
//     idptr->add(gd);
//     if (false) {                           // debug
//         auto mm = idptr->span();
//         cerr << "GenKokkos::BinnedDiffusion_transform: add: "
//              << " poffoset="<<gd->poffset_bin()
//              << " toffoset="<<gd->toffset_bin()
//              << " charge=" << gd->depo()->charge()/units::eplus << " eles"
//              <<", for bin " << bin << " t=[" << mm.first/units::us << "," << mm.second/units::us << "]us\n";
//     }
//     m_diffs.insert(gd);
//     //m_diffs.push_back(gd);
// }

// void GenKokkos::BinnedDiffusion_transform::erase(int begin_impact_number, int end_impact_number)
// {
//     for (int bin=begin_impact_number; bin<end_impact_number; ++bin) {
// 	m_impacts.erase(bin);
//     }
// }

void GenKokkos::BinnedDiffusion_transform::get_charge_matrix_kokkos(KokkosArray::array_xxf& out,
                                                                    std::vector<int>& vec_impact, const int start_pitch,
                                                                    const int start_tick)
{
    std::cout << "yuhw: get_charge_matrix_kokkos\n";

    double wstart, wend, wstart2, wend2;

    wstart = omp_get_wtime();
    const auto ib = m_pimpos.impact_binning();

    // map between reduced impact # to array #

    std::map<int, int> map_redimp_vec;
    std::vector<std::unordered_map<long int, int> > vec_map_pair_pos;
    for (size_t i = 0; i != vec_impact.size(); i++) {
        map_redimp_vec[vec_impact[i]] = int(i);
        std::unordered_map<long int, int> map_pair_pos;
        vec_map_pair_pos.push_back(map_pair_pos);
    }
    wend = omp_get_wtime();
    g_get_charge_vec_time_part1 += wend - wstart;
    cout << "get_charge_matrix_kokkos(): part1 running time : " << g_get_charge_vec_time_part1 << endl;

    wstart = omp_get_wtime();
    const auto rb = m_pimpos.region_binning();
    // map between impact # to channel #
    std::map<int, int> map_imp_ch;
    // map between impact # to reduced impact #
    std::map<int, int> map_imp_redimp;

    std::cout << "yuhw: " << rb.nbins() << std::endl;
    for (int wireind = 0; wireind != rb.nbins(); wireind++) {
        int wire_imp_no = m_pimpos.wire_impact(wireind);
        std::pair<int, int> imps_range = m_pimpos.wire_impacts(wireind);
        for (int imp_no = imps_range.first; imp_no != imps_range.second; imp_no++) {
            map_imp_ch[imp_no] = wireind;
            map_imp_redimp[imp_no] = imp_no - wire_imp_no;
        }
    }

    int min_imp = 0;
    int max_imp = ib.nbins();
    int counter = 0;

    wend = omp_get_wtime();
    g_get_charge_vec_time_part2 += wend - wstart;
    cout << "get_charge_matrix_kokkos(): part2 running time : " << g_get_charge_vec_time_part2 << endl;

    wstart = omp_get_wtime();
    m_t_idx_h[0] = 0;
    m_p_idx_h[0] = 0;
    m_patch_idx_h[0] = 0;
    int max_patch_size = 0;
    int patch_size = 0;
    for (auto diff : m_diffs) {
        if (diff->depo()->charge() == 0) continue;
        wstart2 = omp_get_wtime();
#ifdef HAVE_CUDA_INC
        diff->set_sampling_CUDA(m_pvec_D, m_tvec_D, m_patch_D, m_rand_D, &m_Gen, m_tbins, ib, m_nsigma, m_fluctuate,
                                m_calcstrat);
#else
        diff->set_sampling_pre(counter, patch_size, m_pvecs_h, m_tvecs_h, m_charges_h, m_p_idx_h, m_t_idx_h,
                               m_patch_idx_h, m_tbins, ib, m_nsigma, m_fluctuate, m_calcstrat);
#endif
        wend2 = omp_get_wtime();
        g_get_charge_vec_time_part4 += wend2 - wstart2;
        counter++;
        max_patch_size = max_patch_size > patch_size ? max_patch_size : patch_size;
    }
    wend = omp_get_wtime();
    cout << "get_charge_matrix_kokkos(): set_sampling_pre() time " << wend - wstart << endl;

    auto patch_info = set_sampling_bat(counter, max_patch_size, false);
    auto patch_d = std::get<0>(patch_info);
    auto patch_idx = std::get<1>(patch_info);
    // std::cout << "patch_d: " << typeid(patch_d).name() << std::endl;
    // std::cout << "patch_idx: " << typeid(patch_idx).name() << std::endl;
    wstart = omp_get_wtime();
    cout << "get_charge_matrix_kokkos(): set_sampling_bat() time " << wstart - wend << endl;

    int min_np = 100;
    int min_nt = 100;
    int poffset_limit[2] = {1000000, 0};
    int toffset_limit[2] = {1000000, 0};
    const int npatch = m_diffs.size();
    std::vector<int> poffset_vec(npatch);
    std::vector<int> toffset_vec(npatch);
    for (int idx=0; idx<npatch; ++idx) {
        auto diff = m_diffs[idx];

        auto patch = diff->get_patch();
        // const auto qweight = diff->weights();
        // memcpy(&(patch.data()[0]), &m_patch_h[m_patch_idx_h[idx]],
        //        (m_patch_idx_h[idx + 1] - m_patch_idx_h[idx]) * sizeof(float));

        poffset_vec[idx] = diff->poffset_bin() - start_pitch;
        toffset_vec[idx] = diff->toffset_bin() - start_tick;
        if (poffset_vec[idx]<poffset_limit[0]) poffset_limit[0] = poffset_vec[idx];
        if (poffset_vec[idx]>poffset_limit[1]) poffset_limit[1] = poffset_vec[idx];
        if (toffset_vec[idx]<toffset_limit[0]) toffset_limit[0] = toffset_vec[idx];
        if (toffset_vec[idx]>toffset_limit[1]) toffset_limit[1] = toffset_vec[idx];

        const int np = patch.rows();
        const int nt = patch.cols();
        if (np < min_np) min_np = np;
        if (nt < min_nt) min_nt = nt;
    }

    Kokkos::View<int*, Kokkos::HostSpace> poffset_k_h(&poffset_vec[0], npatch);
    Kokkos::View<int*, Kokkos::HostSpace> toffset_k_h(&toffset_vec[0], npatch);
    // {//debug yuhw
    //     std::cout << "yuhw: debug\n";
    //     for (int i=0;i<npatch;++i) {
    //         std::cout << poffset_k_h(i) << ", " << toffset_k_h(i) << std::endl;
    //     }
    // }

    Kokkos::View<int*> poffset_k(Kokkos::ViewAllocateWithoutInitializing("poffset_k"), npatch);
    Kokkos::View<int*> toffset_k(Kokkos::ViewAllocateWithoutInitializing("toffset_k"), npatch);

    Kokkos::deep_copy(poffset_k, poffset_k_h);
    Kokkos::deep_copy(toffset_k, toffset_k_h);

    // std::cout << "yuhw: poffset_limit: " << poffset_limit[0] << ", "<< poffset_limit[1] << std::endl;
    // std::cout << "yuhw: toffset_limit: " << toffset_limit[0] << ", "<< toffset_limit[1] << std::endl;
    // std::cout << "yuhw: out.extent: " << out.extent(0) << ", "<< out.extent(1) << std::endl;
    // std::cout << "yuhw: " << npatch << ", "<< min_np << ", " << min_nt << std::endl;
    Kokkos::parallel_for(
        "scatter_add",
        Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Left>>({0, 0, 0}, {npatch, min_np, min_nt}),
        KOKKOS_LAMBDA(const int& n, const int& i, const int& j) {
            auto p = poffset_k(n);
            auto t = toffset_k(n);
            auto idx = patch_idx(n)+i*min_nt+j;
            float v = patch_d(idx);
            Kokkos::atomic_add(&out(p+i, t+j), v);
        });
    // std::cout << "yuhw: box_of_one: " << KokkosArray::dump_2d_view(out,20) << std::endl;
    wend = omp_get_wtime();
    g_get_charge_vec_time_part3 += wend - wstart;
    cout << "get_charge_matrix_kokkos(): get_charge_matrix_kokkos() part3 running time : " << g_get_charge_vec_time_part3 << endl;
    cout << "get_charge_matrix_kokkos(): set_sampling() running time : " << g_get_charge_vec_time_part4
         << ", counter : " << counter << endl;
    cout << "get_charge_matrix_kokkos() : m_fluctuate : " << m_fluctuate << endl;

#ifdef HAVE_CUDA_INC
    cout << "get_charge_matrix_kokkos() CUDA : set_sampling() part1 time : " << g_set_sampling_part1
         << ", part2 (CUDA) time : " << g_set_sampling_part2 << endl;
    cout << "GaussianDiffusion::sampling_CUDA() part3 time : " << g_set_sampling_part3
         << ", part4 time : " << g_set_sampling_part4 << ", part5 time : " << g_set_sampling_part5 << endl;
    cout << "GaussianDiffusion::sampling_CUDA() : g_total_sample_size : " << g_total_sample_size << endl;
#else
    cout << "set_sampling(): part1 time : " << g_set_sampling_part1
         << ", part2 time : " << g_set_sampling_part2 << ", part3 time : " << g_set_sampling_part3 << endl;
#endif
}

void GenKokkos::BinnedDiffusion_transform::get_charge_matrix(std::vector<Eigen::SparseMatrix<float>* >& vec_spmatrix, std::vector<int>& vec_impact){
  const auto ib = m_pimpos.impact_binning();

  // map between reduced impact # to array # 
  std::map<int,int> map_redimp_vec;
  for (size_t i =0; i!= vec_impact.size(); i++){
    map_redimp_vec[vec_impact[i]] = int(i);
  }

  const auto rb = m_pimpos.region_binning();
  // map between impact # to channel #
  std::map<int, int> map_imp_ch;
  // map between impact # to reduced impact # 
  std::map<int, int> map_imp_redimp;

  //std::cout << ib.nbins() << " " << rb.nbins() << std::endl;
  for (int wireind=0;wireind!=rb.nbins();wireind++){
    int wire_imp_no = m_pimpos.wire_impact(wireind);
    std::pair<int,int> imps_range = m_pimpos.wire_impacts(wireind);
    for (int imp_no = imps_range.first; imp_no != imps_range.second; imp_no ++){
      map_imp_ch[imp_no] = wireind;
      map_imp_redimp[imp_no] = imp_no - wire_imp_no;
      
      //  std::cout << imp_no << " " << wireind << " " << wire_imp_no << " " << ib.center(imp_no) << " " << rb.center(wireind) << " " <<  ib.center(imp_no) - rb.center(wireind) << std::endl;
      // std::cout << imp_no << " " << map_imp_ch[imp_no] << " " << map_imp_redimp[imp_no] << std::endl;
    }
  }
  
  int min_imp = 0;
  int max_imp = ib.nbins();


   for (auto diff : m_diffs){
    //    std::cout << diff->depo()->time() << std::endl
    //diff->set_sampling(m_tbins, ib, m_nsigma, 0, m_calcstrat);
    diff->set_sampling(m_tbins, ib, m_nsigma, m_fluctuate, m_calcstrat);
    //counter ++;
    
    const auto patch = diff->patch();
    const auto qweight = diff->weights();

    const int poffset_bin = diff->poffset_bin();
    const int toffset_bin = diff->toffset_bin();

    const int np = patch.rows();
    const int nt = patch.cols();
    
    for (int pbin = 0; pbin != np; pbin++){
      int abs_pbin = pbin + poffset_bin;
      if (abs_pbin < min_imp || abs_pbin >= max_imp) continue;
      double weight = qweight[pbin];

      for (int tbin = 0; tbin!= nt; tbin++){
	int abs_tbin = tbin + toffset_bin;
	double charge = patch(pbin, tbin);

	// std::cout << map_redimp_vec[map_imp_redimp[abs_pbin] ] << " " << map_redimp_vec[map_imp_redimp[abs_pbin]+1] << " " << abs_tbin << " " << map_imp_ch[abs_pbin] << std::endl;
	
	vec_spmatrix.at(map_redimp_vec[map_imp_redimp[abs_pbin] ])->coeffRef(abs_tbin,map_imp_ch[abs_pbin]) += charge * weight; 
	vec_spmatrix.at(map_redimp_vec[map_imp_redimp[abs_pbin]+1])->coeffRef(abs_tbin,map_imp_ch[abs_pbin]) += charge*(1-weight);
	
	// if (map_tuple_pos.find(std::make_tuple(map_redimp_vec[map_imp_redimp[abs_pbin]],map_imp_ch[abs_pbin],abs_tbin))==map_tuple_pos.end()){
	//   map_tuple_pos[std::make_tuple(map_redimp_vec[map_imp_redimp[abs_pbin]],map_imp_ch[abs_pbin],abs_tbin)] = vec_vec_charge.at(map_redimp_vec[map_imp_redimp[abs_pbin] ]).size();
	//   vec_vec_charge.at(map_redimp_vec[map_imp_redimp[abs_pbin] ]).push_back(std::make_tuple(map_imp_ch[abs_pbin],abs_tbin,charge*weight));
	// }else{
	//   std::get<2>(vec_vec_charge.at(map_redimp_vec[map_imp_redimp[abs_pbin] ]).at(map_tuple_pos[std::make_tuple(map_redimp_vec[map_imp_redimp[abs_pbin]],map_imp_ch[abs_pbin],abs_tbin)])) += charge * weight;
	// }
	
	// if (map_tuple_pos.find(std::make_tuple(map_redimp_vec[map_imp_redimp[abs_pbin]+1],map_imp_ch[abs_pbin],abs_tbin))==map_tuple_pos.end()){
	//   map_tuple_pos[std::make_tuple(map_redimp_vec[map_imp_redimp[abs_pbin]+1],map_imp_ch[abs_pbin],abs_tbin)] = vec_vec_charge.at(map_redimp_vec[map_imp_redimp[abs_pbin]+1]).size();
	//   vec_vec_charge.at(map_redimp_vec[map_imp_redimp[abs_pbin]+1]).push_back(std::make_tuple(map_imp_ch[abs_pbin],abs_tbin,charge*(1-weight)));
	// }else{
	//   std::get<2>(vec_vec_charge.at(map_redimp_vec[map_imp_redimp[abs_pbin]+1]).at(map_tuple_pos[std::make_tuple(map_redimp_vec[map_imp_redimp[abs_pbin]+1],map_imp_ch[abs_pbin],abs_tbin)]) ) += charge*(1-weight);
	// }
	
	
      }
    }

    

    
    diff->clear_sampling();
    // need to figure out wire #, time #, charge, and weight ...
   }

   for (auto it = vec_spmatrix.begin(); it!=vec_spmatrix.end(); it++){
     (*it)->makeCompressed();
   }
   
   
  
}

// a new function to generate the result for the entire frame ... 
void GenKokkos::BinnedDiffusion_transform::get_charge_vec(std::vector<std::vector<std::tuple<int,int, double> > >& vec_vec_charge, std::vector<int>& vec_impact){

  double wstart, wend, wstart2, wend2;

  wstart = omp_get_wtime();
  const auto ib = m_pimpos.impact_binning();

  // map between reduced impact # to array # 

  std::map<int,int> map_redimp_vec;
  std::vector<std::unordered_map<long int, int> > vec_map_pair_pos;
  for (size_t i =0; i!= vec_impact.size(); i++){
    map_redimp_vec[vec_impact[i]] = int(i);
    std::unordered_map<long int, int> map_pair_pos;
    vec_map_pair_pos.push_back(map_pair_pos);
  }
  wend = omp_get_wtime();
  g_get_charge_vec_time_part1 += wend - wstart;
  cout << "get_charge_vec() : get_charge_vec() part1 running time : " << g_get_charge_vec_time_part1 << endl;


  
  wstart = omp_get_wtime();
  const auto rb = m_pimpos.region_binning();
  // map between impact # to channel #
  std::map<int, int> map_imp_ch;
  // map between impact # to reduced impact # 
  std::map<int, int> map_imp_redimp;


  for (int wireind=0;wireind!=rb.nbins();wireind++){
    int wire_imp_no = m_pimpos.wire_impact(wireind);
    std::pair<int,int> imps_range = m_pimpos.wire_impacts(wireind);
    for (int imp_no = imps_range.first; imp_no != imps_range.second; imp_no ++){
      map_imp_ch[imp_no] = wireind;
      map_imp_redimp[imp_no] = imp_no - wire_imp_no;
    }
  }

  
  int min_imp = 0;
  int max_imp = ib.nbins();
  int counter = 0;

  wend = omp_get_wtime();
  g_get_charge_vec_time_part2 += wend - wstart;
  cout << "get_charge_vec() : get_charge_vec() part2 running time : " << g_get_charge_vec_time_part2 << endl;
  



  wstart = omp_get_wtime();
  m_t_idx_h[0]=0 ;
  m_p_idx_h[0]=0 ;
  m_patch_idx_h[0]=0 ;
  int max_patch_size = 0 ;
  int patch_size =0 ;
  for (auto diff : m_diffs){
    if(diff->depo()->charge()==0) continue;
    wstart2 = omp_get_wtime();
    #ifdef HAVE_CUDA_INC
    diff->set_sampling_CUDA(m_pvec_D, m_tvec_D, m_patch_D, m_rand_D, &m_Gen, m_tbins, ib, m_nsigma, m_fluctuate, m_calcstrat);
    #else
    //diff->set_sampling(m_pvec, m_tvec, m_patch, m_normals, m_tbins, ib, m_nsigma, m_fluctuate, m_calcstrat);
    //diff->set_sampling(m_patch, m_normals, m_ptvecs, (double*)m_ptvecs_h, m_tbins, ib, m_nsigma, m_fluctuate, m_calcstrat);
    diff->set_sampling_pre(counter,patch_size,m_pvecs_h,m_tvecs_h,m_charges_h,m_p_idx_h,m_t_idx_h,m_patch_idx_h, m_tbins, ib, m_nsigma, m_fluctuate, m_calcstrat);
    //diff->set_sampling(m_tbins, ib, m_nsigma, m_fluctuate, m_calcstrat);
    #endif
    wend2 = omp_get_wtime();
    g_get_charge_vec_time_part4 += wend2 - wstart2;
    counter ++;
    max_patch_size = max_patch_size > patch_size ? max_patch_size : patch_size ;
  }
  wend = omp_get_wtime();
  cout << "get_charge_vec() : get_charge_vec() set_sampling_pre() time " << wend- wstart<< endl;

  set_sampling_bat( counter, max_patch_size, true) ;
  wstart = omp_get_wtime();
  cout << "get_charge_vec() : get_charge_vec() set_sampling_bat() time " << wstart-wend<< endl;


  int idx=0 ;
  for (auto diff : m_diffs){
 
     
    auto patch = diff->get_patch();
    const auto qweight = diff->weights();

    memcpy(&(patch.data()[0]), &m_patch_h[m_patch_idx_h[idx]], (m_patch_idx_h[idx+1]-m_patch_idx_h[idx]) * sizeof(float));
    idx++ ;

    const int poffset_bin = diff->poffset_bin();
    const int toffset_bin = diff->toffset_bin();

    const int np = patch.rows();
    const int nt = patch.cols();

    
    for (int pbin = 0; pbin != np; pbin++){
      int abs_pbin = pbin + poffset_bin;
      if (abs_pbin < min_imp || abs_pbin >= max_imp) continue;
      double weight = qweight[pbin];
      auto const channel = map_imp_ch[abs_pbin];
      auto const redimp = map_imp_redimp[abs_pbin];
      auto const array_num_redimp = map_redimp_vec[redimp];
      auto const next_array_num_redimp = map_redimp_vec[redimp+1];

      auto& map_pair_pos = vec_map_pair_pos.at(array_num_redimp);
      auto& next_map_pair_pos = vec_map_pair_pos.at(next_array_num_redimp);

      auto& vec_charge = vec_vec_charge.at(array_num_redimp);
      auto& next_vec_charge = vec_vec_charge.at(next_array_num_redimp);

      for (int tbin = 0; tbin!= nt; tbin++){
        int abs_tbin = tbin + toffset_bin;
        double charge = patch(pbin, tbin);

        long int index1 = channel*100000 + abs_tbin;
        auto it = map_pair_pos.find(index1);
        if (it == map_pair_pos.end()){
          map_pair_pos[index1] = vec_charge.size();
          vec_charge.emplace_back(channel, abs_tbin, charge*weight);
	}else{
          std::get<2>(vec_charge.at(it->second)) += charge * weight;
	}

        auto it1 = next_map_pair_pos.find(index1);
        if (it1 == next_map_pair_pos.end()){
          next_map_pair_pos[index1] = next_vec_charge.size();
          next_vec_charge.emplace_back(channel, abs_tbin, charge*(1-weight));
	}else{
          std::get<2>(next_vec_charge.at(it1->second)) += charge*(1-weight);
	}
	
      }
    }

    if (counter % 5000==0){
      for (auto it = vec_map_pair_pos.begin(); it != vec_map_pair_pos.end(); it++){
	it->clear();
      }
    }

    diff->clear_sampling();
  }
  wend = omp_get_wtime();
  g_get_charge_vec_time_part3 += wend - wstart;
  cout << "get_charge_vec() : get_charge_vec() part3 running time : " << g_get_charge_vec_time_part3 << endl;
  cout << "get_charge_vec() : set_sampling() running time : " << g_get_charge_vec_time_part4 << ", counter : " << counter << endl;
  cout << "get_charge_vec() : m_fluctuate : " << m_fluctuate << endl;

#ifdef HAVE_CUDA_INC
  cout << "get_charge_vec() CUDA : set_sampling() part1 time : " << g_set_sampling_part1 << ", part2 (CUDA) time : " << g_set_sampling_part2 << endl;
  cout << "GaussianDiffusion::sampling_CUDA() part3 time : " << g_set_sampling_part3 << ", part4 time : " << g_set_sampling_part4 << ", part5 time : " << g_set_sampling_part5 << endl;
  cout << "GaussianDiffusion::sampling_CUDA() : g_total_sample_size : " << g_total_sample_size << endl;
#else
  cout << "get_charge_vec() : set_sampling() part1 time : " << g_set_sampling_part1 << ", part2 time : " << g_set_sampling_part2 << ", part3 time : " << g_set_sampling_part3 << endl;
#endif
}

std::tuple< Kokkos::View<float*>, Kokkos::View<unsigned long*> >
GenKokkos::BinnedDiffusion_transform::set_sampling_bat(unsigned long npatches, int max_patch_size, const bool copy_DtoH) {

  //create hostview from pointers
  Kokkos::View<double*, Kokkos::HostSpace> pvecs_v_h(m_pvecs_h,m_p_idx_h[npatches]);
  Kokkos::View<double*, Kokkos::HostSpace> tvecs_v_h(m_tvecs_h,m_t_idx_h[npatches]);
  Kokkos::View<double*, Kokkos::HostSpace> charges_v_h(m_charges_h,npatches);
  Kokkos::View<unsigned long*, Kokkos::HostSpace> p_idx_v_h(&m_p_idx_h[0],npatches+1) ;
  Kokkos::View<unsigned long*, Kokkos::HostSpace> t_idx_v_h(&m_t_idx_h[0],npatches+1) ;
  Kokkos::View<unsigned long*, Kokkos::HostSpace> patch_idx_v_h(&m_patch_idx_h[0],npatches+1) ;
  Kokkos::View<float* , Kokkos::HostSpace> patches_v_h(&m_patch_h[0], m_patch_idx_h[npatches] ) ;

  Kokkos::View<double*> sum_p_v(Kokkos::ViewAllocateWithoutInitializing("PatchSum"), npatches) ;

  //Device Views
  Kokkos::View<unsigned long * > p_idx(Kokkos::ViewAllocateWithoutInitializing("P_idx") , npatches+1) ;
  Kokkos::View<unsigned long * > t_idx(Kokkos::ViewAllocateWithoutInitializing("T_idx") , npatches+1) ;
  Kokkos::View<unsigned long * > patch_idx(Kokkos::ViewAllocateWithoutInitializing("Pat_idx") , npatches+1) ;
  Kokkos::View<float * > patch_d(Kokkos::ViewAllocateWithoutInitializing("Patches") , m_patch_idx_h[npatches]) ;
  Kokkos::View<double * > pvecs_d(Kokkos::ViewAllocateWithoutInitializing("Pvecs") , m_p_idx_h[npatches]) ;
  Kokkos::View<double * > tvecs_d(Kokkos::ViewAllocateWithoutInitializing("Tvecs") , m_t_idx_h[npatches]) ;
  Kokkos::View<double * > charges_d(Kokkos::ViewAllocateWithoutInitializing("Charges") , npatches) ;

  //existing view, make a subview
  auto normals = Kokkos::subview(m_normals,std::make_pair((size_t)0, (size_t)m_patch_idx_h[npatches] ) ) ;

  //Copy of views 
  Kokkos::deep_copy(p_idx, p_idx_v_h) ;
  Kokkos::deep_copy(t_idx, t_idx_v_h) ;
  Kokkos::deep_copy(patch_idx, patch_idx_v_h) ;
  Kokkos::deep_copy(pvecs_d, pvecs_v_h) ;
  Kokkos::deep_copy(tvecs_d, tvecs_v_h) ;
  Kokkos::deep_copy(charges_d, charges_v_h) ;

  bool fl = false ;
  if( m_fluctuate) fl = true    ;

    std::cout<<"Max Patch size : " << max_patch_size <<std::endl ;

  bool is_host= std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value ;

  //is_host =  ;
  //kernel
    Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(npatches,Kokkos::AUTO) ;
  if(is_host) {
    Kokkos::parallel_for( policy,
      KOKKOS_LAMBDA( const Kokkos::TeamPolicy<>::member_type & team ){
       int ip=team.league_rank() ;

       int np=p_idx(ip+1)-p_idx(ip) ;
       int nt=t_idx(ip+1)-t_idx(ip) ;
       int patch_size=np*nt ;
       unsigned long p0 =patch_idx(ip) ;

       double sum = 0.0 ;
       Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, patch_size) ,
         [=] (int & ii, double & lsum ) {
            double v = pvecs_d(p_idx(ip)+ ii%np)*tvecs_d(t_idx(ip)+ ii/np) ;
            patch_d(ii+p0) = (float) v ;
            lsum += v ;
         }, sum ) ;


       double charge=charges_d(ip) ;
       double charge_abs = abs(charge) ;
//       int charge_sign = charge < 0  ? -1 :1 ;

       Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, patch_size) ,
         [=] ( int & ii ) {
           patch_d(ii+p0) *= float(charge_abs/sum) ;
         } ) ;

      if( fl ){
         int n=(int) charge_abs;
         sum =0.0  ;

         Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, patch_size) ,
           [=] (int & ii, double & lsum ) {
               double p =  patch_d(ii+p0)/charge_abs ;
               double q = 1-p ;
               double mu = n*p ;
               double sigma = sqrt(p*q*n) ;
               p = normals(ii+p0)*sigma + mu ;
               lsum += p ;
               }, sum ) ;

         Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, patch_size) ,
           [=] ( int & ii ) {
             patch_d(ii+p0) *= (float) charge_abs/sum ;
           } ) ;

       }
    } ) ;

  } else {
    Kokkos::parallel_for( policy, 
      KOKKOS_LAMBDA( const Kokkos::TeamPolicy<>::member_type & team ){
       int ip=team.league_rank() ;
      
       int np=p_idx(ip+1)-p_idx(ip) ;
       int nt=t_idx(ip+1)-t_idx(ip) ;
       int patch_size=np*nt ;
       unsigned long p0 =patch_idx(ip) ;

       double sum = 0.0 ;

       Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, patch_size) ,
         [=] (int & ii, double & lsum ) { 
            double v = pvecs_d(p_idx(ip)+ ii%np)*tvecs_d(t_idx(ip)+ ii/np) ;
	    patch_d(ii+p0) = (float) v ;
	    lsum += v ;
         }, sum ) ;


       double charge=charges_d(ip) ;
       double charge_abs = abs(charge) ;
//       int charge_sign = charge < 0  ? -1 :1 ;

       Kokkos::parallel_for(Kokkos::TeamThreadRange(team, patch_size) ,
	 [=] ( int & ii ) {
           patch_d(ii+p0) *= float(charge_abs/sum) ;
	 } ) ;
       
       if( fl ){
	 int n=(int) charge_abs;
         sum =0.0  ;

         Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, patch_size) ,
           [=] (int & ii, double & lsum ) { 
               double p =  patch_d(ii+p0)/charge_abs ;
               double q = 1-p ;
               double mu = n*p ;
               double sigma = sqrt(p*q*n) ;
               p = normals(ii+p0)*sigma + mu ;
	       lsum += p ;
               }, sum ) ;

         Kokkos::parallel_for(Kokkos::TeamThreadRange(team, patch_size) ,
	   [=] ( int & ii ) {
             patch_d(ii+p0) *= (float) charge_abs/sum ;
	   } ) ;

       }
//       if(ip== 0 &&team.team_rank()==0 ) printf("team-size = %d,%dx%d \n", team.team_size() ,np,nt); 

    } ) ;

  }  
  if (copy_DtoH==true) Kokkos::deep_copy(patches_v_h, patch_d );
  return {patch_d, patch_idx};
}
// GenKokkos::ImpactData::pointer GenKokkos::BinnedDiffusion_transform::impact_data(int bin) const
// {
//     const auto ib = m_pimpos.impact_binning();
//     if (! ib.inbounds(bin)) {
//         return nullptr;
//     }

//     auto it = m_impacts.find(bin);
//     if (it == m_impacts.end()) {
// 	return nullptr;
//     }
//     auto idptr = it->second;

//     // make sure all diffusions have been sampled 
//     for (auto diff : idptr->diffusions()) {
//       diff->set_sampling(m_tbins, ib, m_nsigma, m_fluctuate, m_calcstrat);
//       //diff->set_sampling(m_tbins, ib, m_nsigma, 0, m_calcstrat);
//     }

//     idptr->calculate(m_tbins.nbins());
//     return idptr;
// }


static
std::pair<double,double> gausdesc_range(const std::vector<GenKokkos::GausDesc> gds, double nsigma)
{
    int ncount = -1;
    double vmin=0, vmax=0;
    for (auto gd : gds) {
        ++ncount;

        const double lvmin = gd.center - gd.sigma*nsigma;
        const double lvmax = gd.center + gd.sigma*nsigma;
        if (!ncount) {
            vmin = lvmin;
            vmax = lvmax;
            continue;
        }
        vmin = std::min(vmin, lvmin);
        vmax = std::max(vmax, lvmax);
    }        
    return std::make_pair(vmin,vmax);
}

std::pair<double,double> GenKokkos::BinnedDiffusion_transform::pitch_range(double nsigma) const
{
    std::vector<GenKokkos::GausDesc> gds;
    for (auto diff : m_diffs) {
        gds.push_back(diff->pitch_desc());
    }
    return gausdesc_range(gds, nsigma);
}

std::pair<int,int> GenKokkos::BinnedDiffusion_transform::impact_bin_range(double nsigma) const
{
    const auto ibins = m_pimpos.impact_binning();
    auto mm = pitch_range(nsigma);
    return std::make_pair(std::max(ibins.bin(mm.first), 0),
                          std::min(ibins.bin(mm.second)+1, ibins.nbins()));
}

std::pair<double,double> GenKokkos::BinnedDiffusion_transform::time_range(double nsigma) const
{
    std::vector<GenKokkos::GausDesc> gds;
    for (auto diff : m_diffs) {
        gds.push_back(diff->time_desc());
    }
    return gausdesc_range(gds, nsigma);
}

std::pair<int,int> GenKokkos::BinnedDiffusion_transform::time_bin_range(double nsigma) const
{
    auto mm = time_range(nsigma);
    return std::make_pair(std::max(m_tbins.bin(mm.first),0),
                          std::min(m_tbins.bin(mm.second)+1, m_tbins.nbins()));
}
