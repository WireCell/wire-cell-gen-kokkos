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


#define MAX_PATCH_SIZE 512 
#define P_BLOCK_SIZE  512
#define MAX_PATCHES  50000
#define MAX_NPSS_DEVICE 1000
#define MAX_NTSS_DEVICE 1000
#define FULL_MASK 0xffffffff
#define RANDOM_BLOCK_SIZE (1024*1024)
#define RANDOM_BLOCK_NUM 512
//#define MAX_RANDOM_LENGTH (RANDOM_BLOCK_NUM*RANDOM_BLOCK_SIZE)
#define MAX_RANDOM_LENGTH (MAX_PATCH_SIZE*MAX_PATCHES)
#define PI 3.14159265358979323846

#define MAX_P_SIZE 50 
#define MAX_T_SIZE 50 



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
        range_max1 = 0xffffffffffffffffULL-1;
        range_max2 = range_max1;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(int i) const {
        //
        typename GeneratorPool::generator_type rand_gen1 = rand_pool1.get_state();//        typename GeneratorPool::generator_type rand_gen2 = rand_pool2.get_state();

        for (int k = 0; k < samples/2; k++) {
            double u1 = (double) rand_gen1.urand64(range_min, range_max1) / range_max1;
            double u2 = (double) rand_gen1.urand64(range_min, range_max2) / range_max2;
            normals(i * samples + 2*k)     = sqrt(-2*log(u1)) * cos(2*PI*u2);
            normals(i * samples + 2*k + 1) = sqrt(-2*log(u1)) * sin(2*PI*u2);
        }

        rand_pool1.free_state(rand_gen1);
        //rand_pool2.free_state(rand_gen2);
        //
    }
   
};




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
    //init_Device();
}


GenKokkos::BinnedDiffusion_transform::~BinnedDiffusion_transform() {
    clear_Device();
}


/*
void GenKokkos::BinnedDiffusion_transform::init_Device() {


    size_t size = MAX_PATCHES;
    size_t samples = MAX_PATCH_SIZE;
    int seed = 2020;

    Kokkos::Random_XorShift64_Pool<> rand_pool1(seed);
    Kokkos::Random_XorShift64_Pool<> rand_pool2(seed+1);
    Kokkos::resize(m_normals, size * samples);

    // TODO: assert() if size*samples % 256 != 0
    Kokkos::parallel_for(size*samples/256, generate_random<Kokkos::Random_XorShift64_Pool<> >(m_normals, rand_pool1, rand_pool2, 256));
}

*/

void GenKokkos::BinnedDiffusion_transform::clear_Device() {

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
    Kokkos::Tools::pushRegion("getchargeMaxtrix");
    Kokkos::Tools::pushRegion("partA") ;
    std::cout << "yuhw: get_charge_matrix_kokkos\n";

    double wstart, wend ;
    wstart = omp_get_wtime();
    const auto ib = m_pimpos.impact_binning();

    // map between reduced impact # to array #
/*
    std::map<int, int> map_redimp_vec;
    std::vector<std::unordered_map<long int, int> > vec_map_pair_pos;
    for (size_t i = 0; i != vec_impact.size(); i++) {
        map_redimp_vec[vec_impact[i]] = int(i);
        std::unordered_map<long int, int> map_pair_pos;
        vec_map_pair_pos.push_back(map_pair_pos);
    }
    wend = omp_get_wtime();
    g_get_charge_vec_time_part1 = wend - wstart;
    cout << "get_charge_matrix_kokkos(): part1 running time : " << g_get_charge_vec_time_part1 << endl;

    wstart = omp_get_wtime();
    const auto rb = m_pimpos.region_binning();
    // map between impact # to channel #
    std::map<int, int> map_imp_ch;
    // map between impact # to reduced impact #
    std::map<int, int> map_imp_redimp;

    // std::cout << "yuhw: " << rb.nbins() << std::endl;
    for (int wireind = 0; wireind != rb.nbins(); wireind++) {
        int wire_imp_no = m_pimpos.wire_impact(wireind);
        std::pair<int, int> imps_range = m_pimpos.wire_impacts(wireind);
        for (int imp_no = imps_range.first; imp_no != imps_range.second; imp_no++) {
            map_imp_ch[imp_no] = wireind;
            map_imp_redimp[imp_no] = imp_no - wire_imp_no;
        }
    }

    //int min_imp = 0;
    int max_imp = ib.nbins();
    int counter = 0;

    wend = omp_get_wtime();
    g_get_charge_vec_time_part2 = wend - wstart;
    cout << "get_charge_matrix_kokkos(): part2 running time : " << g_get_charge_vec_time_part2 << endl;
*/
 //   wstart = omp_get_wtime();
    Kokkos::Tools::popRegion() ;
    // set the size of gd view and create host view
    int npatches = m_diffs.size();
    // typedef Kokkos::View<GenKokkos::GdData *  > gd_vt ;
    Kokkos::Tools::pushRegion("partB") ;
    Kokkos::Tools::pushRegion("host and device allocGdate") ;
    gd_vt gdata(Kokkos::ViewAllocateWithoutInitializing("Gdata"), npatches);
    auto gdata_h = Kokkos::create_mirror_view(gdata);

    Kokkos::Tools::popRegion() ;

    // fill the host view data from diffs
    Kokkos::Tools::pushRegion("hostGdcopy") ;
    int ii = 0;
    for (auto diff : m_diffs) {
        gdata_h(ii).p_ct = diff->pitch_desc().center;
        gdata_h(ii).t_ct = diff->time_desc().center;
        gdata_h(ii).charge = diff->depo()->charge();
        gdata_h(ii).t_sigma = diff->time_desc().sigma;
        gdata_h(ii).p_sigma = diff->pitch_desc().sigma;
        //    if(diff->pitch_desc().sigma == 0 || diff->time_desc().sigma == 0  ) std::cout<<"sigma-0 patch: " <<ii <<
         //   std::endl ;
	//std::cout<<"Gdata: " << ii<<" "<<gdata_h(ii).p_ct<<" "<<gdata_h(ii).t_ct<<" "<<gdata_h(ii).charge<<" "<<gdata_h(ii).t_sigma<<" "<<gdata_h(ii).p_sigma <<std::endl ;
        ii++;
    }
    Kokkos::Tools::popRegion() ;

    // copy to device
    Kokkos::deep_copy(gdata, gdata_h);

    Kokkos::Tools::popRegion() ;
    // make and device friendly Binning  and copy tbin pbin over.
    GenKokkos::DBin tb, pb;
    tb.nbins = m_tbins.nbins();
    tb.minval = m_tbins.min();
    tb.binsize = m_tbins.binsize();
    pb.nbins = ib.nbins();
    pb.minval = ib.min();
    pb.binsize = ib.binsize();

    // perform set_sampling_pre tasks in parallel:
    // should we make it into functions
    // create Device Views
    Kokkos::Tools::pushRegion("createviews") ; 
    size_vt np_d(Kokkos::ViewAllocateWithoutInitializing("P_sizes"), npatches);
    size_vt nt_d(Kokkos::ViewAllocateWithoutInitializing("T_sized"), npatches);
    size_vt offsets_d(Kokkos::ViewAllocateWithoutInitializing("Offset_bins"), npatches * 2);
    idx_vt patch_idx(Kokkos::ViewAllocateWithoutInitializing("Pat_idx"), npatches + 1);
    db_vt pvecs_d(Kokkos::ViewAllocateWithoutInitializing("Pvecs"), npatches * MAX_P_SIZE);
    db_vt tvecs_d(Kokkos::ViewAllocateWithoutInitializing("Tvecs"), npatches * MAX_T_SIZE);
    db_vt qweights_d(Kokkos::ViewAllocateWithoutInitializing("Weights"), npatches * MAX_P_SIZE);
    

    Kokkos::Tools::popRegion() ; 
    // create host view , because it will be needed on host temperarily.
    // patch_v_h is delayed until we know the size (nt np)
    Kokkos::Tools::pushRegion("NTNP+Scan"); 
    //auto nt_v_h = Kokkos::create_mirror_view(nt_d);
   // auto np_v_h = Kokkos::create_mirror_view(np_d);
   // auto patch_idx_v_h = Kokkos::create_mirror_view(patch_idx);
   // auto qweights_v_h = Kokkos::create_mirror_view(qweights_d);
   // auto offsets_v_h = Kokkos::create_mirror_view(offsets_d);

    // Kernel for calculate nt and np and offset_bin s for t and p for each gd
    int nsigma = m_nsigma;
    
    Kokkos::parallel_for("NtNp", npatches, KOKKOS_LAMBDA(int i) {
        double t_s = gdata(i).t_ct - gdata(i).t_sigma * nsigma;
        double t_e = gdata(i).t_ct + gdata(i).t_sigma * nsigma;
        int t_ofb = max(int((t_s - tb.minval) / tb.binsize), 0);
        int ntss = min((int((t_e - tb.minval) / tb.binsize)) + 1, tb.nbins) - t_ofb;

        double p_s = gdata(i).p_ct - gdata(i).p_sigma * nsigma;
        double p_e = gdata(i).p_ct + gdata(i).p_sigma * nsigma;
        int p_ofb = max(int((p_s - pb.minval) / pb.binsize), 0);
        int npss = min((int((p_e - pb.minval) / pb.binsize)) + 1, pb.nbins) - p_ofb;

        nt_d(i) = ntss;
        np_d(i) = npss;
        offsets_d(i) = t_ofb;
        offsets_d(npatches + i) = p_ofb;
    });

    std::cout<<"npatches= "<<npatches <<std::endl ;
    //  kernel calculate index for patch  Can be mergged to previous kernel ?
    unsigned long result;  // total patches points
    Kokkos::parallel_scan("Scan" ,npatches,
                          KOKKOS_LAMBDA(int i, unsigned long& lsum, const bool final) {
                              if (final) {
                                  patch_idx(i) = lsum;
                                  if (i == (npatches - 1)) {
                                      patch_idx(npatches) = lsum + np_d(i) * nt_d(i);
				      //printf("Scan:final: lsum=%lu i=%d np_d=%d nt_d=%d \n", lsum,i , np_d(i), nt_d(i) );
                                  }
                              }
                              //  bool pr =true ;
                              //   if(pr && lsum >18000000 ) {
                              //      printf("xx more i %d np_d(i)=%d, nt_d(i)=%d lsum %lu\n", i , np_d(i) ,
                              //      nt_d(i),lsum ) ;
                              //     pr = false ;
                              //   }
                              lsum += np_d(i) * nt_d(i);
                          },
                          result);

    // debug:
    std::cout << "total patch size: " << result << " WeightStrat: " << m_calcstrat << std::endl;
    Kokkos::Tools::popRegion() ; 
    Kokkos::Tools::pushRegion("patch view create");
    // Allocate space for patches on device
    fl_vt patch_d(Kokkos::ViewAllocateWithoutInitializing("Patches"), result);
    Kokkos::Tools::popRegion() ; 
    // make view on host
    //auto patch_v_h = Kokkos::create_mirror_view(patch_d);
    // make a view pointing to random numbers
    Kokkos::Tools::pushRegion("normal view create");
    int size = (result+255)/256 * 256 ; 
    Kokkos::View <double *> normals(Kokkos::ViewAllocateWithoutInitializing("Normals"), size) ; 
    Kokkos::Tools::popRegion() ; 
    Kokkos::Tools::pushRegion("Random Number create");
    int seed = 2020;
    Kokkos::Random_XorShift64_Pool<> rand_pool1(seed);

    Kokkos::parallel_for(size/256, generate_random<Kokkos::Random_XorShift64_Pool<> >(normals, rand_pool1, rand_pool1, 256));
    //auto normals = Kokkos::subview(m_normals, std::make_pair((size_t) 0, (size_t) result));
    //auto normals = m_normals;
    Kokkos::Tools::popRegion() ;

    // decide weight calculation
    int weightstrat = m_calcstrat;

    // each team resposible for 1 GD , kernel calculate pvecs and tvecs
    //
    Kokkos::Tools::pushRegion("vector caculation");
    bool is_host= std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value ;
    //for AVX2 && GPU
    int patches_per_team = is_host ? 1 : 16 ;
    int team_size = is_host ?  1 : 16 ;
    int vec_len = is_host ? 4 : 8 ;
    Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>((npatches+patches_per_team -1)/patches_per_team, team_size,vec_len );
    //    Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>((npatches+patches_per_team -1)/patches_per_team, Kokkos::AUTO );
    Kokkos::parallel_for("PTvecs", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
       // int ip = team.league_rank()*patches_per_team+ team.team_rank() ;
        //int it = team.team_rank();
      int ip_start = team.league_rank()*patches_per_team;
      int ip_end = (ip_start + patches_per_team) < npatches ? ip_start + patches_per_team : npatches ;
      const double sqrt2 = sqrt(2.0);
      //every thread  for 1 patch.
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,ip_start,ip_end),[&](int ip) {
        double start_t = tb.minval + offsets_d(ip) * tb.binsize;
        double start_p = pb.minval + offsets_d(ip + npatches) * pb.binsize;
        int np = np_d(ip);
        int nt = nt_d(ip);


        // Calculate Pvecs
	
        if (np == 1) {
	   //Kokkos::single(Kokkos::PerThread(team),[&] () { pvecs_d(ip * MAX_P_SIZE) = 1.0; }) ;
	    pvecs_d(ip * MAX_P_SIZE) = 1.0 ;
        } 
        else {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, np), [=](int& ii) {
                double step = pb.binsize;
                double start = start_p;
                double factor = sqrt2 * gdata(ip).p_sigma;
                double x = (start + step * ii - gdata(ip).p_ct) / factor;
                double ef1 = 0.5 * erf(x);
                double ef2 = 0.5 * erf(x + step / factor);
                double val = ef2 - ef1;
                pvecs_d(ip * MAX_P_SIZE + ii) = val;
                //	   if(ip==0 && ii==0 ) printf("pvecs(0)=%f %f %f %f %f %f\n", val,start_p,gdata(0).p_sigma,
                //factor, ef1, ef2);
            });
        }

        // Calculate tvecs
        if (nt == 1) {
		tvecs_d(ip * MAX_T_SIZE) = 1.0;
        }
        else {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, nt), [=](int& ii) {
                double step = tb.binsize;
                double start = start_t;
                double factor = sqrt2 * gdata(ip).t_sigma;
                double x = (start + step * ii - gdata(ip).t_ct) / factor;
                double ef1 = 0.5 * erf(x);
                double ef2 = 0.5 * erf(x + step / factor);
                double val = ef2 - ef1;
                tvecs_d(ip * MAX_T_SIZE + ii) = val;
            });
        }
        // calculate weights
        if (weightstrat == 2) {
            if (gdata(ip).p_sigma == 0) {
		    Kokkos::single(Kokkos::PerThread(team),[&] (){ qweights_d(ip*MAX_P_SIZE) = (start_p + pb.binsize - gdata(ip).p_ct) / pb.binsize;}) ;
            }
            else {
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, np), [=](int& ii) {
                    double rel1 = (start_p + pb.binsize * ii - gdata(ip).p_ct) / gdata(ip).p_sigma;
                    double rel2 = rel1 + pb.binsize / gdata(ip).p_sigma;
                    double gaus1 = exp(-0.5 * rel1 * rel1);
                    double gaus2 = exp(-0.5 * rel2 * rel2);
                    double wt = -1.0 * gdata(ip).p_sigma / pb.binsize * (gaus2 - gaus1) / sqrt(2.0 * PI) /
                                    pvecs_d(ip * MAX_P_SIZE + ii) +
                                (gdata(ip).p_ct - (start_p + (ii + 1) * pb.binsize)) / pb.binsize;
                    qweights_d(ip*MAX_P_SIZE+ii) = -wt;
                });
            }
        }
       });
    });
    Kokkos::Tools::popRegion();

    Kokkos::Tools::pushRegion("Set_sampling_batch fuct") ; 
    set_sampling_bat( npatches, nt_d,  np_d,  patch_idx, pvecs_d , tvecs_d, patch_d, normals,gdata  ) ;
    Kokkos::Tools::popRegion();
    Kokkos::fence() ;
    // std::cout << "patch_d: " << typeid(patch_d).name() << std::endl;
    // std::cout << "patch_idx: " << typeid(patch_idx).name() << std::endl;
    wend = omp_get_wtime();
    g_get_charge_vec_time_part4 += wend-wstart ;
    std::cout <<"set_sampling_Time: "<< wend-wstart <<std::endl;
    
    wstart = omp_get_wtime();
    // cout << "pr21 get_charge_matrix_kokkos(): set_sampling_bat() no DtoH time " << wstart - wend << endl;
    // std::cout << "yuhw: DEBUG: npatches: " << npatches << std::endl;
    // std::cout << "yuhw: DEBUG: np_d: " << KokkosArray::dump_1d_view(np_d,10000) << std::endl;
    // std::cout << "yuhw: DEBUG: nt_d: " << KokkosArray::dump_1d_view(nt_d,10000) << std::endl;
    // std::cout << "yuhw: DEBUG: offsets_d: " << KokkosArray::dump_1d_view(offsets_d,10000) << std::endl;
    // std::cout << "yuhw: DEBUG: patch_idx: " << KokkosArray::dump_1d_view(patch_idx,10000) << std::endl;
    // std::cout << "yuhw: DEBUG: patch_d: " << KokkosArray::dump_1d_view(patch_d,10000) << std::endl;
    // std::cout << "yuhw: DEBUG: qweights_d: " << KokkosArray::dump_1d_view(qweights_d,10000) << std::endl;

    Kokkos::TeamPolicy<> policy_sa = Kokkos::TeamPolicy<>(npatches, Kokkos::AUTO);
    Kokkos::parallel_for("ScatterAdd", policy_sa, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        int ipatch = team.league_rank();
        int np = np_d(ipatch);
        int nt = nt_d(ipatch);
        int p = offsets_d(npatches + ipatch)-start_pitch;
        int t = offsets_d(ipatch)-start_tick;
        int patch_size=np*nt ;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, patch_size),
                             [=] (const int& i) {
                                 auto idx = patch_idx(ipatch) + i;
                                 float charge = patch_d(idx);
                                 double weight = qweights_d(i%np+ipatch*MAX_P_SIZE);
                                 Kokkos::atomic_add(&out(p+i%np, t+i/np), (float)(charge*weight));
                                 Kokkos::atomic_add(&out(p+i%np+1, t+i/np), (float)(charge*(1.-weight)));
                             });
    });
    Kokkos::fence() ;
    // std::cout << "yuhw: box_of_one: " << KokkosArray::dump_2d_view(out,20) << std::endl;
    wend = omp_get_wtime();
    // std::cout << "yuhw: DEBUG: out: " << KokkosArray::dump_2d_view(out,10000) << std::endl;
    g_get_charge_vec_time_part3 += wend - wstart;
    cout<<"ScatterAdd_Time : "<<wend-wstart << endl ; 
    cout << "get_charge_matrix_kokkos(): Total_ScatterAdd_Time : " << g_get_charge_vec_time_part3 << endl;
    cout << "get_charge_matrix_kokkos(): Total_set_sampling_Time : " << g_get_charge_vec_time_part4<< endl ;
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

  double wstart, wend ;

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
  cout << "get_charge_vec() : part1 running time : " << g_get_charge_vec_time_part1 << endl;


  
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
  cout << "get_charge_vec() : part2 running time : " << g_get_charge_vec_time_part2 << endl;
  

  wstart = omp_get_wtime();

  //set the size of gd view and create host view
  int npatches = m_diffs.size() ;
  //typedef Kokkos::View<GenKokkos::GdData *  > gd_vt ;
  gd_vt gdata(Kokkos::ViewAllocateWithoutInitializing("Gdata"), npatches) ;
  auto gdata_h = Kokkos::create_mirror_view(gdata) ;
  
  //fill the host view data from diffs
  int ii=0 ;
  for (auto diff : m_diffs) {
    gdata_h(ii).p_ct=diff->pitch_desc().center ;
    gdata_h(ii).t_ct=diff->time_desc().center ;
    gdata_h(ii).charge=diff->depo()->charge();    
    gdata_h(ii).t_sigma=diff->time_desc().sigma ;
    gdata_h(ii).p_sigma=diff->pitch_desc().sigma ;
//    if(diff->pitch_desc().sigma == 0 || diff->time_desc().sigma == 0  ) std::cout<<"sigma-0 patch: " <<ii << std::endl ; 
    if( gdata_h(ii).charge == 0   ) std::cout<<" 0Charge: " <<ii << std::endl ; 
    ii++ ;
  }

  // copy to device
 Kokkos::deep_copy(gdata, gdata_h) ; 
  
 // make and device friendly Binning  and copy tbin pbin over.
 GenKokkos::DBin  tb ,pb ;
 tb.nbins=m_tbins.nbins() ;
 tb.minval=m_tbins.min() ; 
 tb.binsize=m_tbins.binsize() ; 
 pb.nbins=ib.nbins() ;
 pb.minval=ib.min() ; 
 pb.binsize=ib.binsize() ; 

// perform set_sampling_pre tasks in parallel:
// should we make it into functions
  // create Device Views
  //
  //typedef Kokkos::View<unsigned int * > size_vt ;
 // typedef Kokkos::View<unsigned long * > idx_vt ;
 // typedef Kokkos::View<double * > db_vt ;
 //  typedef Kokkos::View<float *> fl_vt ;

  size_vt np_d(Kokkos::ViewAllocateWithoutInitializing("P_sizes") , npatches) ;
  size_vt nt_d(Kokkos::ViewAllocateWithoutInitializing("T_sized") , npatches) ;
  size_vt offsets_d(Kokkos::ViewAllocateWithoutInitializing("Offset_bins") , npatches*2) ;
  idx_vt  patch_idx(Kokkos::ViewAllocateWithoutInitializing("Pat_idx") , npatches+1) ;
  db_vt pvecs_d(Kokkos::ViewAllocateWithoutInitializing("Pvecs") , npatches*MAX_P_SIZE) ;
  db_vt tvecs_d(Kokkos::ViewAllocateWithoutInitializing("Tvecs") , npatches*MAX_T_SIZE) ;
  db_vt qweights_d(Kokkos::ViewAllocateWithoutInitializing("Weights") , npatches*MAX_P_SIZE) ;

  //create host view , because it will be needed on host temperarily.
  //patch_v_h is delayed until we know the size (nt np) 
  auto nt_v_h = Kokkos::create_mirror_view(nt_d) ;
  auto np_v_h = Kokkos::create_mirror_view(np_d) ;
  auto patch_idx_v_h = Kokkos::create_mirror_view(patch_idx) ;
  auto qweights_v_h = Kokkos::create_mirror_view(qweights_d) ;
  auto offsets_v_h = Kokkos::create_mirror_view(offsets_d) ;
 


  // Kernel for calculate nt and np and offset_bin s for t and p for each gd
  int nsigma=m_nsigma ;

  Kokkos::parallel_for( npatches,
    KOKKOS_LAMBDA( int i  ){
     double t_s = gdata(i).t_ct - gdata(i).t_sigma*nsigma ;
     double t_e = gdata(i).t_ct + gdata(i).t_sigma*nsigma ;
     int  t_ofb = max(int((t_s-tb.minval)/tb.binsize), 0) ;
     int ntss = min((int((t_e-tb.minval)/tb.binsize))+1, tb.nbins)-t_ofb ;

     double p_s = gdata(i).p_ct - gdata(i).p_sigma*nsigma ;
     double p_e = gdata(i).p_ct + gdata(i).p_sigma*nsigma ;
     int p_ofb = max(int((p_s-pb.minval)/pb.binsize), 0) ;
     int npss  = min((int((p_e-pb.minval)/pb.binsize))+1, pb.nbins)- p_ofb ;

     nt_d(i) = ntss ;
     np_d(i) = npss ;
     offsets_d(i) = t_ofb ;
     offsets_d(npatches + i) = p_ofb ;
   //  if( i== 14233  ) {
     //printf("i: %d nsigma: %d p_s: %f p_e: %f p_ct: %f p_sigma: %f pb.minval: %f pb.nbins: %d pb.binsize: %f p_ofb %f, np %d, %f \n",
	//	     i , nsigma, p_s, p_e,   gdata(i).p_ct, gdata(i).p_sigma,  pb.minval,pb.nbins,pb.binsize,p_ofb,npss,(p_s-pb.minval)/pb.binsize ) ; 
//	printf( "i %d p_s: %e pb.min: %e ,%e np=%d \n", i , p_s, pb.minval,(p_s-pb.minval)/pb.binsize, npss) ;
  //   }
   //  if(npss < 2 ) printf("i nt,np= %d , %d, %d \n", i, nt_d(i),np_d(i)) ;
    // if(ntss*npss < 2 ) printf("point  patch %d , %d ,%d \n" , i, nt_d(i) , np_d(i) );
   } ) ;

//  kernel calculate index for patch  Can be mergged to previous kernel ?
// 

  unsigned long result ; // total patches points 
  Kokkos::parallel_scan( npatches,
    KOKKOS_LAMBDA( int i , unsigned long & lsum , const bool final ){
      if (final) {
         patch_idx(i) = lsum ;
         if( i == (npatches -1)) {
	 patch_idx(npatches)= lsum + np_d(i) * nt_d(i) ;
//	 printf("np_d(i)=%d, nt_d(i)=%d lsum=%lu\n", np_d(i) , nt_d(i),lsum ) ; 
	 }
      }
    //  bool pr =true ;
   //   if(pr && lsum >18000000 ) {
   //      printf("xx more i %d np_d(i)=%d, nt_d(i)=%d lsum %lu\n", i , np_d(i) , nt_d(i),lsum ) ;
    //     pr = false ; 
   //   }	 
      lsum += np_d(i)*nt_d(i) ;
    },result ) ;

  //debug:
  std::cout<<"total patch size: "<<result <<" WeightStrat: "<< m_calcstrat << std::endl ;

  // Allocate space for patches on device
  fl_vt patch_d(Kokkos::ViewAllocateWithoutInitializing("Patches") , result) ;
  // make view on host
  auto patch_v_h = Kokkos::create_mirror_view(patch_d) ;
   
  // make a view pointing to random numbers
  //to normals = Kokkos::subview(m_normals,std::make_pair((size_t)0, (size_t)result ) ) ;
  Kokkos::Tools::pushRegion("normal view create");
  int size = (result+255)/256 * 256 ; 
  Kokkos::View <double *> normals(Kokkos::ViewAllocateWithoutInitializing("Normals"), size) ; 
  //Kokkos::resize(m_normals, size );
  Kokkos::Tools::popRegion() ; 
  Kokkos::Tools::pushRegion("Random Number create");
  Kokkos::Timer kt0;
  int seed = 2020;
  Kokkos::Random_XorShift64_Pool<> rand_pool1(seed);

  Kokkos::parallel_for(size/256, generate_random<Kokkos::Random_XorShift64_Pool<> >(normals, rand_pool1, rand_pool1, 256));
  //auto normals = Kokkos::subview(m_normals, std::make_pair((size_t) 0, (size_t) result));
  Kokkos::Tools::popRegion() ;
  std::cout<<"Random number_Time : "<<kt0.seconds() <<std::endl ;
  //decide weight calculation
  int weightstrat=m_calcstrat ;

  // each team resposible for 1 GD , kernel calculate pvecs and tvecs
  Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(npatches,Kokkos::AUTO) ;
  Kokkos::parallel_for( policy,
    KOKKOS_LAMBDA( const Kokkos::TeamPolicy<>::member_type & team ){
      int ip = team.league_rank() ;
      int it = team.team_rank() ;
      double start_t = tb.minval + offsets_d(ip) * tb.binsize ;
      double start_p = pb.minval + offsets_d(ip+npatches) * pb.binsize ;
      int np=np_d(ip) ;
      int nt=nt_d(ip) ;

      const double sqrt2 = sqrt(2.0) ;

      //Calculate Pvecs
      if(np == 1) {
         if( it == 0) pvecs_d(ip * MAX_P_SIZE) =1.0 ;
      } else {
       Kokkos::parallel_for(Kokkos::TeamThreadRange(team, np) ,
         [=] (int & ii ) {
           double step = pb.binsize ;
           double start = start_p ;
           double factor = sqrt2 * gdata(ip).p_sigma  ;
           double x = (start + step * ii - gdata(ip).p_ct ) /factor ;
           double ef1 = 0.5 * erf(x) ;
           double ef2 = 0.5 * erf(x + step/factor) ;
           double val = ef2 -ef1 ;
           pvecs_d(ip * MAX_P_SIZE + ii ) = val ;
//	   if(ip==0 && ii==0 ) printf("pvecs(0)=%f %f %f %f %f %f\n", val,start_p,gdata(0).p_sigma, factor, ef1, ef2);
         } ) ;
       }
      
      //Calculate tvecs
      if(nt == 1) {
         if( it == 0) tvecs_d(ip * MAX_T_SIZE) =1.0 ;
      } else {
       Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nt) ,
         [=] (int & ii ) {
           double step = tb.binsize ;
           double start = start_t ;
           double factor = sqrt2 * gdata(ip).t_sigma  ;
           double x = (start + step * ii - gdata(ip).t_ct ) /factor ;
           double ef1 = 0.5 * erf(x) ;
           double ef2 = 0.5 * erf(x + step/factor) ;
           double val = ef2 -ef1 ;
           tvecs_d(ip * MAX_T_SIZE + ii ) = val ;
//	   if(ip==0 && ii==0 ) printf("tvecs(0)=%f \n", val);
         } ) ;
      }
      //calculate weights
      if(weightstrat == 2 ) { 
	if(gdata(ip).p_sigma==0 ) {
		if(it == 0 ) qweights_d(ip*MAX_P_SIZE)= (start_p + pb.binsize -gdata(ip).p_ct )/pb.binsize ;
	} else {	
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, np) ,
           [=] (int & ii ) {
             double rel1 = (start_p + pb.binsize *ii  - gdata(ip).p_ct )/gdata(ip).p_sigma ;
             double rel2 = rel1+  pb.binsize /gdata(ip).p_sigma ;
             double gaus1 = exp(-0.5*rel1*rel1)  ;
             double gaus2 = exp(-0.5*rel2*rel2) ;
             double wt = -1.0 * gdata(ip).p_sigma/pb.binsize * (gaus2- gaus1) /sqrt(2.0 * PI )/pvecs_d(ip*MAX_P_SIZE + ii ) + (gdata(ip).p_ct - (start_p + (ii+1)* pb.binsize))/pb.binsize ;
             qweights_d(ip*MAX_P_SIZE + ii ) = -wt ;

         } ) ;
	}
      }

    } ) ;




  wend = omp_get_wtime();
  cout << "get_charge_vec() : set_sampling_pre() time " << wend- wstart<< endl;
  g_get_charge_vec_time_part4 += wend -wstart ;

//  set_sampling_bat( counter, max_patch_size) ;
  set_sampling_bat( npatches, nt_d,  np_d,  patch_idx, pvecs_d , tvecs_d, patch_d, normals,gdata  ) ;


  //copy diffs patch , np,nt,weight,index backup to host
  Kokkos::deep_copy( patch_idx_v_h, patch_idx) ;
  Kokkos::deep_copy( np_v_h, np_d) ;
  Kokkos::deep_copy( nt_v_h, nt_d) ;
  Kokkos::deep_copy( qweights_v_h, qweights_d) ;
  Kokkos::deep_copy( patch_v_h, patch_d) ;
  Kokkos::deep_copy( offsets_v_h, offsets_d) ;
  wstart = omp_get_wtime();

  cout << "get_charge_vec() : get_charge_vec() set_sampling_bat() time " << wstart-wend<< endl;
  std::cout<<"debug: patch values: "<<m_diffs.size() << " "<< patch_idx_v_h[m_diffs.size()] <<std::endl ;

  g_get_charge_vec_time_part4 += wstart-wend ;
  //for( long int jj=0 ; jj<m_patch_idx_h[m_diffs.size()] ; jj++ )
  //for( long int jj=0 ; jj<100 ; jj++ )
  //for( long int jj=0 ; jj<patch_idx_v_h[m_diffs.size()] ; jj++ )
// for ( int i1=0 ; i1< m_diffs.size() ; i1++) 
//	for (long int jj= patch_idx_v_h[i1] ; jj< patch_idx_v_h[i1+1] ; jj++) 
//  std::cout<<"PatchValues: "<<i1<< " " <<jj-patch_idx_v_h[i1]<<" "<< patch_v_h[jj] << std::endl ;


/*
  int idx=0 ;
  for (auto diff : m_diffs){
 
     
    auto patch = diff->get_patch();
    const auto qweight = diff->weights();

    memcpy(&(patch.data()[0]), &m_patch_h[m_patch_idx_h[idx]], (m_patch_idx_h[idx+1]-m_patch_idx_h[idx]) * sizeof(float));
    idx++ ;
*/
  int idx=0 ;
  for (auto diff : m_diffs){

    //const int poffset_bin = diff->poffset_bin();
    //const int toffset_bin = diff->toffset_bin();
    const int poffset_bin = offsets_v_h(npatches + idx );
    const int toffset_bin = offsets_v_h(idx);

    //const int np = patch.rows();
    //const int nt = patch.cols();
    const int np = np_v_h(idx);
    const int nt = nt_v_h(idx);
    // std::cout << "DEBUG: "
    // << " imp offset: " << poffset_bin << ", " << toffset_bin
    // << " ch offset: " << map_imp_ch[poffset_bin] << ", " << toffset_bin
    // << std::endl;

    
    for (int pbin = 0; pbin != np; pbin++){
      int abs_pbin = pbin + poffset_bin;
      if (abs_pbin < min_imp || abs_pbin >= max_imp) continue;
     // double weight = qweight[pbin];
      double weight = qweights_v_h(pbin + idx*MAX_P_SIZE) ;
      // double weight = 1.0 ;
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
        //double charge = patch(pbin, tbin);
        double charge = patch_v_h(patch_idx_v_h(idx) + pbin + tbin*np_v_h(idx));

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
    idx++ ;
  }
    // std::cout << "yuhw: get_charge_vec dump: \n";
    // for (size_t redimp=0; redimp<vec_vec_charge.size(); ++redimp) {
    //     std::cout << "redimp: " << redimp << std::endl;
    //     auto v = vec_vec_charge[redimp];
    //     for (auto t : v) {
    //         std::cout << get<0>(t) << ", " << get<1>(t) << ", " << get<2>(t) << "\n";
    //     }
    // }
  wend = omp_get_wtime();
  g_get_charge_vec_time_part3 += wend - wstart;
  cout << "get_charge_vec() :  Total ScaterAdd_Time : " << g_get_charge_vec_time_part3 << endl;
  cout << "get_charge_vec() : set_sampling()_Total_Time : " << g_get_charge_vec_time_part4 << endl;
  cout << "get_charge_vec() : m_fluctuate : " << m_fluctuate << endl;

#ifdef HAVE_CUDA_INC
  cout << "get_charge_vec() CUDA : set_sampling() part1 time : " << g_set_sampling_part1 << ", part2 (CUDA) time : " << g_set_sampling_part2 << endl;
  cout << "GaussianDiffusion::sampling_CUDA() part3 time : " << g_set_sampling_part3 << ", part4 time : " << g_set_sampling_part4 << ", part5 time : " << g_set_sampling_part5 << endl;
  cout << "GaussianDiffusion::sampling_CUDA() : g_total_sample_size : " << g_total_sample_size << endl;
#else
  cout << "get_charge_vec() : set_sampling() part1 time : " << g_set_sampling_part1 << ", part2 time : " << g_set_sampling_part2 << ", part3 time : " << g_set_sampling_part3 << endl;
#endif
}


void GenKokkos::BinnedDiffusion_transform::set_sampling_bat(const unsigned long npatches,  
		const size_vt nt_d ,
		const size_vt np_d, 
		const idx_vt patch_idx , 
		const db_vt pvecs_d,
		const db_vt tvecs_d,
	        fl_vt patch_d,
	        const db_vt normals,
	        const gd_vt gdata ) {
  Kokkos::Tools::pushRegion("set_sampling");


  bool fl = false ;
  if( m_fluctuate) fl = true    ;

  bool is_host= std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value ;

 // debug 
  is_host = false   ;
 // is_host = true   ;
  //kernel
  //int team_size = is_host ?  1 : 16 ;
  //int vec_len = is_host ? 4 : 8 ;

  Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(npatches,Kokkos::AUTO) ;
  if(is_host) {
    Kokkos::parallel_for("set_sampling_bat_OMP",  policy,
      KOKKOS_LAMBDA( const Kokkos::TeamPolicy<>::member_type & team ){
       int ip=team.league_rank() ;
      // if (ip==0) printf(" team size: %d", team.team_size()) ;

     ////  int np=p_idx(ip+1)-p_idx(ip) ;
      //// int nt=t_idx(ip+1)-t_idx(ip) ;
       int np=np_d(ip) ;
       int nt=nt_d(ip) ;
       int patch_size=np*nt ;
       unsigned long p0 =patch_idx(ip) ;

       double sum = 0.0 ;
       Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, patch_size) ,
         [=] (int & ii, double & lsum ) {
     ////       double v = pvecs_d(p_idx(ip)+ ii%np)*tvecs_d(t_idx(ip)+ ii/np) ;
            double v = pvecs_d(ip*MAX_P_SIZE+ ii%np)*tvecs_d(ip * MAX_T_SIZE + ii/np) ;
            patch_d(ii+p0) = (float) v ;
            lsum += v ;
         }, sum ) ;


////       double charge=charges_d(ip) ;
       double charge=gdata(ip).charge ;
       double charge_abs = abs(charge) ;
       int charge_sign = charge < 0  ? -1 :1 ;

       Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, patch_size) ,
         [=] ( int & ii ) {
           patch_d(ii+p0) *= float(charge/sum) ;
         } ) ;

      if( fl ){
         int n=(int) charge_abs;
         sum =0.0  ;

         Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, patch_size) ,
           [=] (int & ii, double & lsum ) {
               double p =  patch_d(ii+p0)/charge ;
               double q = 1-p ;
               double mu = n*p ;
               double sigma = sqrt(p*q*n) ;
               p = normals(ii+p0)*sigma + mu ;
               lsum += p ;
               }, sum ) ;
         Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, patch_size) ,
           [=] ( int & ii ) {
             patch_d(ii+p0) *= (float) charge_abs/sum  ;
           } ) ;

       }
    } ) ;

  } else {
    Kokkos::parallel_for("Set_Sampling_BAT", policy, 
      KOKKOS_LAMBDA( const Kokkos::TeamPolicy<>::member_type & team ){
       int ip=team.league_rank() ;
     //  if (ip==0) printf(" team size: %d", team.team_size()) ;
       int np=np_d(ip) ;
       int nt=nt_d(ip) ;
       int patch_size=np*nt ;
       unsigned long p0 =patch_idx(ip) ;

       double sum = 0.0 ;

//       Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, patch_size) ,
         Kokkos::parallel_reduce(Kokkos::TeamVectorRange(team, patch_size) ,
         [=] (int & ii, double & lsum ) { 
 //     //      double v = pvecs_d(p_idx(ip)+ ii%np)*tvecs_d(t_idx(ip)+ ii/np) ;
     //  if( ip ==17897  ) printf("Patch17897: ii=%d pv=%e tv=%e \n", ii, pvecs_d(ip*MAX_P_SIZE + ii%np), tvecs_d(ip * MAX_T_SIZE+ ii/np) ) ;  
            double v = pvecs_d(ip*MAX_P_SIZE+ ii%np)*tvecs_d(ip * MAX_T_SIZE + ii/np) ;
	    patch_d(ii+p0) = (float) v ;
	    lsum += v ;
         }, sum ) ;


   ////    double charge=charges_d(ip) ;
       double charge=gdata(ip).charge ;
       double charge_abs = abs(charge) ;
       int charge_sign = charge < 0  ? -1 :1 ;
 //  if(ip==17896 ) printf("sum= %e, %e, %e %e %e %e \n", sum, normals(p0), patch_d(p0), charge_abs, pvecs_d(ip),tvecs_d(ip)  ) ;

       Kokkos::parallel_for(Kokkos::TeamVectorRange(team, patch_size) ,
//       Kokkos::parallel_for(Kokkos::TeamThreadRange(team, patch_size) ,
	 [=] ( int & ii ) {
           patch_d(ii+p0) *= float(charge/sum) ;
	 } ) ;
       
       if( fl ){
       //if( 0 ){
	 int n=(int) charge_abs;
         sum =0.0  ;

         Kokkos::parallel_reduce(Kokkos::TeamVectorRange(team, patch_size) ,
//         Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, patch_size) ,
           [=] (int & ii, double & lsum ) { 
               double p =  patch_d(ii+p0)/charge ;
               double q = 1-p ;
               double mu = n*p ;
               double sigma = sqrt(p*q*n) ;
               p = normals(ii+p0)*sigma + mu ;
	       lsum += p ;
               }, sum ) ;

         Kokkos::parallel_for(Kokkos::TeamVectorRange(team, patch_size) ,
//         Kokkos::parallel_for(Kokkos::TeamThreadRange(team, patch_size) ,
	   [=] ( int & ii ) {
             patch_d(ii+p0) *= (float) charge_abs/sum  ;
	   } ) ;

       }
//       if(ip== 0 &&team.team_rank()==0 ) printf("team-size = %d,%dx%d \n", team.team_size() ,np,nt); 

    } ) ;

  }  
    Kokkos::Tools::popRegion() ;
 
//  Kokkos::deep_copy(patches_v_h, patch_d ) ;
//  for(int n=0; n<100 ; n++) std::cout<<patches_v_h(n)<<std::endl ; 


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
