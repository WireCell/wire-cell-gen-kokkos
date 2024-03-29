#include "WireCellGenKokkos/ImpactTransform.h"
#include "WireCellUtil/Testing.h"
#include "WireCellUtil/FFTBestLength.h"

#include "WireCellGenKokkos/KokkosArray.h"

#include <iostream>  // debugging.
#include <omp.h>

double g_get_charge_vec_time = 0.0;
double g_fft_time = 0.0;

using namespace std;

using namespace WireCell;
GenKokkos::ImpactTransform::ImpactTransform(IPlaneImpactResponse::pointer pir, BinnedDiffusion_transform& bd)
  : m_pir(pir)
  , m_bd(bd)
  , log(Log::logger("sim"))
  {}

bool GenKokkos::ImpactTransform::transform_vector()
{
    double timer_transform = omp_get_wtime();
    // arrange the field response (210 in total, pitch_range/impact)
    // number of wires nwires ...
    m_num_group = std::round(m_pir->pitch() / m_pir->impact()) + 1;  // 11
    m_num_pad_wire = std::round((m_pir->nwires() - 1) / 2.);         // 10

    const auto pimpos = m_bd.pimpos();

    // std::cout << "yuhw: transform_vector: \n"
    // << " m_num_group: " << m_num_group
    // << " m_num_pad_wire: " << m_num_pad_wire
    // << " pitch: " << m_pir->pitch()
    // << " impact: " << m_pir->impact()
    // << std::endl;
    for (int i = 0; i != m_num_group; i++) {
        double rel_cen_imp_pos;
        if (i != m_num_group - 1) {
            rel_cen_imp_pos = -m_pir->pitch() / 2. + m_pir->impact() * i + 1e-9;
        }
        else {
            rel_cen_imp_pos = -m_pir->pitch() / 2. + m_pir->impact() * i - 1e-9;
        }
        m_vec_impact.push_back(std::round(rel_cen_imp_pos / m_pir->impact()));
        std::map<int, IImpactResponse::pointer> map_resp;  // already in freq domain

        for (int j = 0; j != m_pir->nwires(); j++) {
            map_resp[j - m_num_pad_wire] = m_pir->closest(rel_cen_imp_pos - (j - m_num_pad_wire) * m_pir->pitch());
            Waveform::compseq_t response_spectrum = map_resp[j - m_num_pad_wire]->spectrum();
            // std::cout
            // << " igroup: " << i
            // << " rel_cen_imp_pos: " << rel_cen_imp_pos
            // << " iwire: " << j
            // << " dist: " << rel_cen_imp_pos - (j - m_num_pad_wire) * m_pir->pitch()
            // << std::endl;
        }
        m_vec_map_resp.push_back(map_resp);

        std::vector<std::tuple<int, int, double> > vec_charge;  // ch, time, charge
        m_vec_vec_charge.push_back(vec_charge);
    }

    // now work on the charge part ...
    // trying to sampling ...
    double wstart = omp_get_wtime();
    m_bd.get_charge_vec(m_vec_vec_charge, m_vec_impact);
    double wend = omp_get_wtime();
    g_get_charge_vec_time += wend - wstart;
    log->debug("ImpactTransform::ImpactTransform() : get_charge_vec() Total_Time : {}", g_get_charge_vec_time);
    std::pair<int, int> impact_range = m_bd.impact_bin_range(m_bd.get_nsigma());
    std::pair<int, int> time_range = m_bd.time_bin_range(m_bd.get_nsigma());

    int start_ch = std::floor(impact_range.first * 1.0 / (m_num_group - 1)) - 1;
    int end_ch = std::ceil(impact_range.second * 1.0 / (m_num_group - 1)) + 2;
    if ((end_ch - start_ch) % 2 == 1) end_ch += 1;
    int start_tick = time_range.first - 1;
    int end_tick = time_range.second + 2;
    if ((end_tick - start_tick) % 2 == 1) end_tick += 1;

    int npad_wire = 0;
    const size_t ntotal_wires = fft_best_length(end_ch - start_ch + 2 * m_num_pad_wire, 1);

    npad_wire = (ntotal_wires - end_ch + start_ch) / 2;
    m_start_ch = start_ch - npad_wire;
    m_end_ch = end_ch + npad_wire;

    int npad_time = m_pir->closest(0)->waveform_pad();
    const size_t ntotal_ticks = fft_best_length(end_tick - start_tick + npad_time);

    npad_time = ntotal_ticks - end_tick + start_tick;
    m_start_tick = start_tick;
    m_end_tick = end_tick + npad_time;

    wstart = omp_get_wtime();
    Array::array_xxc acc_data_f_w =
        Array::array_xxc::Zero(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);

    int num_double = (m_vec_vec_charge.size() - 1) / 2;

    // speed up version , first five
    for (int i = 0; i != num_double; i++) {
        Array::array_xxc c_data = Array::array_xxc::Zero(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);

        // fill normal order
        for (size_t j = 0; j != m_vec_vec_charge.at(i).size(); j++) {
            c_data(std::get<0>(m_vec_vec_charge.at(i).at(j)) + npad_wire - start_ch,
                   std::get<1>(m_vec_vec_charge.at(i).at(j)) - m_start_tick) +=
                std::get<2>(m_vec_vec_charge.at(i).at(j));
        }
        m_vec_vec_charge.at(i).clear();
        m_vec_vec_charge.at(i).shrink_to_fit();

        // fill reverse order
        int ii = num_double * 2 - i;
        for (size_t j = 0; j != m_vec_vec_charge.at(ii).size(); j++) {
            c_data(end_ch + npad_wire - 1 - std::get<0>(m_vec_vec_charge.at(ii).at(j)),
                   std::get<1>(m_vec_vec_charge.at(ii).at(j)) - m_start_tick) +=
                std::complex<float>(0, std::get<2>(m_vec_vec_charge.at(ii).at(j)));
        }
        m_vec_vec_charge.at(ii).clear();
        m_vec_vec_charge.at(ii).shrink_to_fit();

        // Do FFT on time
        c_data = Array::dft_cc(c_data, 0);
        // Do FFT on wire
        c_data = Array::dft_cc(c_data, 1);

        // std::cout << i << std::endl;
        {
            Array::array_xxc resp_f_w =
                Array::array_xxc::Zero(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);
            {
                Waveform::compseq_t rs1 = m_vec_map_resp.at(i)[0]->spectrum();
                // do a inverse FFT
                Waveform::realseq_t rs1_t = Waveform::idft(rs1);
                // pick the first xxx ticks
                Waveform::realseq_t rs1_reduced(m_end_tick - m_start_tick, 0);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    if (icol >= int(rs1_t.size())) break;
                    rs1_reduced.at(icol) = rs1_t[icol];
                }
                // do a FFT
                rs1 = Waveform::dft(rs1_reduced);

                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    resp_f_w(0, icol) = rs1[icol];
                }
            }

            for (int irow = 0; irow != m_num_pad_wire; irow++) {
                Waveform::compseq_t rs1 = m_vec_map_resp.at(i)[irow + 1]->spectrum();
                Waveform::realseq_t rs1_t = Waveform::idft(rs1);
                Waveform::realseq_t rs1_reduced(m_end_tick - m_start_tick, 0);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    if (icol >= int(rs1_t.size())) break;
                    rs1_reduced.at(icol) = rs1_t[icol];
                }
                rs1 = Waveform::dft(rs1_reduced);
                Waveform::compseq_t rs2 = m_vec_map_resp.at(i)[-irow - 1]->spectrum();
                Waveform::realseq_t rs2_t = Waveform::idft(rs2);
                Waveform::realseq_t rs2_reduced(m_end_tick - m_start_tick, 0);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    if (icol >= int(rs2_t.size())) break;
                    rs2_reduced.at(icol) = rs2_t[icol];
                }
                rs2 = Waveform::dft(rs2_reduced);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    resp_f_w(irow + 1, icol) = rs1[icol];
                    resp_f_w(end_ch - start_ch - 1 - irow + 2 * npad_wire, icol) = rs2[icol];
                }
            }
            // std::cout << " vector resp: " << i << " : " << Array::idft_cr(resp_f_w, 0) << std::endl;

            // Do FFT on wire for response // slight larger
            resp_f_w = Array::dft_cc(resp_f_w, 1);  // Now becomes the f and f in both time and wire domain ...
            // multiply them together
            c_data = c_data * resp_f_w;
        }

        // Do inverse FFT on wire
        c_data = Array::idft_cc(c_data, 1);

        // Add to wire result in frequency
        acc_data_f_w += c_data;
    }


    log->info("yuhw: start-end {}-{} {}-{}",m_start_ch, m_end_ch, m_start_tick, m_end_tick);

    // central region ...
    {
        int i = num_double;
        // fill response array in frequency domain

        Array::array_xxc data_f_w;
        {
            Array::array_xxf data_t_w =
                Array::array_xxf::Zero(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);
            // fill charge array in time-wire domain // slightly larger
            for (size_t j = 0; j != m_vec_vec_charge.at(i).size(); j++) {
                data_t_w(std::get<0>(m_vec_vec_charge.at(i).at(j)) + npad_wire - start_ch,
                         std::get<1>(m_vec_vec_charge.at(i).at(j)) - m_start_tick) +=
                    std::get<2>(m_vec_vec_charge.at(i).at(j));
            }
            // m_decon_data = data_t_w; // DEBUG tap data_t_w out
            m_vec_vec_charge.at(i).clear();
            m_vec_vec_charge.at(i).shrink_to_fit();

            // Do FFT on time
            data_f_w = Array::dft_rc(data_t_w, 0);
            // Do FFT on wire
            data_f_w = Array::dft_cc(data_f_w, 1);
        }

        {
            Array::array_xxc resp_f_w =
                Array::array_xxc::Zero(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);

            {
                Waveform::compseq_t rs1 = m_vec_map_resp.at(i)[0]->spectrum();

                // do a inverse FFT
                Waveform::realseq_t rs1_t = Waveform::idft(rs1);
                // pick the first xxx ticks
                Waveform::realseq_t rs1_reduced(m_end_tick - m_start_tick, 0);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    if (icol >= int(rs1_t.size())) break;
                    rs1_reduced.at(icol) = rs1_t[icol];
                }
                // do a FFT
                rs1 = Waveform::dft(rs1_reduced);

                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    resp_f_w(0, icol) = rs1[icol];
                }
            }
            for (int irow = 0; irow != m_num_pad_wire; irow++) {
                Waveform::compseq_t rs1 = m_vec_map_resp.at(i)[irow + 1]->spectrum();
                Waveform::realseq_t rs1_t = Waveform::idft(rs1);
                Waveform::realseq_t rs1_reduced(m_end_tick - m_start_tick, 0);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    if (icol >= int(rs1_t.size())) break;
                    rs1_reduced.at(icol) = rs1_t[icol];
                }
                rs1 = Waveform::dft(rs1_reduced);
                Waveform::compseq_t rs2 = m_vec_map_resp.at(i)[-irow - 1]->spectrum();
                Waveform::realseq_t rs2_t = Waveform::idft(rs2);
                Waveform::realseq_t rs2_reduced(m_end_tick - m_start_tick, 0);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    if (icol >= int(rs2_t.size())) break;
                    rs2_reduced.at(icol) = rs2_t[icol];
                }
                rs2 = Waveform::dft(rs2_reduced);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    resp_f_w(irow + 1, icol) = rs1[icol];
                    resp_f_w(end_ch - start_ch - 1 - irow + 2 * npad_wire, icol) = rs2[icol];
                }
            }
            // std::cout << " vector resp: " << i << " : " << Array::idft_cr(resp_f_w, 0) << std::endl;
            // Do FFT on wire for response // slight larger
            resp_f_w = Array::dft_cc(resp_f_w, 1);  // Now becomes the f and f in both time and wire domain ...
            // multiply them together
            data_f_w = data_f_w * resp_f_w;
        }

        // Do inverse FFT on wire
        data_f_w = Array::idft_cc(data_f_w, 1);

        // Add to wire result in frequency
        acc_data_f_w += data_f_w;
    }
    // m_decon_data = Array::idft_cr(acc_data_f_w, 0); // DEBUG only central
    acc_data_f_w = Array::idft_cc(acc_data_f_w, 0);  //.block(npad_wire,0,nwires,nsamples);
    Array::array_xxf real_m_decon_data = acc_data_f_w.real();
    Array::array_xxf img_m_decon_data = acc_data_f_w.imag().colwise().reverse();
    m_decon_data = real_m_decon_data + img_m_decon_data;

    double timer_fft = omp_get_wtime() - wstart;
    log->debug("ImpactTransform::transform_vector: FFT_Time: {}", timer_fft);
    timer_transform = omp_get_wtime() - timer_transform;
    log->debug("ImpactTransform::transform_vector: Total: {}", timer_transform);
    g_fft_time += timer_fft ;
    log->debug("ImpactTransform::transform_vector: Total_FFT_Time: {}", g_fft_time);


    
    log->debug("ImpactTransform::transform_vector: # of channels: {} # of ticks: {}", m_decon_data.rows(), m_decon_data.cols());
    log->debug("transform_vector: m_decon_data.sum(): {}", m_decon_data.sum());

    return true;
}


bool GenKokkos::ImpactTransform::transform_matrix()
{
    double timer_transform = omp_get_wtime();
    double td0(0.0), td1(.0)   ;
    // arrange the field response (210 in total, pitch_range/impact)
    // number of wires nwires ...
    m_num_group = std::round(m_pir->pitch() / m_pir->impact()) + 1;  // 11
    m_num_pad_wire = std::round((m_pir->nwires() - 1) / 2.);         // 10

    std::pair<int, int> impact_range = m_bd.impact_bin_range(m_bd.get_nsigma());
    std::pair<int, int> time_range = m_bd.time_bin_range(m_bd.get_nsigma());

    int start_ch = std::floor(impact_range.first * 1.0 / (m_num_group - 1)) - 1;
    int end_ch = std::ceil(impact_range.second * 1.0 / (m_num_group - 1)) + 2;
    if ((end_ch - start_ch) % 2 == 1) end_ch += 1;
    int start_pitch = impact_range.first - 1;
    int end_pitch = impact_range.second + 2;
    if ((end_pitch - start_pitch) % 2 == 1) end_pitch += 1;
    int start_tick = time_range.first - 1;
    int end_tick = time_range.second + 2;
    if ((end_tick - start_tick) % 2 == 1) end_tick += 1;

    int npad_wire = 0;
    const size_t ntotal_wires = fft_best_length(end_ch - start_ch + 2 * m_num_pad_wire, 1);
    npad_wire = (ntotal_wires - end_ch + start_ch) / 2;
    m_start_ch = start_ch - npad_wire;
    m_end_ch = end_ch + npad_wire;

    int npad_time = m_pir->closest(0)->waveform_pad();
    const size_t ntotal_ticks = fft_best_length(end_tick - start_tick + npad_time);
    npad_time = ntotal_ticks - end_tick + start_tick;
    m_start_tick = start_tick;
    m_end_tick = end_tick + npad_time;

    int npad_pitch = 0;
    const size_t ntotal_pitches = fft_best_length((end_ch - start_ch + 2 * npad_wire)*(m_num_group - 1), 1);
    npad_pitch = (ntotal_pitches - end_pitch + start_pitch) / 2;
    start_pitch = start_pitch - npad_pitch;
    end_pitch = end_pitch + npad_pitch;
    
    // std::cout << "yuhw: "
    // << " channel: { " << start_ch << ", " << end_ch << " } "
    // << " pitch: { " << start_pitch << ", " << end_pitch << " }"
    // << " tick: { " << m_start_tick << ", " << m_end_tick << " }\n";

    // now work on the charge part ...
    // trying to sampling ...
    double wstart = omp_get_wtime();
    td0 += wstart-timer_transform ;
    KokkosArray::array_xxf f_data = KokkosArray::Zero<KokkosArray::array_xxf>(end_pitch - start_pitch, m_end_tick - m_start_tick);;
    m_bd.get_charge_matrix_kokkos(f_data, m_vec_impact, start_pitch, m_start_tick);
    // Kokkos::fence();
    //std::cout << KokkosArray::dump_2d_view(f_data, 10);
    double wend = omp_get_wtime();
    td1 += wend-wstart ;
    g_get_charge_vec_time += wend - wstart;
    //log->debug("ImpactTransform::ImpactTransform() : get_charge_vec() Total running time : {}", g_get_charge_vec_time);
    log->debug("ImpactTransform::ImpactTransform() : get_charge_matrix() Total_Time :  {}", g_get_charge_vec_time);

    wstart = omp_get_wtime();
    KokkosArray::array_xxc acc_data_f_w = KokkosArray::Zero<KokkosArray::array_xxc>(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);
    KokkosArray::array_xxf acc_data_t_w = KokkosArray::Zero<KokkosArray::array_xxf>(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);
    log->info("yuhw: pitch   {} {} {} {}",start_pitch, end_pitch, f_data.extent(0), f_data.extent(1));
    log->info("yuhw: start-end {}-{} {}-{} {} {}",m_start_ch, m_end_ch, m_start_tick, m_end_tick, acc_data_f_w.extent(0), acc_data_f_w.extent(1));
    log->info("yuhw: m_num_group {}",m_num_group);

    //  Convolution with Field Response
    {
        // fine grained resp. matrix
        KokkosArray::array_xxc resp_f_w_k =
            KokkosArray::Zero<KokkosArray::array_xxc>(end_pitch - start_pitch, m_end_tick - m_start_tick);
        int nimpact = m_pir->nwires() * (m_num_group - 1);
        assert(end_pitch - start_pitch >= nimpact);
        double max_impact = (0.5 + m_num_pad_wire) * m_pir->pitch();
        // TODO this loop could be parallerized by seperating it into 2 parts
        // 1, extract the m_pir into a Kokkos::Array first
        // 2, do various operation in a parallerized fasion
	
        Kokkos::View<int*> idx_d(Kokkos::ViewAllocateWithoutInitializing("Idx"),nimpact) ;
	KokkosArray::array_xxc resp_redu(Kokkos::ViewAllocateWithoutInitializing("resp_redu"),m_end_tick - m_start_tick,nimpact) ;
	auto idx_h = Kokkos::create_mirror_view(idx_d);
        
	auto ip0 = m_pir->closest(-max_impact + 0.01 * m_pir->impact()) ;
        int sp_size = ip0->spectrum() .size() ;
	KokkosArray::array_xxc sp_fs(Kokkos::ViewAllocateWithoutInitializing("SP_FS"),sp_size,nimpact) ;
	auto sps_h = Kokkos::create_mirror_view(sp_fs);
	int fillsize = min(sp_size, m_end_tick - m_start_tick );



        for (int jimp = 0; jimp < nimpact; ++jimp) {
            float impact = -max_impact + jimp * m_pir->impact();
             //std::cout << "yuhw: jimp " << jimp << ", " << impact;
            // to avoid exess of boundary.
            // central is < 0, after adding 0.01 * m_pir->impact(), it is > 0
            impact += impact < 0 ? 0.01 * m_pir->impact() : -0.01 * m_pir->impact();
            auto ip = m_pir->closest(impact);
            Waveform::compseq_t sp_f = ip->spectrum();


	    memcpy((void*) &sps_h(0, jimp) , (void*) &sp_f[0] , sp_size*2*sizeof(float) ) ; 
           
            // 0 and right on the left; left on the right
            // int idx = impact < 0 ? end_pitch - start_pitch - (std::round(nimpact / 2.) - jimp)
            //                      : jimp - std::round(nimpact / 2.);
            // 0 and left and the left; right on the right
            int idx = impact < 0.1 * m_pir->impact() ? std::round(nimpact / 2.) - jimp
                                 : end_pitch - start_pitch - (jimp - std::round(nimpact / 2.));
	    idx_h(jimp) = idx ;
        }

        Kokkos::deep_copy(idx_d, idx_h);
        Kokkos::deep_copy(sp_fs, sps_h );

	auto  sp_ts = KokkosArray::idft_cr(sp_fs,1) ;
	if (fillsize < sp_size )  
		Kokkos::resize(sp_ts, fillsize, nimpact) ;
	resp_redu = KokkosArray::dft_rc(sp_ts,1) ;
	Kokkos::resize(sp_ts,0,0) ;
	Kokkos::resize(sp_fs,0,0) ;

	Kokkos::parallel_for( "FillResp",
            Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {resp_redu.extent(0), resp_redu.extent(1)}),
            KOKKOS_LAMBDA(const KokkosArray::Index& i0, const KokkosArray::Index& i1) {
                resp_f_w_k(idx_d(i1), i0) = resp_redu (i0, i1) ;
            });


        auto data_d = KokkosArray::dft_rc(f_data, 0);
	Kokkos::resize(f_data,0,0) ;
        auto data_c = KokkosArray::dft_cc(data_d, 1);

        KokkosArray::dft_cc(resp_f_w_k, data_d, 1);
	resp_f_w_k = data_d ;

        // S(f) * R(f)
        Kokkos::parallel_for( "S*R",
            Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {data_c.extent(0), data_c.extent(1)}),
            KOKKOS_LAMBDA(const KokkosArray::Index& i0, const KokkosArray::Index& i1) {
                data_c(i0, i1) *= resp_f_w_k(i0, i1);
                // data_c(i0, i1) *= 1.0;
            });
        // transfer wire to time domain
	KokkosArray::idft_cc(data_c,data_d, 1);
	data_c = data_d ;
	Kokkos::resize(resp_f_w_k, 0 , 0 ) ;

        // extract M(channel) from M(impact)
        Kokkos::parallel_for( "Pick ",
            Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {acc_data_f_w.extent(0), acc_data_f_w.extent(1)}),
            KOKKOS_LAMBDA(const KokkosArray::Index& i0, const KokkosArray::Index& i1) {
                acc_data_f_w(i0, i1) = data_c((i0 + 1) * 10, i1);
            });
    }

    acc_data_t_w = KokkosArray::idft_cr(acc_data_f_w, 0);

    m_decon_data_v = acc_data_t_w ;

    auto acc_data_t_w_h = Kokkos::create_mirror_view(acc_data_t_w);
    //std::cout << "yuhw: acc_data_t_w_h: " << KokkosArray::dump_2d_view(acc_data_t_w,10) << std::endl;
    Kokkos::deep_copy(acc_data_t_w_h, acc_data_t_w);
    //std::cout << "mdeconn_data_dv: (75,1) " << acc_data_t_w_h(75,1) <<std::endl ;
    Eigen::Map<Eigen::ArrayXXf> acc_data_t_w_eigen((float*) acc_data_t_w_h.data(), acc_data_t_w_h.extent(0), acc_data_t_w_h.extent(1));
    m_decon_data = acc_data_t_w_eigen; // FIXME: reduce this copy
   // std::cout << "mdeconn_data: (75,1) " << m_decon_data(75,1) <<std::endl ;
    
    double timer_fft = omp_get_wtime() - wstart;
    g_fft_time += timer_fft ;
    std::cout<<"m_decon_data.sum: "<<m_decon_data.rows()<<"," <<m_decon_data.cols()<<",  "<< m_decon_data.sum() <<std::endl ;
    log->debug("ImpactTransform::transform_matrix: FFT: {}", timer_fft);
    timer_transform = omp_get_wtime() - timer_transform;
    log->debug("ImpactTransform::transform_matrix: Total: {}", timer_transform);
    log->debug("ImpactTransform::transform_matrix: # of channels: {} # of ticks: {}", m_decon_data.rows(), m_decon_data.cols());
    log->debug("ImpactTransform::transform_matrix: m_decon_data.sum(): {}", m_decon_data.sum());
    std::cout<<"Tranform_matrix_p0_Time: "<<td0 <<std::endl;
    std::cout<<"get_charge_matrix_Time: "<<td1 <<std::endl;
    std::cout<<"FFTs_Time: "<<timer_fft <<std::endl;
    log->debug("ImpactTransform::transform_matrix: Total_FFT_Time: {}", g_fft_time);
    return true;
}

GenKokkos::ImpactTransform::~ImpactTransform() {}

Waveform::realseq_t GenKokkos::ImpactTransform::waveform(int iwire) const
{
    if(iwire==0) std::cout<<"s-ch,e_ch=" << m_start_ch<<" "<< m_end_ch<<" S_tick,E_tick=" << m_start_tick<<" " <<m_end_tick<<std::endl;
    const int nsamples = m_bd.tbins().nbins();
    if (iwire < m_start_ch || iwire >= m_end_ch) {
        return Waveform::realseq_t(nsamples, 0.0);
    }
    else {
        Waveform::realseq_t wf(nsamples, 0.0);
        for (int i = 0; i != nsamples; i++) {
            if (i >= m_start_tick && i < m_end_tick) {
                wf.at(i) = m_decon_data(iwire - m_start_ch, i - m_start_tick);
            }
            else {
                // wf.at(i) = 1e-25;
            }
            // std::cout << m_decon_data(iwire-m_start_ch,i-m_start_tick) << std::endl;
        }

        if (m_pir->closest(0)->long_aux_waveform().size() > 0) {
            // now convolute with the long-range response ...
            const size_t nlength = fft_best_length(nsamples + m_pir->closest(0)->long_aux_waveform_pad());

            //nlength = nsamples;

            //std::cout << nlength << " " << nsamples + m_pir->closest(0)->long_aux_waveform_pad() << std::endl;

            wf.resize(nlength, 0);
            Waveform::realseq_t long_resp = m_pir->closest(0)->long_aux_waveform();
            long_resp.resize(nlength, 0);
            Waveform::compseq_t spec = Waveform::dft(wf);
            Waveform::compseq_t long_spec = Waveform::dft(long_resp);
           for (size_t i = 0; i != nlength; i++) {
                spec.at(i) *= long_spec.at(i);
            }
            wf = Waveform::idft(spec);
            wf.resize(nsamples, 0);
         }

        return wf;
    }
}

KokkosArray::array_xxf GenKokkos::ImpactTransform::waveform_v(int nwires) const
{

    	const int nsamples = m_bd.tbins().nbins();
    	auto wfs = KokkosArray::Zero<KokkosArray::array_xxf> ( nsamples, nwires) ;	

    	if (m_pir->closest(0)->long_aux_waveform().size() > 0) {
	   const size_t nlength = fft_best_length(nsamples + m_pir->closest(0)->long_aux_waveform_pad()) ;
	   Kokkos::resize(wfs, nlength, nwires) ;

     
     	   int chs_start = m_start_ch > 0 ? m_start_ch : 0 ;
    	   int chs_end = m_end_ch > nwires ? nwires : m_end_ch ;
     	   int samples_start = m_start_tick >0 ? m_start_tick : 0 ;
     	   int samples_end = m_end_tick > nsamples ? nsamples : m_end_tick ;
	   auto data = m_decon_data_v ;
	   int ofs_ch = m_start_ch ;
	   int ofs_tick = m_start_tick  ;

           Kokkos::parallel_for( "ConvData4LongRes ",
               Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({samples_start, chs_start}, { samples_end , chs_end }),
               KOKKOS_LAMBDA(const KokkosArray::Index& i0, const KokkosArray::Index& i1) {
                   wfs(i0 , i1 ) = data(i1 - ofs_ch , i0 - ofs_tick );
            });

     	   auto specs = KokkosArray::dft_rc(wfs,1) ;

           Waveform::realseq_t long_resp = m_pir->closest(0)->long_aux_waveform();
           long_resp.resize(nlength, 0);
           Waveform::compseq_t long_spec = Waveform::dft(long_resp);
	    
     
	   KokkosArray::array_xc  long_spec_d("long_resp", nlength) ;
     	   auto long_spec_h = Kokkos::create_mirror_view(long_spec_d) ;
     	   memcpy((void*) &long_spec_h(0) , (void*) &long_spec[0] , nlength*2*sizeof(float) ) ;
     	   Kokkos::deep_copy(long_spec_d, long_spec_h ); 

           // S(f) * LongR(f)
           Kokkos::parallel_for( "S*LongR",
           Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, { specs.extent(0), specs.extent(1)}),
                KOKKOS_LAMBDA(const KokkosArray::Index& i0, const KokkosArray::Index& i1) {
                specs(i0, i1) *= long_spec_d(i0);
            });
	   KokkosArray::idft_cr(specs,wfs, 1 ); 
      
	   Kokkos::resize(wfs,  nsamples, nwires) ; 
    	}
        return wfs;
    
}
