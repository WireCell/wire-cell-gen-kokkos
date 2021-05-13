#include "WireCellGenKokkos/ImpactTransform.h"
#include "WireCellUtil/Testing.h"
#include "WireCellUtil/FFTBestLength.h"

#include "WireCellGenKokkos/KokkosArray.h"

#include <iostream>  // debugging.
#include <omp.h>

double g_get_charge_vec_time = 0.0;

using namespace std;

using namespace WireCell;
GenKokkos::ImpactTransform::ImpactTransform(IPlaneImpactResponse::pointer pir, BinnedDiffusion_transform& bd)
  : m_pir(pir)
  , m_bd(bd)
  , log(Log::logger("sim"))
  {}

bool GenKokkos::ImpactTransform::transform()
{
    // arrange the field response (210 in total, pitch_range/impact)
    // number of wires nwires ...
    m_num_group = std::round(m_pir->pitch() / m_pir->impact()) + 1;  // 11
    m_num_pad_wire = std::round((m_pir->nwires() - 1) / 2.);         // 10

    const auto pimpos = m_bd.pimpos();

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
    log->debug("ImpactTransform::ImpactTransform() : get_charge_vec() Total running time : {}", g_get_charge_vec_time);

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

    acc_data_f_w = Array::idft_cc(acc_data_f_w, 0);  //.block(npad_wire,0,nwires,nsamples);
    Array::array_xxf real_m_decon_data = acc_data_f_w.real();
    Array::array_xxf img_m_decon_data = acc_data_f_w.imag().colwise().reverse();
    m_decon_data = real_m_decon_data + img_m_decon_data;

    log->debug("ImpactTransform: # of channels: {} # of ticks: {}", m_decon_data.rows(), m_decon_data.cols());

    return true;
}


bool GenKokkos::ImpactTransform::transform_kokkos()
{
    // arrange the field response (210 in total, pitch_range/impact)
    // number of wires nwires ...
    m_num_group = std::round(m_pir->pitch() / m_pir->impact()) + 1;  // 11
    m_num_pad_wire = std::round((m_pir->nwires() - 1) / 2.);         // 10

    const auto pimpos = m_bd.pimpos();

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

    // now work on the charge part ...
    // trying to sampling ...
    double wstart = omp_get_wtime();
    KokkosArray::array_xxf f_data = KokkosArray::Zero<KokkosArray::array_xxf>(end_pitch - start_pitch + 2 * npad_pitch, m_end_tick - m_start_tick);;
    m_bd.get_charge_matrix_kokkos(f_data, m_vec_impact, start_pitch, m_start_tick);
    // Kokkos::fence();
    // std::cout << KokkosArray::dump_2d_view(f_data, 100);
    double wend = omp_get_wtime();
    g_get_charge_vec_time += wend - wstart;
    log->debug("ImpactTransform::ImpactTransform() : get_charge_vec() Total running time : {}", g_get_charge_vec_time);

    KokkosArray::array_xxc acc_data_f_w = KokkosArray::Zero<KokkosArray::array_xxc>(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);
    log->info("yuhw: pitch   {} {} {} {}",start_pitch, end_pitch, f_data.extent(0), f_data.extent(1));
    log->info("yuhw: channel {} {} {} {}",m_start_ch, m_end_ch, acc_data_f_w.extent(0), acc_data_f_w.extent(1));
    // central region ...
    {
        KokkosArray::array_xxc data_f_w = KokkosArray::Zero<KokkosArray::array_xxc>(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);
        {
            KokkosArray::array_xxf data_t_w =
                KokkosArray::Zero<KokkosArray::array_xxf>(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);
            Kokkos::parallel_for("data_t_w",
                                 Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>(
                                     {0, 0}, {data_t_w.extent(0), data_t_w.extent(1)}),
                                 KOKKOS_LAMBDA(const KokkosArray::Index& i0, const KokkosArray::Index& i1) {
                                    //  data_t_w(i0, i1) = f_data(i0 * (m_num_group - 1), i1);
                                     data_t_w(i0, i1) = f_data(i0*10, i1);
                                 });
            // Kokkos::fence();

            // Do FFT on time
            data_f_w = KokkosArray::dft_rc(data_t_w, 0);
            // Do FFT on wire
            data_f_w = KokkosArray::dft_cc(data_f_w, 1);
        }

        // Do inverse FFT on wire
        data_f_w = KokkosArray::idft_cc(data_f_w, 1);

        // Add to wire result in frequency
        Kokkos::parallel_for("acc_data_f_w",
                            Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {data_f_w.extent(0), data_f_w.extent(1)}),
                            KOKKOS_LAMBDA(const KokkosArray::Index& i0, const KokkosArray::Index& i1) { acc_data_f_w(i0, i1) += data_f_w(i0, i1);});
    }

    auto acc_data_t_w = KokkosArray::idft_cr(acc_data_f_w, 0);

    auto acc_data_t_w_h = Kokkos::create_mirror_view(acc_data_t_w);
    std::cout << "yuhw: acc_data_t_w_h: " << KokkosArray::dump_2d_view(acc_data_t_w,1000000) << std::endl;
    Kokkos::deep_copy(acc_data_t_w_h, acc_data_t_w_h);
    Eigen::Map<Eigen::ArrayXXf> acc_data_f_w_eigen((float*) acc_data_t_w_h.data(), acc_data_t_w_h.extent(0), acc_data_t_w_h.extent(1));
    m_decon_data = acc_data_f_w_eigen; // FIXME: reduce this copy

    log->debug("ImpactTransform: # of channels: {} # of ticks: {}", m_decon_data.rows(), m_decon_data.cols());

    return true;
}

GenKokkos::ImpactTransform::~ImpactTransform() {}

Waveform::realseq_t GenKokkos::ImpactTransform::waveform(int iwire) const
{
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

            // nlength = nsamples;

            //   std::cout << nlength << " " << nsamples + m_pir->closest(0)->long_aux_waveform_pad() << std::endl;

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
