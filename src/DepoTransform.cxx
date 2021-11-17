/* Comentary:

   This class is one element in a 2 by 2 matrix of classes which start
   very symmetric.

   (Depo, Impact) outer-product (Zipper, Transform)

   The "Depo" class is high level and uses the low level "Impact" class.

   The "Transform" version is newer than the "Zipper" version and
   while it started with a symmetric interface, it must grow out of
   it.

   The "Transform" version is faster than the "Zipper" for most
   realistically complicated event topology.  It will be made faster
   yet but this will break the symmetry:

   1) The "Tranform" version builds a dense array so sending its
   output through the narrow waveform() is silly.

   2) It can be optimized if the array has a channel basis instead of
   a wire one so the "zipper" interface is not fitting.

   3) It may be further optimized by performing its transforms on two
   planes from each face (for two-faced APAs) simultaneously.  This
   requires a different intrerface, possibly by (ab)using
   BinnedDiffusion in a different way than "zipper".

   4) It may be optimized yet further by breaking up the FFTs to do
   smaller ones (in time) on charge (X) FR and then doing the larger
   ones over ER(X)RC(X)RC.  This work also requires extension of PIR.
   This improvement may also be applicable to zipper.

   All of this long winded explantion, which nobody will ever read, is
   "justification" for the otherwise disgusting copy-paste which is
   going on with these two pairs of classes.

 */

#include "WireCellGenKokkos/DepoTransform.h"
#include "WireCellGenKokkos/ImpactTransform.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellIface/IAnodePlane.h"
#include "WireCellIface/SimpleTrace.h"
#include "WireCellIface/SimpleFrame.h"
#include "WireCellGenKokkos/BinnedDiffusion_transform.h"
#include "WireCellUtil/Units.h"
#include "WireCellUtil/Point.h"

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>

WIRECELL_FACTORY(GenKokkosDepoTransform, WireCell::GenKokkos::DepoTransform, WireCell::IDepoFramer, WireCell::IConfigurable)

using namespace WireCell;
using namespace std;

GenKokkos::DepoTransform::DepoTransform()
  : m_start_time(0.0 * units::ns)
  , m_readout_time(5.0 * units::ms)
  , m_tick(0.5 * units::us)
  , m_drift_speed(1.0 * units::mm / units::us)
  , m_nsigma(3.0)
  , m_frame_count(0)
  , m_transform("transform_vector")
  , l(Log::logger("sim"))
{
}

GenKokkos::DepoTransform::~DepoTransform() {}

void GenKokkos::DepoTransform::configure(const WireCell::Configuration& cfg)
{
    auto anode_tn = get<string>(cfg, "anode", "");
    m_anode = Factory::find_tn<IAnodePlane>(anode_tn);

    m_nsigma = get<double>(cfg, "nsigma", m_nsigma);
    bool fluctuate = get<bool>(cfg, "fluctuate", false);
    m_rng = nullptr;
    if (fluctuate) {
        auto rng_tn = get<string>(cfg, "rng", "");
        m_rng = Factory::find_tn<IRandom>(rng_tn);
    }

    m_readout_time = get<double>(cfg, "readout_time", m_readout_time);
    m_tick = get<double>(cfg, "tick", m_tick);
    m_start_time = get<double>(cfg, "start_time", m_start_time);
    m_drift_speed = get<double>(cfg, "drift_speed", m_drift_speed);
    m_frame_count = get<int>(cfg, "first_frame_number", m_frame_count);

    auto jpirs = cfg["pirs"];
    if (jpirs.isNull() or jpirs.empty()) {
        std::string msg = "must configure with some plane impact response components";
        l->error(msg);
        THROW(ValueError() << errmsg{"Gen::Ductor: " + msg});
    }
    m_pirs.clear();
    for (auto jpir : jpirs) {
        auto tn = jpir.asString();
        auto pir = Factory::find_tn<IPlaneImpactResponse>(tn);
        m_pirs.push_back(pir);
    }

    m_transform = get<string>(cfg, "transform", m_transform);
}
WireCell::Configuration GenKokkos::DepoTransform::default_configuration() const
{
    Configuration cfg;

    /// How many Gaussian sigma due to diffusion to keep before truncating.
    put(cfg, "nsigma", m_nsigma);

    /// Whether to fluctuate the final Gaussian deposition.
    put(cfg, "fluctuate", false);

    /// The open a gate.  This is actually a "readin" time measured at
    /// the input ("reference") plane.
    put(cfg, "start_time", m_start_time);

    /// The time span for each readout.  This is actually a "readin"
    /// time span measured at the input ("reference") plane.
    put(cfg, "readout_time", m_readout_time);

    /// The sample period
    put(cfg, "tick", m_tick);

    /// The nominal speed of drifting electrons
    put(cfg, "drift_speed", m_drift_speed);

    /// Allow for a custom starting frame number
    put(cfg, "first_frame_number", m_frame_count);

    /// Name of component providing the anode plane.
    put(cfg, "anode", "");
    /// Name of component providing the anode pseudo random number generator.
    put(cfg, "rng", "");

    /// Plane impact responses
    cfg["pirs"] = Json::arrayValue;

    /// Transform function to use
    put(cfg, "transform", m_transform);

    return cfg;
}

bool GenKokkos::DepoTransform::operator()(const input_pointer& in, output_pointer& out)
{
    Kokkos::Timer timer0 ;
    if (!in) {
        out = nullptr;
        return true;
    }
    
    auto depos = in->depos();

    Binning tbins(m_readout_time / m_tick, m_start_time, m_start_time + m_readout_time);
    ITrace::vector traces;
    for (auto face : m_anode->faces()) {
        // Select the depos which are in this face's sensitive volume
        IDepo::vector face_depos, dropped_depos;
        auto bb = face->sensitive();
        if (bb.empty()) {
            l->debug("anode {} face {} is marked insensitive, skipping", m_anode->ident(), face->ident());
            continue;
        }

        for (auto depo : (*depos)) {
            if (bb.inside(depo->pos())) {
                face_depos.push_back(depo);
            }
            else {
                dropped_depos.push_back(depo);
            }
        }

        if (face_depos.size()) {
            auto ray = bb.bounds();
            l->debug(
                "anode: {}, face: {}, processing {} depos spanning "
                "t:[{},{}]ms, bb:[{}-->{}]cm",
                m_anode->ident(), face->ident(), face_depos.size(), face_depos.front()->time() / units::ms,
                face_depos.back()->time() / units::ms, ray.first / units::cm, ray.second / units::cm);
        }
        if (dropped_depos.size()) {
            auto ray = bb.bounds();
            l->debug(
                "anode: {}, face: {}, dropped {} depos spanning "
                "t:[{},{}]ms, outside bb:[{}-->{}]cm",
                m_anode->ident(), face->ident(), dropped_depos.size(), dropped_depos.front()->time() / units::ms,
                dropped_depos.back()->time() / units::ms, ray.first / units::cm, ray.second / units::cm);
        }

        int iplane = -1;
        for (auto plane : face->planes()) {
            double t0 = timer0.seconds() ;
            ++iplane;

            const Pimpos* pimpos = plane->pimpos();

            Binning tbins(m_readout_time / m_tick, m_start_time, m_start_time + m_readout_time);

            GenKokkos::BinnedDiffusion_transform bindiff(*pimpos, tbins, m_nsigma, m_rng);
            for (auto depo : face_depos) {
                depo = modify_depo(plane->planeid(), depo);
                bindiff.add(depo, depo->extent_long() / m_drift_speed, depo->extent_tran());
            }
            double t1 = timer0.seconds() ;


            auto& wires = plane->wires();

            auto pir = m_pirs.at(iplane);
            GenKokkos::ImpactTransform transform(pir, bindiff);
            if(m_transform.compare("transform_vector")==0) {
                transform.transform_vector();
            } else if (m_transform.compare("transform_matrix")==0) {
                transform.transform_matrix();
            } else {
                THROW(ValueError() << errmsg{"No transform function named: " + m_transform});
            }
            double t2 = timer0.seconds() ;

            const int nwires = pimpos->region_binning().nbins();
            std::cout<<"nwires: "<<nwires << " p1: " << t1-t0 << " p2: "<< t2-t1<< std::endl ;
            Kokkos::Timer timer ;

            for (int iwire = 0; iwire < nwires; ++iwire) {
                auto wave = transform.waveform(iwire);

                auto mm = Waveform::edge(wave);
                if (mm.first == (int) wave.size()) {  // all zero
                    continue;
                }

                int chid = wires[iwire]->channel();
                int tbin = mm.first;

                ITrace::ChargeSequence charge(wave.begin() + mm.first, wave.begin() + mm.second);
                auto trace = make_shared<SimpleTrace>(chid, tbin, charge);
                traces.push_back(trace);
            }
            std::cout<< "Nwire loop time: "<<timer.seconds()<<std::endl ;
        }
    }

    auto frame = make_shared<SimpleFrame>(m_frame_count, m_start_time, traces, m_tick);
    ++m_frame_count;
    out = frame;
    std::cout<<"Depotransform::Operator() time: "<<timer0.seconds()<<std::endl ;
    return true;
}
