#!/usr/bin/env python


TOP = '.'
APPNAME = 'Junk'

from waflib.extras import wcb
wcb.package_descriptions.append(("WCT", dict(
    incs=["WireCellUtil/Units.h"],
    libs=["WireCellUtil", "WireCellIface","gomp"], mandatory=True)))

def options(opt):
    opt.load("wcb")

def configure(cfg):

    cfg.load("wcb")

    # boost 1.59 uses auto_ptr and GCC 5 deprecates it vociferously.
    cfg.env.CXXFLAGS += ['-Wno-deprecated-declarations']
    cfg.env.CXXFLAGS += ['-Wall', '-Wno-unused-local-typedefs', '-Wno-unused-function']
    cfg.env.CXXFLAGS += ['-fopenmp']
    # cfg.env.CXXFLAGS += ['-Wpedantic', '-Werror']


def build(bld):
    bld.load('wcb')
    bld.smplpkg('WireCellGenKokkos', use='WCT SPDLOG JSONCPP BOOST EIGEN FFTW KOKKOS CUDA')
    bld.install_files('${PREFIX}/share/wirecell',
                        bld.path.ant_glob("cfg/pgrapher/common/**/*.jsonnet") +
                        bld.path.ant_glob("cfg/pgrapher/experiment/**/*.jsonnet"),
                        relative_trick=True)
