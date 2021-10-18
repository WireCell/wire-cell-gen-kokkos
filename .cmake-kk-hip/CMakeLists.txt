cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(wire-cell-gen-kokkos CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS Off)

file(GLOB all_files ${PROJECT_SOURCE_DIR}/../src/*.cxx)
file(GLOB all_kk_files ${PROJECT_SOURCE_DIR}/../src/*.kokkos)



foreach(source_file IN LISTS all_kk_files)
  message(STATUS "Here ${source_file}")
  get_filename_component(exec_name ${source_file} NAME)
  set(renamed_source_file ${PROJECT_SOURCE_DIR}/src/${exec_name}.cxx)
  configure_file(${source_file} ${renamed_source_file} COPYONLY)
  list(APPEND all_files ${renamed_source_file})
  #add_executable(${exec_name} ${renamed_source_file})
  #target_link_libraries(${exec_name} PRIVATE Kokkos::kokkos)
endforeach()

include_directories(SYSTEM $ENV{BOOST_INC})

set(BOOST_ROOT $ENV{BOOST_DIR} )
set(BOOST_INCLUDEDIR $ENV{BOOST_INC})
set(BOOST_LIBRARYDIR $ENV{BOOST_LIB})
set(Boost_USE_MULTITHREADED ON)

find_package(Boost REQUIRED COMPONENTS)
find_package(Kokkos REQUIRED)

string(APPEND CMAKE_CXX_FLAGS " -g -O2 -pedantic -Wall")
string(APPEND CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

add_library(WireCellGenKokkos SHARED ${all_files})
target_include_directories(WireCellGenKokkos
  PRIVATE
    ${PROJECT_SOURCE_DIR}/../inc
    $ENV{EIGEN_INC}
    $ENV{JSONCPP_INC}
    $ENV{JSONNET_INC}
    $ENV{SPDLOG_INC}
    $ENV{WIRECELL_INC}
)
set_target_properties(WireCellGenKokkos
	PROPERTIES COMPILE_OPTIONS "-DEIGEN_NO_CUDA;-DEIGEN_DONT_VECTORIZE")

target_link_directories(WireCellGenKokkos PRIVATE $ENV{JSONCPP_LIB} $ENV{WIRECELL_LIB})
target_link_libraries(WireCellGenKokkos PRIVATE jsoncpp WireCellIface WireCellUtil Boost::headers Kokkos::kokkos)

add_subdirectory(test)
