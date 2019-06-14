find_path(EIGEN3_DIR signature_of_eigen3_matrix_library
	  PATH_SUFFIXES include/eigen3
	  DOC "The path to Eigen3 headers")
mark_as_advanced(EIGEN3_DIR)
include_directories(${EIGEN3_DIR})
set(EIGEN3_LICENSE ${EIGEN3_DIR}/COPYING.LGPL)

#find_package(MKL)
#if(MKL_FOUND)
#  add_definitions(-DEIGEN_USE_MKL_ALL)
#  set(EIGEN3_LIBS ${MKL_LIBS})
#endif()
