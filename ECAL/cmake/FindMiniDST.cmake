# FindMiniDST.cmake
# Finds the MiniDST library and headers.
#
# Sets:
#   MiniDST_FOUND
#   MiniDST_INCLUDE_DIRS
#   MiniDST_LIBRARIES
#   MiniDST_VERSION (if version.h exists)
#
# Hints (set these on the cmake command line or in your environment):
#   MiniDST_ROOT   - root of a MiniDST installation  (e.g. /data/HPS)
#   MINIDST_DIR    - same idea, alternative variable

# Collect candidate search paths from hints, environment, and common prefixes
set(_MiniDST_search_roots
    ${MiniDST_ROOT}
    $ENV{MiniDST_ROOT}
    ${MINIDST_DIR}
    $ENV{MINIDST_DIR}
    /net/home/maurik/$ENV{OS_DISTRO}/
    /data/HPS
    /net/data/endeavour2/HPS
)

find_path(MiniDST_INCLUDE_DIR
    NAMES MiniDst.h
    PATHS ${_MiniDST_search_roots}
    PATH_SUFFIXES include
    DOC "MiniDST include directory"
)

find_library(MiniDST_LIBRARY
    NAMES MiniDST MiniDst
    PATHS ${_MiniDST_search_roots}
    PATH_SUFFIXES lib lib64
    DOC "MiniDST library"
)

# Standard CMake boilerplate to set MiniDST_FOUND etc.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MiniDST
    REQUIRED_VARS MiniDST_LIBRARY MiniDST_INCLUDE_DIR
)

if(MiniDST_FOUND)
    set(MiniDST_INCLUDE_DIRS ${MiniDST_INCLUDE_DIR})
    set(MiniDST_LIBRARIES    ${MiniDST_LIBRARY})

    # Create an imported target so consumers can just do:
    #   target_link_libraries(mytarget MiniDST::MiniDST)
    if(NOT TARGET MiniDST::MiniDST)
        add_library(MiniDST::MiniDST SHARED IMPORTED)
        set_target_properties(MiniDST::MiniDST PROPERTIES
            IMPORTED_LOCATION             "${MiniDST_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${MiniDST_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(MiniDST_INCLUDE_DIR MiniDST_LIBRARY)
