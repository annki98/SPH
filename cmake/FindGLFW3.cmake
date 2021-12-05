#
# Try to find GLFW3 library and include path.
# Once done this will define
#
# GLFW3_FOUND
# GLFW3_INCLUDE_DIR
# GLFW3_LIBRARIES
# 

SET(GLFW3_ROOT_ENV $ENV{GLFW3_ROOT})
IF (GLFW3_ROOT_ENV)
    file(TO_CMAKE_PATH ${GLFW3_ROOT_ENV} GLFW3_ROOT_ENV)
ENDIF ()

# For older cmake versions use root variable as additional search directory explicitly
IF (${CMAKE_VERSION} VERSION_LESS_EQUAL "3.12.0")
    SET(GLFW_ADDITIONAL_SEARCH_DIRS ${GLFW3_ROOT_ENV})
ENDIF ()


if (WIN32)

    IF (MINGW)

        SET(GLFW_SEARCH_PATH_SUFFIX mingw)
        SET(GLFW_LIB_NAME libglfw3)

    ELSEIF (MSVC)

        # MSVC toolset suffix
        # Checks for version
        IF (MSVC_TOOLSET_VERSION EQUAL 140 OR MSVC_VERSION EQUAL 1900)

            SET(GLFW_SEARCH_PATH_SUFFIX "msvc140")
        ELSEIF (MSVC_TOOLSET_VERSION EQUAL 141 OR
        (MSVC_VERSION GREATER_EQUAL 1910 AND MSVC_VERSION LESS_EQUAL 1919))
            SET(GLFW_SEARCH_PATH_SUFFIX "msvc141")
        ELSE ()
            SET(GLFW_SEARCH_PATH_SUFFIX "") # good luck
        ENDIF ()

        SET(GLFW_LIB_NAME glfw3)


    ENDIF ()

    find_library(GLFW3_LIBRARY_RELEASE
            NAMES ${GLFW_LIB_NAME}
            PATHS
            ${GLFW_ADDITIONAL_SEARCH_DIRS}
            PATH_SUFFIXES
            ${GLFW_SEARCH_PATH_SUFFIX}
            ${GLFW_SEARCH_PATH_SUFFIX}/lib
            )
    find_library(GLFW3_LIBRARY_DEBUG
            NAMES "${GLFW_LIB_NAME}d"
            PATHS
            ${GLFW_ADDITIONAL_SEARCH_DIRS}
            PATH_SUFFIXES
            ${GLFW_SEARCH_PATH_SUFFIX}
            ${GLFW_SEARCH_PATH_SUFFIX}/lib
            )

    FIND_PATH(GLFW3_INCLUDE_DIR GLFW/glfw3.h
            PATHS
            ${GLFW_ADDITIONAL_SEARCH_DIRS}
            PATH_SUFFIXES ${GLFW_SEARCH_PATH_SUFFIX}/include
            )

    include(SelectLibraryConfigurations)
    select_library_configurations(GLFW3)


    if (NOT GLFW3_LIBRARY_RELEASE)
        set(GLFW3_LIBRARY_RELEASE "GLFW3_LIBRARY_RELEASE-NOTFOUND" CACHE FILEPATH "Path to a library.")
    endif ()
    if (NOT GLFW3_LIBRARY_DEBUG)
        set(GLFW3_LIBRARY_DEBUG "GLFW3_LIBRARY_DEBUG-NOTFOUND" CACHE FILEPATH "Path to a library.")
    endif ()

    get_property(_isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if (GLFW3_LIBRARY_DEBUG AND GLFW3_LIBRARY_RELEASE AND
            NOT GLFW3_LIBRARY_DEBUG STREQUAL GLFW3_LIBRARY_RELEASE AND
    (_isMultiConfig OR CMAKE_BUILD_TYPE))
        # if the generator is multi-config or if CMAKE_BUILD_TYPE is set for
        # single-config generators, set optimized and debug libraries
        set(GLFW3_INTERFACE_LIBRARY "")
        foreach (_libname IN LISTS GLFW3_LIBRARY_RELEASE)
            list(APPEND GLFW3_INTERFACE_LIBRARY $<$<CONFIG:RelWithDebInfo>:${_libname}>)
            list(APPEND GLFW3_INTERFACE_LIBRARY $<$<CONFIG:Release>:${_libname}>)
            list(APPEND GLFW3_INTERFACE_LIBRARY $<$<CONFIG:MinSizeRelease>:${_libname}>)
            #list( APPEND GLFW3_LIBRARY ${_libname} )
        endforeach ()
        foreach (_libname IN LISTS GLFW3_LIBRARY_DEBUG)
            list(APPEND GLFW3_INTERFACE_LIBRARY $<$<CONFIG:Debug>:${_libname}>)
        endforeach ()
    elseif (GLFW3_LIBRARY_RELEASE)
        set(GLFW3_INTERFACE_LIBRARY ${GLFW3_LIBRARY_RELEASE})
    elseif (GLFW3_LIBRARY_DEBUG)
        set(GLFW3_INTERFACE_LIBRARY ${GLFW3_LIBRARY_DEBUG})
    else ()
        set(GLFW3_INTERFACE_LIBRARY "GLFW3_LIBRARY-NOTFOUND")
    endif ()


ELSEIF (APPLE)

    FIND_PATH(GLFW3_INCLUDE_DIR GLFW/glfw3.h DOC "Path to GLFW include directory."
            HINTS ${GLFW3_ROOT_ENV}/include
            PATHS /usr/include /usr/local/include /opt/local/include
            )

    FIND_LIBRARY(GLFW3_LIBRARY
            NAMES libglfw3.a glfw libglfw3.dylib
            PATHS $ENV{GLFW3_ROOT_ENV}/build/src /usr/lib /usr/local/lib /opt/local/lib
            )

    SET(GLFW3_LIBRARIES ${GLFW3_LIBRARY})
ELSE ()
    FIND_PATH(GLFW3_INCLUDE_DIR GLFW/glfw3.h
            PATHS
            ${GLFW_ADDITIONAL_SEARCH_DIRS})
    FIND_LIBRARY(GLFW3_LIBRARY
            PATHS
            ${GLFW_ADDITIONAL_SEARCH_DIRS}
            NAMES glfw3 glfw
            PATH_SUFFIXES dynamic)

    SET(GLFW3_LIBRARIES ${GLFW3_LIBRARY})
ENDIF ()


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLFW3
        REQUIRED_VARS GLFW3_INCLUDE_DIR GLFW3_LIBRARY)
set(GLFW3_INCLUDE_DIRS ${GLFW3_INCLUDE_DIR})


if (GLFW3_FOUND AND NOT TARGET GLFW3::glfw3)
    add_library(GLFW3::glfw3 INTERFACE IMPORTED)

    IF (NOT GLFW3_INTERFACE_LIBRARY)
        SET(GLFW3_INTERFACE_LIBRARY ${GLFW3_LIBRARIES})
    ENDIF ()
    set_target_properties(GLFW3::glfw3 PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${GLFW3_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${GLFW3_INTERFACE_LIBRARY}"
            )
endif ()
