#
# Try to find GLM library and include path.
# Once done this will define
#
# GLM_FOUND
# GLM_INCLUDE_PATH
# 
SET(GLM_ROOT_ENV $ENV{GLM_ROOT})
IF (GLM_ROOT_ENV)
    file(TO_CMAKE_PATH ${GLM_ROOT_ENV} GLM_ROOT_ENV)
ENDIF ()

FIND_PATH(GLM_INCLUDE_PATH glm/glm.hpp
        PATHS
        ${GLM_ROOT_ENV}/include
        )

SET(GLM_FOUND "NO")
IF (GLM_INCLUDE_PATH)
    SET(GLM_FOUND "YES")
    message("EXTERNAL LIBRARY 'GLM' FOUND")
ELSE ()
    message("ERROR: EXTERNAL LIBRARY 'GLM' NOT FOUND")
ENDIF (GLM_INCLUDE_PATH)

if (GLM_FOUND AND NOT TARGET GLM::glm)
    add_library(GLM::glm INTERFACE IMPORTED)
    set_target_properties(GLM::glm PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${GLM_INCLUDE_PATH}"
            )
endif ()
