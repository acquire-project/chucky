# Nix.cmake — NixOS OpenGL driver RPATH helper
#
# set_nix_rpath(target ...)
#   Sets BUILD_RPATH and INSTALL_RPATH for NixOS OpenGL driver.
#   No-op on WIN32.

function(set_nix_rpath)
    if(WIN32)
        return()
    endif()
    foreach(tgt IN LISTS ARGN)
        set_target_properties(
            ${tgt}
            PROPERTIES
                BUILD_RPATH "/run/opengl-driver/lib"
                INSTALL_RPATH "/run/opengl-driver/lib"
        )
    endforeach()
endfunction()
