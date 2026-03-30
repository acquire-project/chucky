# CoverageReport.cmake — invoked by the `coverage` target at build time
#
# Expects: LLVM_COV, PROFDATA_FILE, COVERAGE_DIR, BINARY_DIR, SOURCE_DIR

# Discover all test executables (named test_*)
file(GLOB_RECURSE _test_exes
    "${BINARY_DIR}/tests/test_*"
)
# Filter to actual executables (skip .profraw, .o, .d, etc.)
set(_exes)
foreach(_f ${_test_exes})
    if(IS_DIRECTORY "${_f}")
        continue()
    endif()
    # Skip files with extensions — executables on Linux have none
    get_filename_component(_ext "${_f}" EXT)
    if(_ext)
        continue()
    endif()
    # Verify it's executable
    execute_process(
        COMMAND test -x "${_f}"
        RESULT_VARIABLE _rc
    )
    if(_rc EQUAL 0)
        list(APPEND _exes "${_f}")
    endif()
endforeach()

if(NOT _exes)
    message(FATAL_ERROR "No test executables found in ${BINARY_DIR}/tests/")
endif()

# Build -object flags: first exe is positional, rest are -object args
list(POP_FRONT _exes _first)
set(_object_args)
foreach(_e ${_exes})
    list(APPEND _object_args "-object" "${_e}")
endforeach()

# Generate HTML report
execute_process(
    COMMAND ${LLVM_COV} show
        "${_first}"
        ${_object_args}
        "-instr-profile=${PROFDATA_FILE}"
        -format=html
        "-output-dir=${COVERAGE_DIR}"
        -ignore-filename-regex=tests/.*
    RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "llvm-cov show failed (${_rc})")
endif()

# Also print a summary to the terminal
execute_process(
    COMMAND ${LLVM_COV} report
        "${_first}"
        ${_object_args}
        "-instr-profile=${PROFDATA_FILE}"
        -ignore-filename-regex=tests/.*
)

message(STATUS "Coverage report: ${COVERAGE_DIR}/index.html")
