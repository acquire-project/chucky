# Coverage.cmake — source-based code coverage with clang
#
# Creates: CHUCKY_ENABLE_COVERAGE option
# Provides: coverage custom target (runs tests, merges profiles, generates HTML)
#
# Usage:
#   cmake -B build -DCHUCKY_ENABLE_COVERAGE=ON -DCHUCKY_ENABLE_GPU=OFF
#   cmake --build build
#   cmake --build build --target coverage
#
# The HTML report is written to ${CMAKE_BINARY_DIR}/coverage/index.html

option(CHUCKY_ENABLE_COVERAGE "Enable code coverage instrumentation (clang only)" OFF)

if(NOT CHUCKY_ENABLE_COVERAGE)
    return()
endif()

if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "Coverage requires Clang (found ${CMAKE_C_COMPILER_ID})")
endif()

find_program(LLVM_PROFDATA llvm-profdata REQUIRED)
find_program(LLVM_COV llvm-cov REQUIRED)

# Add instrumentation flags to all C/CXX targets.
string(APPEND CMAKE_C_FLAGS   " -fprofile-instr-generate -fcoverage-mapping")
string(APPEND CMAKE_CXX_FLAGS " -fprofile-instr-generate -fcoverage-mapping")
string(APPEND CMAKE_EXE_LINKER_FLAGS    " -fprofile-instr-generate")
string(APPEND CMAKE_SHARED_LINKER_FLAGS " -fprofile-instr-generate")

set(PROFDATA_FILE "${CMAKE_BINARY_DIR}/coverage.profdata")
set(COVERAGE_DIR  "${CMAKE_BINARY_DIR}/coverage")

# Custom target: run tests, merge profiles, generate report.
add_custom_target(coverage
    # 1. Clean stale profile data
    COMMAND ${CMAKE_COMMAND} -E rm -f ${PROFDATA_FILE}
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/profraw"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/profraw"

    # 2. Run tests (LLVM_PROFILE_FILE controls where .profraw files land)
    #    Tolerates test failures — coverage is generated from whatever passes.
    COMMAND ${CMAKE_COMMAND}
        -DCTEST=${CMAKE_CTEST_COMMAND}
        -DBINARY_DIR=${CMAKE_BINARY_DIR}
        -P "${CMAKE_CURRENT_LIST_DIR}/CoverageRunTests.cmake"

    # 3. Merge raw profiles
    COMMAND ${LLVM_PROFDATA} merge -sparse
        "${CMAKE_BINARY_DIR}/profraw"
        -o ${PROFDATA_FILE}

    # 4. Generate HTML report (uses all test executables as object files)
    COMMAND ${CMAKE_COMMAND}
        -DLLVM_COV=${LLVM_COV}
        -DPROFDATA_FILE=${PROFDATA_FILE}
        -DCOVERAGE_DIR=${COVERAGE_DIR}
        -DBINARY_DIR=${CMAKE_BINARY_DIR}
        -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
        -P "${CMAKE_CURRENT_LIST_DIR}/CoverageReport.cmake"

    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running tests and generating coverage report..."
    VERBATIM
)

# Also add a summary-only target for quick terminal output.
add_custom_target(coverage-report
    COMMAND ${LLVM_COV} report
        -instr-profile=${PROFDATA_FILE}
        -ignore-filename-regex="tests/.*"
        -object "$<TARGET_FILE:test_index_ops>"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Showing coverage summary..."
    VERBATIM
)
