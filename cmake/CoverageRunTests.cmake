# CoverageRunTests.cmake — run ctest, tolerate failures
# Expects: CTEST, BINARY_DIR

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
        "LLVM_PROFILE_FILE=${BINARY_DIR}/profraw/test-%p-%m.profraw"
        ${CTEST} --test-dir ${BINARY_DIR} --output-on-failure
    RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
    message(WARNING "Some tests failed (exit ${_rc}) — generating coverage from passing tests")
endif()
