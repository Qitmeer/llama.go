# gzip_file.cmake — compress a file using CMake built-in (cross-platform)
# Usage: cmake -DINPUT=<src> -DOUTPUT=<dst.gz> -P gzip_file.cmake
#
# Uses file(ARCHIVE_CREATE) with FORMAT raw / COMPRESSION GZip (CMake 3.18+)
# to avoid depending on the system gzip tool (unavailable on Windows by default).

if(NOT INPUT OR NOT OUTPUT)
    message(FATAL_ERROR "gzip_file.cmake: INPUT and OUTPUT must be set")
endif()

file(ARCHIVE_CREATE
    OUTPUT "${OUTPUT}"
    PATHS "${INPUT}"
    FORMAT raw
    COMPRESSION GZip
    COMPRESSION_LEVEL 9
)
