
add_library(mpp_utils STATIC
    mpi_enc_utils.c
    mpp_opt.c
    utils.c
    dictionary.c
    iniparser.c
    )

target_link_libraries(mpp_utils ${MPP_LIB})
