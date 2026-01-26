#ifndef __SUPER_ENC_COMMON_H__
#define __SUPER_ENC_COMMON_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef enum {
    SOC_RK3576 = 0,
    SOC_RK3588,
    SOC_BUTT
} SocName;

#define SE_FREE(ptr)   do { if(ptr) free(ptr); ptr = NULL; } while (0)
#define FPRINT(fp, ...)  do { if (fp) { fprintf(fp, ## __VA_ARGS__); } }while(0)
#define FCLOSE(fp)  do { if(fp)  fclose(fp); fp = NULL; } while (0)

#endif // __SUPER_ENC_COMMON_H__