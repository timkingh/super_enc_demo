#ifndef __SUPER_ENC_COMMON_H__
#define __SUPER_ENC_COMMON_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef enum {
    SOC_RK3576 = 0,
    SOC_RK3588 = 1,
    SOC_RK3572 = 2,
    SOC_RV1126B = 3,
    SOC_BUTT
} SocName;

#define SE_FREE(ptr)   do { if(ptr) free(ptr); ptr = NULL; } while (0)
#define FPRINT(fp, ...)  do { if (fp) { fprintf(fp, ## __VA_ARGS__); } }while(0)
#define FCLOSE(fp)  do { if(fp)  fclose(fp); fp = NULL; } while (0)

/* CME (Coarse Motion Estimation) CU data per CTU */
typedef struct {
    int cu_x;
    int cu_y;
    int cu_size;
    int mv_x;
    int mv_y;
    int sad_b16;
} CmeCuInfo;

/* CME per-frame data, organized in CTU raster scan order */
typedef struct {
    int ctu_total;          /* total CTU count = w_in_ctu * h_in_ctu */
    CmeCuInfo *cu_data;     /* array of ctu_total entries, CTU raster scan order */
} CmeFrameData;

#endif // __SUPER_ENC_COMMON_H__