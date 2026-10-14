#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "postproc_common.h"

/* [2]: different scenes
 * [5]: different labels
 */
LabelName label_names[2][5] = {
    { LABEL_PERSON, LABEL_BOOK, LABEL_BUTT, LABEL_BUTT, LABEL_BUTT },
    { LABEL_PERSON, LABEL_BICYCLE, LABEL_CAR, LABEL_MOTORCYCLE, LABEL_BUS }
};

char *labels[OBJ_CLASS_NUM];

char *coco_cls_to_name(int cls_id)
{
    if (cls_id >= OBJ_CLASS_NUM) {
        return NULL;
    }

    if (labels[cls_id]) {
        return labels[cls_id];
    }

    return NULL;
}

static char *readLine(FILE *fp, char *buffer, int *len)
{
    int ch;
    int i = 0;
    size_t buff_len = 0;

    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL; // Out of memory

    while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL) {
            free(buffer);
            return NULL; // Out of memory
        }
        buffer = (char *)tmp;

        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';

    *len = buff_len;

    // Detect end
    if (ch == EOF && (i == 0 || ferror(fp))) {
        free(buffer);
        return NULL;
    }

    return buffer;
}

static int readLines(const char *fileName, char *lines[], int max_line)
{
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;

    if (file == NULL) {
        printf("Open %s fail!\n", fileName);
        return -1;
    }

    while ((s = readLine(file, s, &n)) != NULL) {
        lines[i++] = s;
        if (i >= max_line)
            break;
    }
    fclose(file);

    return i;
}

int loadLabelName(const char *locationFilename, char *label[])
{
    printf("load label %s\n", locationFilename);
    readLines(locationFilename, label, OBJ_CLASS_NUM);

    return 0;
}
