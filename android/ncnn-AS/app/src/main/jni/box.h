#ifndef BOX_H
#define BOX_H

typedef struct{
    float x, y, w, h;
} box;
typedef struct{
    float dx, dy, dw, dh;
} dbox;
typedef struct{
    int index;
    int classc;
    float **probs;
} sortable_bbox;
;
dbox diou(box a, box b);
void do_nms_obj(box *boxes, float **probs, int total, int classes, float thresh);


#endif

