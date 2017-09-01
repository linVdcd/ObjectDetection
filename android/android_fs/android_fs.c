//
// Created by lin on 17-8-9.
//

#include <stdio.h>

#include "android_fs.h"
#include <errno.h>


static int android_read(void* cookie, char* buf, int size) {
    return AAsset_read((AAsset*)cookie, buf, size);
}

static int android_write(void* cookie, const char* buf, int size) {
    return EACCES; // can't provide write access to the apk
}

static fpos_t android_seek(void* cookie, fpos_t offset, int whence) {
    return AAsset_seek((AAsset*)cookie, offset, whence);
}

static int android_close(void* cookie) {
    AAsset_close((AAsset*)cookie);
    return 0;
}



FILE* android_fopen(const char* fname, const char* mode) {
    if(mode[0] == 'w') return NULL;

    AAsset* asset = AAssetManager_open(mgr, fname, 0);
    if(!asset) return NULL;

    return funopen(asset, android_read, android_write, android_seek, android_close);
}