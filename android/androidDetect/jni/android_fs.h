//
// Created by lin on 17-8-9.
//

#ifndef NNPACK_DARKNET_ANDROID_FS_H
#define NNPACK_DARKNET_ANDROID_FS_H

#include <stdio.h>
#include <android/asset_manager.h>
AAssetManager* mgr;
#ifdef __cplusplus
extern "C"
#endif
FILE* android_fopen(const char* fname, const char* mode);
#ifdef __cplusplus
extern "C"
#endif
#endif //NNPACK_DARKNET_ANDROID_FS_H
