//
// Created by lin on 17-8-9.
//

#ifndef NNPACK_DARKNET_ANDROID_FS_H
#define NNPACK_DARKNET_ANDROID_FS_H

#include <stdio.h>
#include <android/asset_manager.h>
AAssetManager* mgr;
FILE* android_fopen(const char* fname, const char* mode);
#endif //NNPACK_DARKNET_ANDROID_FS_H
