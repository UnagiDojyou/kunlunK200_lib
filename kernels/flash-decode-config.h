#pragma once
#include "flash-config.h"
struct FlashDecodeConfig {KunlunFlashConfig p;int chunks; int chunk_keys; int reserved[7];};

static_assert(sizeof(FlashDecodeConfig)==192,"Flash-Decoding private metadata layout");
