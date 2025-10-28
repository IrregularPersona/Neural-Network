#pragma once

namespace utility
{
  float newton_sqrt(float x) {
    if (x < 0)
      return -1;

    float res = (x > 1) ? x * 0.5f : 1.0f;

    for (int i = 0; i < 4; i++) {
      res = 0.5f * (res + x / res);
    }

    return res;
  }
}
