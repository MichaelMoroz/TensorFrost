#pragma once

#include <string>

namespace TensorFrost {

void StartRenderDocCapture();
std::string EndRenderDocCapture(bool launchReplayUI = false);
bool IsRenderDocAvailable();

}  // namespace TensorFrost
