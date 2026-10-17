#include <kunlun/k200.h>
#include <cstdio>
#include <mutex>
#include <vector>

int main() {
    std::lock_guard<std::recursive_mutex> lock(kunlun::runtime_mutex());
    kunlun::Buffer buffer(0,4096);
    std::vector<unsigned char> input(4096,37), output(4096);
    kunlun::select(0);
    kunlun::check(kunlun::copy(buffer.data(),input.data(),input.size(),kunlun::host_to_device));
    kunlun::check(kunlun::wait());
    kunlun::check(kunlun::copy(output.data(),buffer.data(),output.size(),kunlun::device_to_host));
    kunlun::check(kunlun::wait());
    std::printf("KunlunK200 %s: %s\n",kunlun::version(),input == output ? "PASS" : "FAIL");
    return input == output ? 0 : 1;
}
