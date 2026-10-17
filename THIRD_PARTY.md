# Third-party components

The existing MIT license covers repository-authored source. Vendor SDK components have their own terms and are installed separately. This fork does not add SDK archives or shared libraries to Git. The XDNN payload formerly present under `third_party/xdnn/` is removed from the current tree; it remains in the preserved upstream history.

## Tested KL1 XDNN

Kunlunxin XDNN for Ubuntu x86_64, package date 20220919. Set `XDNN` to its external directory, containing `include/` and `so/`. The exact upstream version is also available from the sam6321/KunlunK200 base commit `5ead5bb10ab6d747e0f33b739826f111e42ec06a`.

| Component | Bytes | SHA256 |
|---|---:|---|
| Original `xdnn-ubuntu_x86_64.tar.gz` (20220919) | 16434613 | `d5ba7b20a1217dd78c7f438aec7ad74c83400c9b241f0b1f725b3037ed72a922` |
| `so/libxpuapi.so` | 60728800 | `e97dc7d2ce0f34df8d6584d1e9eb49e22344ce55c2cd2dcc9ff0e9ca88bff0f5` |

This library exports the KL1 `xdnn::fc` and vintage INT8/INT16 entry points used by the original benchmarks. Do not silently substitute a different ABI.

## XRE and XTDK

The current library is tested with XRE 4.33.0 and native x86 XTDK 2.7.98.2. The XTDK archive name includes LLVM15, but the device compiler in this package reports clang 10.0.1; the kernel build uses that package's resource headers and builtins together.

* [XRE 4.33](https://klx-sdk-release-public.su.bcebos.com/xre/release/4.33.0.1/xre-ubuntu_2004_x86_64.tar.gz): SHA256 `fef220c80fae1a6cc5cb4df5509c345686e29820fac65a46686ad8eb7d8697f5`.
* [Native XTDK 2.7.98.2](https://klx-sdk-release-public.su.bcebos.com/xtdk_llvm15/release_paddle/2.7.98.2/xtdk-llvm15-ubuntu1604_x86_64.tar.gz).
* The upstream legacy QEMU scripts use [XPU SDK 2.0.0.61](https://paddle-wheel.bj.bcebos.com/kunlun/xpu_sdk_v2.0.0.61.tar.gz): SHA256 `f67da36385e0f436a7f0c27135be5f39e6d8ddac0af658b474843b1a85438e18`.

The kernel module is built from the separate mobius/kunlun_k200 fork, with Ubuntu 26.04 compatibility changes. The DKMS module bundled in the XRE archive is not the validated driver. No SDK, model license, or vendor binary is relicensed under this repository's MIT license.
