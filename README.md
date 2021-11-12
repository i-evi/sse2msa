# sse2msa
A C/C++ header file that converts Intel SSE intrinsics to MIPS/MIPS64 MSA intrinsics.

## Introduction

Inspired by [sse2neon](https://github.com/DLTcollab/sse2neon), `sse2msa` translates Intel SSE (Streaming SIMD Extensions) intrinsics to MIPS MSA.

## Mapping and Coverage

|Header file | Extension |
|---|---|
|`<mmintrin.h>` | MMX |
|`<xmmintrin.h>` | SSE |
|`<emmintrin.h>` | SSE2 |
|`<pmmintrin.h>` | SSE3 |
|`<tmmintrin.h>` | SSSE3 |
|`<smmintrin.h>` | SSE4.1 |
|`<nmmintrin.h>` | SSE4.2 |

`sse2msa` aims to support SSE, SSE2, SSE3, SSSE3, SSE4.1 and SSE4.2 extension.

## Example

The header file `sse2msa.h` provides "SSE intrinsics" implemented with MSA intrinsics, on MIPS/MIPS64 targets, here's a example:

```c
#include <stdio.h>

#if defined(__x86_64)
#include <xmmintrin.h>
#elif defined(__mips)
#include "sse2msa.h"
#endif

#define NELEM_F32 (sizeof(__m128) / sizeof(float))

int main()
{
	float out[NELEM_F32];
	__m128 a = _mm_set_ps1(1.0);
	_mm_storeu_ps(out, _mm_add_ps(a, a));
	for (int i = 0; i < NELEM_F32; ++i) {
		printf("%f\t", out[i]);
	}
	putc('\n', stdout);
	return 0;
}
```

On MIPS/MIPS64 targets, append the following compiler option:

```bash
  -mmsa
```


## Related Projects
* [sse2neon](https://github.com/DLTcollab/sse2neon): A C/C++ header file that converts Intel SSE intrinsics to Arm/Aarch64 NEON intrinsics.
* [SIMDe](https://github.com/simd-everywhere/simde): Fast and portable implementations of SIMD
  intrinsics on hardware which doesn't natively support them, such as calling SSE functions on ARM.

## Reference
* [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)
* [The MIPS32 SIMD Architecture Module](https://www.mips.com/?do-download=the-mips32-simd-architecture-module)
* [The MIPS64 SIMD Architecture Module](https://www.mips.com/?do-download=the-mips64-simd-architecture-module)
* [MIPS SIMD Programming White Paper](https://www.mips.com/?do-download=mips-simd-programming-white-paper)
## Licensing

`sse2msa` is freely redistributable under the MIT License.
