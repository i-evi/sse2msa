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
