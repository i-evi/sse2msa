#ifndef _COMMON_DEF_H_
#define _COMMON_DEF_H_

#ifdef __x86_64
	#define int64_t long long int
#endif

static int insts;
static int print_udef = 0;

static uint8_t buffer_a[BUFSIZ];
static void *vp_a = (void*)buffer_a;
static int32_t i32p_a[BUFSIZ >> 2];
static int64_t i64p_a[BUFSIZ >> 3];
static float f32p_a[BUFSIZ >> 2];
static double f64p_a[BUFSIZ >> 3];
static __m64 m64p_a[BUFSIZ >> 3];
static __m128i m128ip_a[BUFSIZ >> 4];
static int8_t _i8[BUFSIZ >> 0];
static int16_t _i16[BUFSIZ >> 1];
static int32_t _i32[BUFSIZ >> 2];
static int64_t _i64[BUFSIZ >> 3];
static uint32_t _u32[BUFSIZ >> 2];
static uint32_t _u64[BUFSIZ >> 3];
static float _f32[BUFSIZ >> 2];
static double _f64[BUFSIZ >> 3];
static __m64 _m64[BUFSIZ >> 3];
static __m128 _m128[BUFSIZ >> 4];
static __m128i _m128i[BUFSIZ >> 4];
static __m128d _m128d[BUFSIZ >> 4];

#define _immi8_0x1c 0x1c
#define _immi8_0x02 0x02
#define _immi8_0x01 0x01
#define _immi32_0x01 0x01

static void init_mem(void *p, size_t size)
{
	uint8_t *u = (uint8_t*)p;
	for (size_t i = 0; i < size; ++i) {
		u[i] = i;
	}
}

#define init_array(p) init_mem(p, sizeof(p))

static void test_data_init()
{
	init_array(buffer_a);
	init_array(i32p_a);
	init_array(i64p_a);
	init_array(f32p_a);
	init_array(f64p_a);
	init_array(m64p_a);
	init_array(m128ip_a);
	init_array(_i8);
	init_array(_i16);
	init_array(_i32);
	init_array(_i64);
	init_array(_u32);
	init_array(_u64);
	init_array(_f32);
	init_array(_f64);
	init_array(_m64);
	init_array(_m128);
	init_array(_m128i);
	init_array(_m128d);
}

#define PRINT128_HEX(p) \
do {                                  \
	uint8_t *u = (uint8_t*)p;     \
	for (int i = 0; i < 16; ++i)  \
		printf("%02x", u[i]); \
} while (0);

#define PRINT128_F32(p) \
do {                                         \
	float *u = (float*)p;                \
	for (int i = 0; i < 4; ++i) {        \
		if (isnan(u[i]))             \
			printf("NAN ");      \
		else                         \
			printf("%e ", u[i]); \
	}                                    \
} while (0);

#define PRINT128_F64(p) \
do {                                         \
	double *u = (double*)p;              \
	for (int i = 0; i < 4; ++i) {        \
		if (isnan(u[i]))             \
			printf("NAN ");      \
		else                         \
			printf("%e ", u[i]); \
	}                                    \
} while (0);

#define V_TEST(op) \
do {         \
	printf("%d: " # op, ++insts); \
	op();                         \
	printf("\t OK");              \
	putc('\n', stdout);           \
} while (0);

#define V_VP_I32(fmt, op) \
do {                                          \
	printf("%d: " # op, ++insts);         \
	putc('\t', stdout);                   \
	init_mem(buffer_a, sizeof(buffer_a)); \
	op(vp_a, _i32[0]);                    \
	PRINT128_ ## fmt(vp_a);               \
	putc('\n', stdout);                   \
} while(0);

#define V_VP_IMMI32(fmt, op, imm) \
do {                                          \
	printf("%d: " # op, ++insts);         \
	putc('\t', stdout);                   \
	init_mem(buffer_a, sizeof(buffer_a)); \
	op(vp_a, imm);                        \
	PRINT128_ ## fmt(vp_a);               \
	putc('\n', stdout);                   \
} while(0);

#define V_VP_M128I(fmt, op) \
do {                                          \
	printf("%d: " # op, ++insts);         \
	putc('\t', stdout);                   \
	init_mem(buffer_a, sizeof(buffer_a)); \
	op(vp_a, _m128i[0]);                  \
	PRINT128_ ## fmt(vp_a);               \
	putc('\n', stdout);                   \
} while(0);

#define V_I32P_I32(fmt, op) \
do {                                      \
	printf("%d: " # op, ++insts);     \
	putc('\t', stdout);               \
	init_mem(i32p_a, sizeof(i32p_a)); \
	op(i32p_a, _i32[0]);              \
	PRINT128_ ## fmt(i32p_a);         \
	putc('\n', stdout);               \
} while(0);

#define V_I64P_I64(fmt, op) \
do {                                      \
	printf("%d: " # op, ++insts);     \
	putc('\t', stdout);               \
	init_mem(i64p_a, sizeof(i64p_a)); \
	op(i64p_a, _i64[0]);              \
	PRINT128_ ## fmt(i64p_a);         \
	putc('\n', stdout);               \
} while(0);

#define V_F32P_M128(fmt, op) \
do {                                      \
	printf("%d: " # op, ++insts);     \
	putc('\t', stdout);               \
	init_mem(f32p_a, sizeof(f32p_a)); \
	op(f32p_a, _m128[0]);             \
	PRINT128_ ## fmt(f32p_a);         \
	putc('\n', stdout);               \
} while(0);

#define V_F64P_M128D(fmt, op) \
do {                                      \
	printf("%d: " # op, ++insts);     \
	putc('\t', stdout);               \
	init_mem(f64p_a, sizeof(f64p_a)); \
	op(f64p_a, _m128d[0]);            \
	PRINT128_ ## fmt(f64p_a);         \
	putc('\n', stdout);               \
} while(0);

#define V_M64P_M128(fmt, op) \
do {                                      \
	printf("%d: " # op, ++insts);     \
	putc('\t', stdout);               \
	init_mem(m64p_a, sizeof(m64p_a)); \
	op(m64p_a, _m128[0]);             \
	PRINT128_ ## fmt(m64p_a);         \
	putc('\n', stdout);               \
} while(0);

#define V_M64P_M64(fmt, op) \
do {                                      \
	printf("%d: " # op, ++insts);     \
	putc('\t', stdout);               \
	init_mem(m64p_a, sizeof(m64p_a)); \
	op(m64p_a, _m64[0]);              \
	PRINT128_ ## fmt(m64p_a);         \
	putc('\n', stdout);               \
} while(0);

#define V_M128IP_M128I(fmt, op) \
do {                                          \
	printf("%d: " # op, ++insts);         \
	putc('\t', stdout);                   \
	init_mem(m128ip_a, sizeof(m128ip_a)); \
	op(m128ip_a, _m128i[0]);              \
	PRINT128_ ## fmt(m128ip_a);           \
	putc('\n', stdout);                   \
} while(0);

#define I32_U32(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	int a[4];                     \
       	a[0] = op(_u32[0]);           \
       	a[1] = op(_u32[1]);           \
       	a[2] = op(_u32[2]);           \
       	a[3] = op(_u32[3]);           \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define I32_M64(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	int a[4];                     \
       	a[0] = op(_m64[0]);           \
       	a[1] = op(_m64[1]);           \
       	a[2] = op(_m64[2]);           \
       	a[3] = op(_m64[3]);           \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define I32_M64_IMMI8(fmt, op, imm) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	int a[4];                     \
       	a[0] = op(_m64[0], imm);      \
       	a[1] = op(_m64[1], imm);      \
       	a[2] = op(_m64[2], imm);      \
       	a[3] = op(_m64[3], imm);      \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define I32_M128(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	int a[4];                     \
       	a[0] = op(_m128[0]);          \
       	a[1] = op(_m128[1]);          \
       	a[2] = op(_m128[2]);          \
       	a[3] = op(_m128[3]);          \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define F32_M128(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	float a[4];                   \
       	a[0] = op(_m128[0]);          \
       	a[1] = op(_m128[1]);          \
       	a[2] = op(_m128[2]);          \
       	a[3] = op(_m128[3]);          \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define I32_M128D(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	int a[4];                     \
       	a[0] = op(_m128d[0]);         \
       	a[1] = op(_m128d[1]);         \
       	a[2] = op(_m128d[2]);         \
       	a[3] = op(_m128d[3]);         \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define I32_M128I(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	int a[4];                     \
       	a[0] = op(_m128i[0]);         \
       	a[1] = op(_m128i[1]);         \
       	a[2] = op(_m128i[2]);         \
       	a[3] = op(_m128i[3]);         \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define I32_M128I_IMMI8(fmt, op, imm) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	int a[4];                     \
       	a[0] = op(_m128i[0], imm);    \
       	a[1] = op(_m128i[1], imm);    \
       	a[2] = op(_m128i[2], imm);    \
       	a[3] = op(_m128i[3], imm);    \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define I32_M128_IMMI8(fmt, op, imm) \
do {                                   \
	printf("%d: " # op, ++insts);  \
	putc('\t', stdout);            \
	int a[4];                      \
       	a[0] = op(_m128[0], imm);      \
       	a[1] = op(_m128[1], imm);      \
       	a[2] = op(_m128[2], imm);      \
       	a[3] = op(_m128[3], imm);      \
	PRINT128_ ## fmt(a);           \
	putc('\n', stdout);            \
} while(0);

#define I32_M128_M128(fmt, op) \
do {                                   \
	printf("%d: " # op, ++insts);  \
	putc('\t', stdout);            \
	int a[4];                      \
       	a[0] = op(_m128[0], _m128[1]); \
       	a[1] = op(_m128[2], _m128[3]); \
       	a[2] = op(_m128[4], _m128[5]); \
       	a[3] = op(_m128[6], _m128[7]); \
	PRINT128_ ## fmt(a);           \
	putc('\n', stdout);            \
} while(0);

#define I32_M128I_M128I(fmt, op) \
do {                                     \
	printf("%d: " # op, ++insts);    \
	putc('\t', stdout);              \
	int a[4];                        \
       	a[0] = op(_m128i[0], _m128i[1]); \
       	a[1] = op(_m128i[2], _m128i[3]); \
       	a[2] = op(_m128i[4], _m128i[5]); \
       	a[3] = op(_m128i[6], _m128i[7]); \
	PRINT128_ ## fmt(a);             \
	putc('\n', stdout);              \
} while(0);

#define I32_M128D_M128D(fmt, op) \
do {                                     \
	printf("%d: " # op, ++insts);    \
	putc('\t', stdout);              \
	int a[4];                        \
       	a[0] = op(_m128d[0], _m128d[1]); \
       	a[1] = op(_m128d[2], _m128d[3]); \
       	a[2] = op(_m128d[4], _m128d[5]); \
       	a[3] = op(_m128d[6], _m128d[7]); \
	PRINT128_ ## fmt(a);             \
	putc('\n', stdout);              \
} while(0);

#define M128I_V(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128i a = op();             \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define UDEF_M128I_V(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128i a = op();             \
	if (print_udef) {             \
		PRINT128_ ## fmt(&a); \
	} else {                      \
		printf("OK");         \
	}                             \
	putc('\n', stdout);           \
} while(0);

#define M128_F32(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128 a = op(_f32[0]);       \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128D_F64(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128d a = op(_f64[0]);      \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128I_I8(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128i a = op(_i8[0]);       \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128I_I16(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128i a = op(_i16[0]);      \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128I_I32(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128i a = op(_i32[0]);      \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128I_I64(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128i a = op(_i64[0]);      \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128I_M64(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128i a = op(_m64[0]);      \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128_M64(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128 a = op(_m64[0]);       \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128_M64_M64(fmt, op) \
do {                                     \
	printf("%d: " # op, ++insts);    \
	putc('\t', stdout);              \
	__m128 a = op(_m64[0], _m64[1]); \
	PRINT128_ ## fmt(&a);            \
	putc('\n', stdout);              \
} while(0);

#define M128D_M64(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128d a = op(_m64[0]);      \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M64_M128(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m64 a[2];                   \
	a[0] = op(_m128[0]);          \
	a[1] = op(_m128[1]);          \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define M64_M128I(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m64 a[2];                   \
	a[0] = op(_m128i[0]);         \
	a[1] = op(_m128i[1]);         \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define M64_M128D(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m64 a[2];                   \
	a[0] = op(_m128d[0]);         \
	a[1] = op(_m128d[1]);         \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define F64_M128D(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	double a[2];                  \
	a[0] = op(_m128d[0]);         \
	a[1] = op(_m128d[1]);         \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define I64_M128(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	int64_t a[2];                 \
	a[0] = op(_m128[0]);          \
	a[1] = op(_m128[1]);          \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define I64_M128I(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	int64_t a[2];                 \
	a[0] = op(_m128i[0]);         \
	a[1] = op(_m128i[1]);         \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define I64_M128D(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	int64_t a[2];                 \
	a[0] = op(_m128d[0]);         \
	a[1] = op(_m128d[1]);         \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define I64_U64(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	int64_t a[2];                 \
	a[0] = op(_u64[0]);           \
	a[1] = op(_u64[1]);           \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define M128_M128_F32(op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128 a = op(_m128[0]);      \
	print_f32_128bit(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128_M128(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128 a = op(_m128[0]);      \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128_M128D(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128 a = op(_m128d[0]);     \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128_M128I(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128 a = op(_m128i[0]);     \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128D_M128I(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128d a = op(_m128i[0]);    \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128I_M128I(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128i a = op(_m128i[0]);    \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128I_M128(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128i a = op(_m128[0]);     \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128I_M128D(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128i a = op(_m128d[0]);    \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M64_M64(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m64 a[2];                   \
	a[0] = op(_m64[0]);           \
	a[1] = op(_m64[1]);           \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define M64_M64_M64(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m64 a[2];                   \
	a[0] = op(_m64[0], _m64[1]);  \
	a[1] = op(_m64[2], _m64[3]);  \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define M64_M64_I32_IMMI8(fmt, op, imm) \
do {                                      \
	printf("%d: " # op, ++insts);     \
	putc('\t', stdout);               \
	__m64 a[2];                       \
	a[0] = op(_m64[0], _i32[0], imm); \
	a[1] = op(_m64[1], _i32[1], imm); \
	PRINT128_ ## fmt(a);              \
	putc('\n', stdout);               \
} while(0);

#define M64_M64_M64_IMMI8(fmt, op, imm) \
do {                                      \
	printf("%d: " # op, ++insts);     \
	putc('\t', stdout);               \
	__m64 a[2];                       \
	a[0] = op(_m64[0], _m64[1], imm); \
	a[1] = op(_m64[2], _m64[3], imm); \
	PRINT128_ ## fmt(a);              \
	putc('\n', stdout);               \
} while(0);

#define M64_M64_IMMI8(fmt, op, imm) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m64 a[2];                   \
	a[0] = op(_m64[0], imm);      \
	a[1] = op(_m64[1], imm);      \
	PRINT128_ ## fmt(a);          \
	putc('\n', stdout);           \
} while(0);

#define M128_M128_I32(fmt, op) \
do {                                      \
	printf("%d: " # op, ++insts);     \
	putc('\t', stdout);               \
	__m128 a = op(_m128[0], _i32[0]); \
	PRINT128_ ## fmt(&a);             \
	putc('\n', stdout);               \
} while(0);

#define M128_M128_I64(fmt, op) \
do {                                      \
	printf("%d: " # op, ++insts);     \
	putc('\t', stdout);               \
	__m128 a = op(_m128[0], _i64[0]); \
	PRINT128_ ## fmt(&a);             \
	putc('\n', stdout);               \
} while(0);

#define M128D_M128D_I32(fmt, op) \
do {                                        \
	printf("%d: " # op, ++insts);       \
	putc('\t', stdout);                 \
	__m128d a = op(_m128d[0], _i32[0]); \
	PRINT128_ ## fmt(&a);               \
	putc('\n', stdout);                 \
} while(0);

#define M128D_M128D_I64(fmt, op) \
do {                                        \
	printf("%d: " # op, ++insts);       \
	putc('\t', stdout);                 \
	__m128d a = op(_m128d[0], _i64[0]); \
	PRINT128_ ## fmt(&a);               \
	putc('\n', stdout);                 \
} while(0);

#define M128I_M128I_IMMI8(fmt, op, imm) \
do {                                    \
	printf("%d: " # op, ++insts);   \
	putc('\t', stdout);             \
	__m128i a = op(_m128i[0], imm); \
	PRINT128_ ## fmt(&a);           \
	putc('\n', stdout);             \
} while(0);

#define M128I_M128I_I32_IMMI8(fmt, op, imm) \
do {                                            \
	printf("%d: " # op, ++insts);           \
	putc('\t', stdout);                     \
	__m128i a = op(_m128i[0], _i32[0],imm); \
	PRINT128_ ## fmt(&a);                   \
	putc('\n', stdout);                     \
} while(0);

#define M128_M128_M128_IMMI8(fmt, op, imm) \
do {                                            \
	printf("%d: " # op, ++insts);           \
	putc('\t', stdout);                     \
	__m128 a = op(_m128[0], _m128[1], imm); \
	PRINT128_ ## fmt(&a);                   \
	putc('\n', stdout);                     \
} while(0);

#define M128D_M128D_M128D_IMMI8(fmt, op, imm) \
do {                                               \
	printf("%d: " # op, ++insts);              \
	putc('\t', stdout);                        \
	__m128d a = op(_m128d[0], _m128d[1], imm); \
	PRINT128_ ## fmt(&a);                      \
	putc('\n', stdout);                        \
} while(0);

#define M128I_M128I_M128I_IMMI8(fmt, op, imm) \
do {                                               \
	printf("%d: " # op, ++insts);              \
	putc('\t', stdout);                        \
	__m128i a = op(_m128i[0], _m128i[1], imm); \
	PRINT128_ ## fmt(&a);                      \
	putc('\n', stdout);                        \
} while(0);

#define M128I_M128I_M128I_M128I(fmt, op) \
do {                                                     \
	printf("%d: " # op, ++insts);                    \
	putc('\t', stdout);                              \
	__m128i a = op(_m128i[0], _m128i[1], _m128i[2]); \
	PRINT128_ ## fmt(&a);                            \
	putc('\n', stdout);                              \
} while(0);

#define M128D_M128D_M128D_M128D(fmt, op) \
do {                                                     \
	printf("%d: " # op, ++insts);                    \
	putc('\t', stdout);                              \
	__m128d a = op(_m128d[0], _m128d[1], _m128d[2]); \
	PRINT128_ ## fmt(&a);                            \
	putc('\n', stdout);                              \
} while(0);

#define M128_M128_M128_M128(fmt, op) \
do {                                                 \
	printf("%d: " # op, ++insts);                \
	putc('\t', stdout);                          \
	__m128 a = op(_m128[0], _m128[1], _m128[2]); \
	PRINT128_ ## fmt(&a);                        \
	putc('\n', stdout);                          \
} while(0);

#define M128_M128_M64(fmt, op) \
do {                                      \
	printf("%d: " # op, ++insts);     \
	putc('\t', stdout);               \
	__m128 a = op(_m128[0], _m64[0]); \
	PRINT128_ ## fmt(&a);             \
	putc('\n', stdout);               \
} while(0);

#define M128_M128_M128(fmt, op) \
do {                                       \
	printf("%d: " # op, ++insts);      \
	putc('\t', stdout);                \
	__m128 a = op(_m128[0], _m128[1]); \
	PRINT128_ ## fmt(&a);              \
	putc('\n', stdout);                \
} while(0);

#define M128_M128_M128D(fmt, op) \
do {                                        \
	printf("%d: " # op, ++insts);       \
	putc('\t', stdout);                 \
	__m128 a = op(_m128[0], _m128d[1]); \
	PRINT128_ ## fmt(&a);               \
	putc('\n', stdout);                 \
} while(0);

#define M128I_M128I_M128I(fmt, op) \
do {                                          \
	printf("%d: " # op, ++insts);         \
	putc('\t', stdout);                   \
	__m128i a = op(_m128i[0], _m128i[1]); \
	PRINT128_ ## fmt(&a);                 \
	putc('\n', stdout);                   \
} while(0);

#define M128D_M128D_M128D(fmt, op) \
do {                                          \
	printf("%d: " # op, ++insts);         \
	putc('\t', stdout);                   \
	__m128d a = op(_m128d[0], _m128d[1]); \
	PRINT128_ ## fmt(&a);                 \
	putc('\n', stdout);                   \
} while(0);

#define M128D_M128D_M128(fmt, op) \
do {                                          \
	printf("%d: " # op, ++insts);         \
	putc('\t', stdout);                   \
	__m128d a = op(_m128d[0], _m128[0]);  \
	PRINT128_ ## fmt(&a);                 \
	putc('\n', stdout);                   \
} while(0);

#define M128I_M128I(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128i a = op(_m128i[0]);    \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128D_M128D(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128d a = op(_m128d[0]);    \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128D_M128D_F64(op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128d a = op(_m128d[0]);    \
	print_f64_128bit(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128D_M128(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128d a = op(_m128[0]);     \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define M128_V(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128 a = op();              \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define UDEF_M128_V(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128 a = op();              \
	if (print_udef) {             \
		PRINT128_ ## fmt(&a); \
	} else {                      \
		printf("OK");         \
	}                             \
	putc('\n', stdout);           \
} while(0);

#define M128D_V(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128d a = op();             \
	PRINT128_ ## fmt(&a);         \
	putc('\n', stdout);           \
} while(0);

#define UDEF_M128D_V(fmt, op) \
do {                                  \
	printf("%d: " # op, ++insts); \
	putc('\t', stdout);           \
	__m128d a = op();             \
	if (print_udef) {             \
		PRINT128_ ## fmt(&a); \
	} else {                      \
		printf("OK");         \
	}                             \
	putc('\n', stdout);           \
} while(0);

#define M128_F32x4(fmt, op) \
do {                                                       \
	printf("%d: " # op, ++insts);                      \
	putc('\t', stdout);                                \
	__m128 a = op(_f32[0], _f32[1], _f32[2], _f32[3]); \
	PRINT128_ ## fmt(&a);                              \
	putc('\n', stdout);                                \
} while(0);

#define M128D_F64x2(fmt, op) \
do {                                      \
	printf("%d: " # op, ++insts);     \
	putc('\t', stdout);               \
	__m128d a = op(_f64[0], _f64[1]); \
	PRINT128_ ## fmt(&a);             \
	putc('\n', stdout);               \
} while(0);

#define M128I_I8x16(fmt, op) \
do {                                                             \
	printf("%d: " # op, ++insts);                            \
	putc('\t', stdout);                                      \
	__m128i a = op(_i8[0], _i8[1], _i8[2], _i8[3], _i8[4],   \
		_i8[5], _i8[6], _i8[7], _i8[8], _i8[9], _i8[10], \
		_i8[11], _i8[12], _i8[13], _i8[14], _i8[15]);    \
	PRINT128_ ## fmt(&a);                                    \
	putc('\n', stdout);                                      \
} while(0);

#define M128I_I16x8(fmt, op) \
do {                                                          \
	printf("%d: " # op, ++insts);                         \
	putc('\t', stdout);                                   \
	__m128i a = op(_i16[0], _i16[1], _i16[2],             \
		_i16[3], _i16[4], _i16[5], _i16[6], _i16[7]); \
	PRINT128_ ## fmt(&a);                                 \
	putc('\n', stdout);                                   \
} while(0);

#define M128I_I32x4(fmt, op) \
do {                                                        \
	printf("%d: " # op, ++insts);                       \
	putc('\t', stdout);                                 \
	__m128i a = op(_i32[0], _i32[1], _i32[2], _i32[3]); \
	PRINT128_ ## fmt(&a);                               \
	putc('\n', stdout);                                 \
} while(0);

#define M128I_I64x2(fmt, op) \
do {                                      \
	printf("%d: " # op, ++insts);     \
	putc('\t', stdout);               \
	__m128i a = op(_i64[0], _i64[1]); \
	PRINT128_ ## fmt(&a);             \
	putc('\n', stdout);               \
} while(0);

#define M128I_M64x2(fmt, op) \
do {                                      \
	printf("%d: " # op, ++insts);     \
	putc('\t', stdout);               \
	__m128i a = op(_m64[0], _m64[1]); \
	PRINT128_ ## fmt(&a);             \
	putc('\n', stdout);               \
} while(0);

#define M128I_M128IP(fmt, op) \
do {                                          \
	printf("%d: " # op, ++insts);         \
	putc('\t', stdout);                   \
	init_mem(m128ip_a, sizeof(m128ip_a)); \
	__m128i a = op(m128ip_a);             \
	PRINT128_ ## fmt(&a);                 \
	putc('\n', stdout);                   \
} while(0);

#define M128_F32P(fmt, op) \
do {                                      \
	printf("%d: " # op, ++insts);     \
	putc('\t', stdout);               \
	init_mem(f32p_a, sizeof(f32p_a)); \
	__m128 a = op(f32p_a);            \
	PRINT128_ ## fmt(&a);             \
	putc('\n', stdout);               \
} while(0);

#define M128D_F64P(fmt, op) \
do {                                      \
	printf("%d: " # op, ++insts);     \
	putc('\t', stdout);               \
	init_mem(f64p_a, sizeof(f64p_a)); \
	__m128d a = op(f64p_a);           \
	PRINT128_ ## fmt(&a);             \
	putc('\n', stdout);               \
} while(0);

#define M128I_VP(fmt, op) \
do {                                          \
	printf("%d: " # op, ++insts);         \
	putc('\t', stdout);                   \
	init_mem(buffer_a, sizeof(buffer_a)); \
	__m128i a = op(vp_a);                 \
	PRINT128_ ## fmt(&a);                 \
	putc('\n', stdout);                   \
} while(0);

#define M128_M128_M64P(fmt, op) \
do {                                      \
	printf("%d: " # op, ++insts);     \
	putc('\t', stdout);               \
	init_mem(m64p_a, sizeof(m64p_a)); \
	__m128 a = op(_m128[0], m64p_a);  \
	PRINT128_ ## fmt(&a);             \
	putc('\n', stdout);               \
} while(0);

#define M128D_M128D_F64P(fmt, op) \
do {                                       \
	printf("%d: " # op, ++insts);      \
	putc('\t', stdout);                \
	init_mem(f64p_a, sizeof(f64p_a));  \
	__m128d a = op(_m128d[0], f64p_a); \
	PRINT128_ ## fmt(&a);              \
	putc('\n', stdout);                \
} while(0);

#define TEST_MALLOC_AND_FREE() \
do {                                          \
	printf("%d: " "_mm_malloc", ++insts); \
	void *ptr = _mm_malloc(16, 16);       \
	if (print_udef)                       \
		printf("\t %p", ptr);         \
	else                                  \
	        printf("\t OK");              \
	putc('\n', stdout);                   \
	printf("%d: " "_mm_free", ++insts);   \
	_mm_free(ptr);                        \
	printf("\t OK");                      \
	putc('\n', stdout);                   \
} while(0);

#endif /* _COMMON_DEF_H_ */
