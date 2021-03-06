#include <cstdio>
#include "SHA1.h"
#include "bigInt.h"

#define SHA1_MAX_FILE_BUFFER (32 * 20 * 820)

// Rotate p_val32 by p_nBits bits to the left
#define ROL32(p_val32,p_nBits) (((p_val32)<<(p_nBits))|((p_val32)>>(32-(p_nBits))))

#define SHABLK0(i) (m_block->l[i] = \
	(ROL32(m_block->l[i],24) & 0xFF00FF00) | (ROL32(m_block->l[i],8) & 0x00FF00FF))
#define SHABLK(i) (m_block->l[i&15] = ROL32(m_block->l[(i+13)&15] ^ \
	m_block->l[(i+8)&15] ^ m_block->l[(i+2)&15] ^ m_block->l[i&15],1))

// SHA-1 rounds
#define S_R0(v,w,x,y,z,i) {z+=((w&(x^y))^y)+SHABLK0(i)+0x5A827999+ROL32(v,5);w=ROL32(w,30);}
#define S_R1(v,w,x,y,z,i) {z+=((w&(x^y))^y)+SHABLK(i)+0x5A827999+ROL32(v,5);w=ROL32(w,30);}
#define S_R2(v,w,x,y,z,i) {z+=(w^x^y)+SHABLK(i)+0x6ED9EBA1+ROL32(v,5);w=ROL32(w,30);}
#define S_R3(v,w,x,y,z,i) {z+=(((w|x)&y)|(w&x))+SHABLK(i)+0x8F1BBCDC+ROL32(v,5);w=ROL32(w,30);}
#define S_R4(v,w,x,y,z,i) {z+=(w^x^y)+SHABLK(i)+0xCA62C1D6+ROL32(v,5);w=ROL32(w,30);}


__device__ CSHA1::CSHA1(){
	initSHA1(this);
}

__device__ void CSHA1::Reset(){
	m_state[0] = 0x67452301;
	m_state[1] = 0xEFCDAB89;
	m_state[2] = 0x98BADCFE;
	m_state[3] = 0x10325476;
	m_state[4] = 0xC3D2E1F0;

	m_count[0] = 0;
	m_count[1] = 0;
}

__device__ void CSHA1::Transform(uint32_t* pState, const uint8_t* pBuffer){
	uint32_t a = pState[0], b = pState[1], c = pState[2], d = pState[3], e = pState[4];

	memcpy(m_block, pBuffer, 64);

	// 4 rounds of 20 operations each, loop unrolled
	S_R0(a,b,c,d,e, 0); S_R0(e,a,b,c,d, 1);
	S_R0(d,e,a,b,c, 2); S_R0(c,d,e,a,b, 3);
	S_R0(b,c,d,e,a, 4); S_R0(a,b,c,d,e, 5);
	S_R0(e,a,b,c,d, 6); S_R0(d,e,a,b,c, 7);
	S_R0(c,d,e,a,b, 8); S_R0(b,c,d,e,a, 9);
	S_R0(a,b,c,d,e,10); S_R0(e,a,b,c,d,11);
	S_R0(d,e,a,b,c,12); S_R0(c,d,e,a,b,13);
	S_R0(b,c,d,e,a,14); S_R0(a,b,c,d,e,15);
	S_R1(e,a,b,c,d,16); S_R1(d,e,a,b,c,17);
	S_R1(c,d,e,a,b,18); S_R1(b,c,d,e,a,19);
	S_R2(a,b,c,d,e,20); S_R2(e,a,b,c,d,21);
	S_R2(d,e,a,b,c,22); S_R2(c,d,e,a,b,23);
	S_R2(b,c,d,e,a,24); S_R2(a,b,c,d,e,25);
	S_R2(e,a,b,c,d,26); S_R2(d,e,a,b,c,27);
	S_R2(c,d,e,a,b,28); S_R2(b,c,d,e,a,29);
	S_R2(a,b,c,d,e,30); S_R2(e,a,b,c,d,31);
	S_R2(d,e,a,b,c,32); S_R2(c,d,e,a,b,33);
	S_R2(b,c,d,e,a,34); S_R2(a,b,c,d,e,35);
	S_R2(e,a,b,c,d,36); S_R2(d,e,a,b,c,37);
	S_R2(c,d,e,a,b,38); S_R2(b,c,d,e,a,39);
	S_R3(a,b,c,d,e,40); S_R3(e,a,b,c,d,41);
	S_R3(d,e,a,b,c,42); S_R3(c,d,e,a,b,43);
	S_R3(b,c,d,e,a,44); S_R3(a,b,c,d,e,45);
	S_R3(e,a,b,c,d,46); S_R3(d,e,a,b,c,47);
	S_R3(c,d,e,a,b,48); S_R3(b,c,d,e,a,49);
	S_R3(a,b,c,d,e,50); S_R3(e,a,b,c,d,51);
	S_R3(d,e,a,b,c,52); S_R3(c,d,e,a,b,53);
	S_R3(b,c,d,e,a,54); S_R3(a,b,c,d,e,55);
	S_R3(e,a,b,c,d,56); S_R3(d,e,a,b,c,57);
	S_R3(c,d,e,a,b,58); S_R3(b,c,d,e,a,59);
	S_R4(a,b,c,d,e,60); S_R4(e,a,b,c,d,61);
	S_R4(d,e,a,b,c,62); S_R4(c,d,e,a,b,63);
	S_R4(b,c,d,e,a,64); S_R4(a,b,c,d,e,65);
	S_R4(e,a,b,c,d,66); S_R4(d,e,a,b,c,67);
	S_R4(c,d,e,a,b,68); S_R4(b,c,d,e,a,69);
	S_R4(a,b,c,d,e,70); S_R4(e,a,b,c,d,71);
	S_R4(d,e,a,b,c,72); S_R4(c,d,e,a,b,73);
	S_R4(b,c,d,e,a,74); S_R4(a,b,c,d,e,75);
	S_R4(e,a,b,c,d,76); S_R4(d,e,a,b,c,77);
	S_R4(c,d,e,a,b,78); S_R4(b,c,d,e,a,79);

	// Add the working vars back into state
	pState[0] += a;
	pState[1] += b;
	pState[2] += c;
	pState[3] += d;
	pState[4] += e;
}

__device__ void CSHA1::Update(const uint8_t* pbData, uint32_t uLen){
	uint32_t j = ((m_count[0] >> 3) & 0x3F);
	if((m_count[0] += (uLen << 3)) < (uLen << 3))
		++m_count[1]; // Overflow
	m_count[1] += (uLen >> 29);

	uint32_t i;
	if((j + uLen) > 63){
		i = 64 - j;
		memcpy(&m_buffer[j], pbData, i);
		Transform(m_state, m_buffer);

		for( ; (i + 63) < uLen; i += 64)
			Transform(m_state, &pbData[i]);
		j = 0;
	}
	else
		i = 0;

	if((uLen - i) != 0)
		memcpy(&m_buffer[j], &pbData[i], uLen - i);
}

__device__ void CSHA1::Final(){
	uint32_t i;

	uint8_t pbFinalCount[8];
	for(i = 0; i < 8; ++i)
		pbFinalCount[i] =
		    static_cast<uint8_t>((m_count[!(i >= 4)]
		    	>> ((3 - (i & 3)) * 8) ) & 0xFF);

	Update((uint8_t*)"\200", 1);
	while((m_count[0] & 504) != 448)
		Update((uint8_t*)"\0", 1);
	Update(pbFinalCount, 8); // Cause a Transform()

	for(i = 0; i < 20; ++i)
		m_digest[i] = static_cast<uint8_t>((m_state[i >> 2]
			>> ((3 - (i & 3)) * 8)) & 0xFF);
}

__device__ void CSHA1::GetHash(uint8_t* dest) const{
	memcpy(dest, m_digest, 20);
}
__device__ const uint8_t* CSHA1::GetPtr() const{
	return m_digest;
}

__device__ void initSHA1(CSHA1* sha){
	sha->m_block = (SHA1_WORKSPACE_BLOCK*)sha->m_workspace;
	sha->Reset();
}






static __constant__ uint8_t d_h[20];	//Hash to check, passed by host
static __device__ inline bool checkHash(const uint8_t h[20]){
	bool equal = true;
	#pragma unroll 5
	for(int i = 0; i < 20; i++)
		equal &= d_h[i] == h[i];
	return equal;
}


//Chars to search
static __shared__ char symbols[64];
static __device__ bool done = 0;
static __global__ void sha1_autoGpu(int len, int devIndex=0, int devNum=1){
	extern __shared__ char share[];

	const int idx = threadIdx.x;
	const int id = idx + blockIdx.x*blockDim.x;
	//myString and myMD5 are in shared memory for performance
	char* myString =    share + idx * len * sizeof(char);
	CSHA1* mySHA = (CSHA1*)(share + len * blockDim.x * sizeof(char)
	                    +sizeof(CSHA1) * idx);

	if(idx < 64){
		const char sym[] = {
			'a','b','c','d','e','f','g','h','i','j','k','l','m',
			'n','o','p','q','r','s','t','u','v','w','x','y','z',
			'A','B','C','D','E','F','G','H','I','J','K','L','M',
			'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
			'0','1','2','3','4','5','6','7','8','9',' ','-'};
		 symbols[idx] = sym[idx];
	}
	__syncthreads();

	//Will hold the maximum number of iterations needed
	uint128_t max(1);
	//returns number of fixed symbols, currently 3
	genPostfix(myString, id, len, devIndex, symbols);

	//number of iterations 64^(len-fix)
	max <<= 6*(len-FIXED);

	for(uint128_t it; it < max; ++it){
		//generate the prefix for the current iteration
		genString(myString, it, len, symbols);
		//calc SHA1
		initSHA1(mySHA);
		mySHA->Update((uint8_t*)myString, len);
		mySHA->Final();

		//Get the Hash
		bool equal = checkHash(mySHA->GetPtr());
		if(equal){
			done=true;
			//we saved ourselves null-termination up until now
			myString[len] = 0;

			//Print some information
			printf("ID: %d\n", id);
			printf("Iteration: %016llx%016llx\n", it.val[1],it.val[0]);
			printShaGPU(mySHA->GetPtr());
			printf("String: ");
			for(int i = 0; i < len; i++)
				printf("%c", myString[i]);
			printf("\n");
		}
		//If anyone found the has found the hash return
		if(done){
			return;
		}
	}
}



void sha1_auto(const uint8_t* h, int len){
	if(len < 2){
		printf("Less than 2 chars not yet supported!");
		exit(-1);
	}

	int devNum;
	cudaGetDeviceCount(&devNum);
	cudaStream_t streams[devNum];
	for(int i = 0; i < devNum; i++){
		cudaSetDevice(i);
		cudaStreamCreate(&streams[i]);

		//Copy hash into constant memory
		cudaMemcpyToSymbol(d_h, h, 20*sizeof(uint8_t));
		checkCUDAError("cudaMemcpyToSymbol");

		//We allocate shared memory for each threads string and MD5 object
		sha1_autoGpu<<<BLOCKNUM/devNum, THREADNUM,
		              (len+sizeof(CSHA1))*THREADNUM,streams[i]>>>
		              (len, i, devNum);
	}
	for(int i = 0; i < devNum; i++){
		cudaSetDevice(i);
		cudaDeviceSynchronize();
		checkCUDAError("md5_autoGpu");
	}
}
