// VectorAddition.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include <Windows.h>
//header file to use SSE Intrinsic notation
//#include <intrin.h>
#include<iostream>
#include<cmath>
#include<xmmintrin.h>
#include<fvec.h>
#include<stdio.h>
#include <vector>

using namespace std;
void printMatrix(float* matrix, int size);

//the compiler will decide if we should vectorize this function and how.
//for a loop to be vectorized it must be inlined and terminate due to its end condition
//my inner loop is the only loop that is inlined so it is the only one that will be vectorized
void matrixMultAutoVec(float *matrix1,float * matrix2,const int size,float *__restrict result){
	float	temp;
	int		iterator1;
	float*	data1;
	float*	data2;
	float*	data3;
	float*	data4;
	for (int k = 0; k < size; ++k){
		iterator1 = k*size;
		data3 = &matrix1[iterator1];
		data4 = &result[iterator1];
		for (int j = 0; j < size; ++j){
			temp		= *(data3++);
			data1		= data4;
			data2		= &matrix2[j*size];
			for (int i = 0; i < size; ++i)	*(data1++) +=temp * *(data2++);
		}
	}
}

//Uses vector objects to multiply 2 matrices. Vector objects overload the standard */= operators
//and uses SSE intructions to optimize our code
void matrixMultVec(float* matrix1,float* matrix2, const int size,float* result){
	F32vec4 *matrix1TMP;
	F32vec4 *matrix2TMP;
	F32vec4 *matrix11TMP;
	F32vec4 *Answer;
	F32vec4 *matrix111TMP	= (F32vec4*) matrix1;
	F32vec4 *matrix22TMP	= (F32vec4*) matrix2;
	F32vec4 *Answer1		= (F32vec4*) result;
	register int nextRow = size >> 2;
	register _declspec(align(16)) struct {int i,j,k;} iter;
	for(iter.i = 0;iter.i<size;iter.i+=4,++matrix22TMP,++Answer1){
		Answer = Answer1;
		matrix11TMP = matrix111TMP;
		for(iter.j = 0;iter.j<size;++iter.j,matrix11TMP+=nextRow,Answer+=nextRow){
			matrix1TMP = matrix11TMP;
			matrix2TMP = matrix22TMP;
			for(iter.k = 0;iter.k<size;iter.k+=4,++matrix1TMP){
				
				*Answer		+=	_mm_shuffle_ps(*matrix1TMP,*matrix1TMP,0x00)*(*matrix2TMP);

				*Answer		+=	_mm_shuffle_ps(*matrix1TMP,*matrix1TMP,0x55)*(*(matrix2TMP+nextRow));

				*Answer		+=	_mm_shuffle_ps(*matrix1TMP,*matrix1TMP,0xAA)*(*(matrix2TMP+(nextRow<<1)));

				*Answer		+=	_mm_shuffle_ps(*matrix1TMP,*matrix1TMP,0xFF)*(*(matrix2TMP+((nextRow<<1)+nextRow)));
				
				matrix2TMP += (nextRow<<2);
			}
		}
	}
}

//uses intrinsics to allow us to use SSE intructions along side our standard C++ instructions.
//this allows us to do things such as loops even easier and quicker.
void matrixMultIntrin(float* matrix1, float* matrix2, const int size, float* result){
	__m128 t0,sum,t1;
	int itemNum = size*size;
	float* matrix1TMP,*matrix11TMP,*matrix111 = matrix1,*matrix1End,*matrix11End;
	float* matrix2TMP,*matrix22TMP,*matrix222 = matrix2,*matrix2End = matrix2+(size);
	float* answer	= result;
	float* answerr	= result;
	for(;matrix222<matrix2End;matrix222+=4, answerr+=4){
		matrix11TMP = matrix111;
		matrix22TMP = matrix222;
		result = answerr;
		for(matrix11End = matrix11TMP + itemNum;matrix11TMP<matrix11End;matrix11TMP+=size,result+=size){
			matrix1TMP	= matrix11TMP;
			matrix2TMP	= matrix22TMP;
			sum			= _mm_set1_ps(0.0);
			for(matrix1End = matrix1TMP+size;matrix1TMP<matrix1End;){
				t0		= _mm_load_ps(matrix1TMP);
				t1		= _mm_load_ps(matrix2TMP);
				sum		= _mm_add_ps(_mm_mul_ps(_mm_set1_ps(*(matrix1TMP++)),t1),sum);

				sum		= _mm_add_ps(_mm_mul_ps(_mm_set1_ps(*((matrix1TMP++))),t1),sum);
	
				sum		= _mm_add_ps(_mm_mul_ps(_mm_set1_ps(*((matrix1TMP++))),t1),sum);

				sum		= _mm_add_ps(_mm_mul_ps(_mm_set1_ps(*(matrix1TMP++)),t1),sum);

				matrix2TMP += size<<2;
			}
		_mm_store_ps(result,sum);
		}	
	}
	result = answer;
}

//uses Assembly Language to implement matrix multiplication. It is more efficient than the rest but it is
//more complex. This implementation uses row times column multiplication to solve the matrix multiplation
void matrixMultSSE_1st(float* matrix1, float* matrix2, const int size, float* result){
	int rownumber = size;
	__asm
	{
		mov		esi, matrix1
		mov		edi, matrix2
		mov		ecx, rownumber
		mov		ebx, ecx
		shl		ebx, 2
		mov		eax, result
PREREQ:
		xorps   xmm0,xmm0
		push esi
		push edi
		push ecx
		push eax
		mov ecx, rownumber
READY:
		push ecx
		push esi
		push edi
		mov ecx, rownumber
	START:
		movaps xmm1,xmmword ptr[esi]
		movaps xmm7,xmm1
		shufps xmm1,xmm1,0x00
		movaps xmm2,xmmword ptr[edi]
		mulps  xmm2,xmm1
		addps  xmm0,xmm2
		
		movaps	xmm1,xmm7
		shufps	xmm1,xmm1,0x55
		add		edi ,ebx
		movaps	xmm2,xmmword ptr[edi]
		mulps	xmm2,xmm1
		addps	xmm0,xmm2


		movaps	xmm1,xmm7
		shufps	xmm1,xmm1,0xAA
		add		edi ,ebx
		movaps	xmm2,xmmword ptr[edi]
		mulps	xmm2,xmm1
		addps	xmm0,xmm2
		
		movaps	xmm1,xmm7
		shufps	xmm1,xmm1,0xFF
		add		edi ,ebx
		movaps	xmm2,xmmword ptr[edi]
		mulps	xmm2,xmm1
		addps	xmm0,xmm2

		add esi,0x10
		add edi,ebx
		sub ecx,0x4
	jnz START

		movaps	xmmword ptr[eax], xmm0
		xorps	xmm0,xmm0
		pop		edi
		pop		esi
		add		esi, ebx
		add		eax, ebx
		pop		ecx
		sub		ecx, 0x1
			
	jnz READY
		pop eax
		pop ecx
		pop edi
		pop esi

		add eax, 0x10
		add edi, 0x10
		sub ecx,0x4
	jnz PREREQ
	}
}

//uses Assebly Language to solve matrix multiplations but instead of using row column multiplication it uses 
//column column multiplication to take advantage of spatial locality principle. We access data segments of our array 
//sequencially therefore reducing the number of cache misses and improving performance over the prior implementation
void matrixMultSSE_2nd(float* matrix1, float* matrix2, const int _size, float* result){
	__asm
	{
		mov		esi, matrix1
		mov		edi, matrix2
		mov		ecx, _size//rownumber
		mov		ebx, result
		mov		eax, ecx;//rownumber
		shl		eax, 2
	PREREQ :
		push ecx
		push esi;
		push edi
		mov ecx, _size;//rownumber
	START :
		push ecx
		mov		edx, ebx
		mov		ecx, _size;//rownumber
		movaps xmm1, xmmword ptr[esi];
		movaps xmm7, xmm1
		shufps xmm1, xmm1, 0x00;

	ILOOP :
		movaps xmm0, xmmword ptr[edx];
		movaps xmm2, xmmword ptr[edi];
		mulps  xmm2, xmm1;
		addps  xmm0, xmm2;
		movaps xmmword ptr[edx], xmm0;
		add  edx,0x10
		add  edi,0x10
		sub  ecx, 0x4
	jnz		ILOOP

	mov ecx, _size;//rownumber
	mov edx, ebx;
	movaps xmm1, xmm7//[esi];
	shufps xmm1, xmm1, 0x55

	ILOOP2 :
		movaps xmm0, xmmword ptr[edx];
		movaps xmm2, xmmword ptr[edi];
		mulps xmm2, xmm1;
		addps  xmm0, xmm2;
		movaps xmmword ptr[edx], xmm0;
		add edx, 0x10
		add edi, 0x10
		sub ecx, 0x4;
	jnz ILOOP2

	mov ecx, _size;//rownumber
	mov edx, ebx;
	movaps xmm1, xmm7//[esi];
	shufps xmm1, xmm1, 0xAA

	ILOOP3 :
		movaps xmm0, xmmword ptr[edx];
		movaps xmm2, xmmword ptr[edi];
		mulps xmm2, xmm1;
		addps  xmm0, xmm2;
		movaps xmmword ptr[edx], xmm0;
		add edx, 0x10
		add edi, 0x10
		sub ecx, 0x4;
	jnz ILOOP3

	mov ecx, _size;//rownumber
	mov edx, ebx;
	movaps xmm1, xmm7//[esi];
	shufps xmm1, xmm1, 0xFF

	ILOOP4 :
		movaps xmm0, xmmword ptr[edx];
		movaps xmm2, xmmword ptr[edi];
		mulps xmm2, xmm1;
		addps  xmm0, xmm2;
		movaps xmmword ptr[edx], xmm0;
		add edx, 0x10
		add edi, 0x10
		sub ecx, 0x4;
	jnz ILOOP4

		add esi, 0x10
		pop ecx
		sub ecx, 0x04
	jnz START

		pop edi; matrix 2
		pop esi; matrix 1
		pop ecx
		add esi, eax
		add ebx, eax
		sub ecx, 0x1
	jnz PREREQ
	}
}


//uses standard scalar programming implementation to solve our matrix multiplication.
//this is the function we are trying to optimize using SSE instructions.
void matrixMult(float* matrix1, float*matrix2, int size, float* result){
	float* matrix1_iter(matrix1), *matrix2_iter(matrix2), *result_iter(result), *temp(matrix1);
	for (int k = 0; k < size; ++k){
		for (int j = 0; j < size; ++j){
			for (int i = 0; i < size; ++i){
				*result += *matrix1_iter * *matrix2_iter;
				++matrix1_iter;
				matrix2_iter += size;
			}
			matrix1_iter = temp;
			matrix2_iter = matrix2 + j + 1;
			++result;
		}
		matrix1_iter = matrix1 + size*(k + 1);
		matrix2_iter = matrix2;
		temp = matrix1_iter;
	}
}


void fillArray(float* Array, int size){
	float* iter = &Array[0];
	float* end = iter + size;
	int i = 1;
	while (iter < end){
		*iter = rand()%4;
		++iter;
	}
}


void printMatrix(float* matrix, int size){
	float* matrix_iter(matrix);
	float* matrix_end(matrix + size);
	int space_count = 0;
	int dimention = sqrt(size);
	while (matrix_iter != matrix_end){
		cout << *matrix_iter << "\t";
		++space_count;
		++matrix_iter;
		if (space_count != 0 && space_count%dimention == 0) cout << endl;
	}
	cout << endl;
}


void clearArray(float* matrix,int size){
	float* begin = &matrix[0];
	float* end = begin+size;
	while(begin<end){
		*begin = 0;
		++begin;
	}
}

void clearArrayIndex(float* matrix,int size){
	for(int i = 0;i<size;++i){
		matrix[i] = 0;
	}
}


int _tmain(int argc, _TCHAR* argv[])
{
	const int size = 8;

	//by default int is allocated at an address space that is a multiple of 4, using __declspec(align(16)) we 
	//tell the compiler we want to allocate our data in an address space that is a multiple of 16
	__declspec(align(16)) float m3[size*size],	m4[size*size],	m5[size*size]	= { 0 };

	fillArray(m3, size*size);
	fillArray(m4, size*size);

	//to measure time using query performance counters each code segment of which the time needs to be measured for should have two counters
	__int64 ctr1 = 0, ctr2 = 0, ctr3 = 0, ctr4 = 0, freq = 0;
	/*********************************************************************************************/
	/*****************************		GET CLOCK FREQUENCY		**********************************/
	/*********************************************************************************************/

	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
	printf("This computer has the following Clock Frequency and Period\n");
	printf("Frequency: %I64d\tPeriod: %e\n\n", freq, 1.0 / freq);

	/*********************************************************************************************/
	/***********************		START TIMING INDEX IMPLEMENTATION		**********************/
	/*********************************************************************************************/
	if(size<=8){
		printMatrix(m3,size*size);
		cout<<endl<<"*"<<endl<<endl;
		printMatrix(m4,size*size);
		cout<<endl<<"="<<endl<<endl;
	}
	
	QueryPerformanceCounter((LARGE_INTEGER *)&ctr1);
	matrixMult(m3, m4, size, m5);

	QueryPerformanceCounter((LARGE_INTEGER *)&ctr2);

	if(size<=8) printMatrix(m5,size*size);
	
	printf("Using INDEXING our matrix NXN multiplication of size N=%d\ncompleted in %I64d clock cycles, %e seconds\n\n", size,(ctr2 - ctr1), (ctr2 - ctr1)*1.0 / freq);

	clearArray(m5,size*size);
	
	/*********************************************************************************************/
	/***********************		START TIMING SSE IMPLEMENTATION		**************************/
	/*********************************************************************************************/
	QueryPerformanceCounter((LARGE_INTEGER *)&ctr3);
	matrixMultSSE_2nd(m3, m4, size,  m5);
	//matrixMultSSE_1st(m3, m4, size, m5);
	//matrixMultAutoVec(m3,m4,size,m5);
	//matrixMultVec(m3,m4,size,m5);
	//matrixMultIntrin(m3,m4,size,m5);
	QueryPerformanceCounter((LARGE_INTEGER *)&ctr4);

	//use the bottom print matrix statement to verify the result of our optimized functions.
	//printMatrix(m5, size*size);

	printf("Using Intel SSE notation our matrix NXN multiplication of size N=%d \ncompleted in %I64d clock cycles, %e seconds\n\n", size, (ctr4 - ctr3), (ctr4 - ctr3)*1.0 / freq);

	ctr4 -= ctr3; ctr2 -= ctr1;
	if (ctr4 > ctr2)
	{
		printf("Using indexing is %fX faster\n", (1.0 * ctr4) / (1.0 * ctr2));

	}
	else if (ctr4 < ctr2)
	{
		printf("Using SSE is %3.2fx faster\n", (1.0 * ctr2) / (1.0 * ctr4));
	}
	else
		printf("Both methods are equal in speed\n");

	system("PAUSE");

	return 0;
}
