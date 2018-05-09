//#include <cuda_runtime.h>
//#include "device_launch_parameters.h"
//#include <helper_cuda.h>
////#include "sm_20_atomic_functions.h"
//
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/count.h>
//#include <stdio.h>
//
//#define REAL float
////#define USE_CONST_MEM
//#define HANDLE_ERROR checkCudaErrors
//
//float   elapsedTime;
//#define START_GPU {\
//elapsedTime = 0.0;\
//cudaEvent_t     start, stop;\
//checkCudaErrors(cudaEventCreate(&start)); \
//checkCudaErrors(cudaEventCreate(&stop));\
//checkCudaErrors(cudaEventRecord(start, 0));\
//
//#define END_GPU \
//checkCudaErrors(cudaEventRecord(stop, 0));\
//checkCudaErrors(cudaEventSynchronize(stop));\
//checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop)); \
//printf("GPU Time used:  %3.2f ms\n", elapsedTime);\
//checkCudaErrors(cudaEventDestroy(start));\
//checkCudaErrors(cudaEventDestroy(stop));}
//
//#define START_CPU {\
//double start = omp_get_wtime();
//
//#define END_CPU \
//double end = omp_get_wtime();\
//double duration = end - start;\
//printf("CPU Time used: %3.1f ms\n", duration * 1000);}
//
////############################################################################
//#ifdef _WIN64
//#define GLUT_NO_LIB_PRAGMA
//#pragma comment (lib, "opengl32.lib")
//#pragma comment (lib, "glut64.lib")
//#endif //_WIN64
//
///* On Windows, include the local copy of glut.h and glext.h */
//#include "GL/glut.h"
//#include "GL/glext.h"
//#define GET_PROC_ADDRESS( str ) wglGetProcAddress( str )
//
////----------------------��װ��bitmap��------------------------------
//struct CPUAnimBitmap {
//	//��������
//	unsigned char    *pixels;
//	int     width, height;
//	//һ��ָ��
//	void    *dataBlock;
//
//	//���Զ�̬�����ú�����ָ��
//	void(*fAnim)(void*, int);
//	void(*animExit)(void*);
//	void(*clickDrag)(void*, int, int, int, int);
//	int     dragStartX, dragStartY;
//	
//	CPUAnimBitmap(int w, int h, void *d = NULL) {
//		width = w;
//		height = h;
//		//r g b alph
//		pixels = new unsigned char[width * height * 4];
//		dataBlock = d;
//		clickDrag = NULL;
//	}
//
//	~CPUAnimBitmap() {
//		delete[] pixels;
//	}
//
//	unsigned char* get_ptr(void) const { return pixels; }
//	long image_size(void) const { return width * height * 4; }
//
//	void click_drag(void(*f)(void*, int, int, int, int)) {
//		clickDrag = f;
//	}
//
//	//��Ⱦ���ͼƬ
//	//input: f����ʹ��GPU����õ�bitmap��ͼƬ�ĺ���
//	//		 e��cuda ������
//	void anim_and_exit(void(*f)(void*, int), void(*e)(void*)) {
//		CPUAnimBitmap**   bitmap = get_bitmap_ptr();
//		*bitmap = this;
//		fAnim = f;
//		animExit = e;
//		// a bug in the Windows GLUT implementation prevents us from
//		// passing zero arguments to glutInit()
//		int c = 1;
//		char* dummy = "";
//		glutInit(&c, &dummy);
//		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
//		glutInitWindowSize(width, height);
//		glutCreateWindow("bitmap");
//		glutKeyboardFunc(Key);
//		glutDisplayFunc(Draw);
//
//		if (clickDrag != NULL)
//			glutMouseFunc(mouse_func);
//
//		//glutIdleFunc����ȫ�ֵĻص���������û�д����¼�����ʱ��
//		//GLUT�����ܿ���ִ�к�̨�������������������
//		//������ã����idle function�ᱻ���ϵ��ã�ֱ���д����¼�������
//		glutIdleFunc(idle_func);
//		glutMainLoop();
//	}
//
//	// static method used for glut callbacks
//	static CPUAnimBitmap** get_bitmap_ptr(void) {
//		static CPUAnimBitmap*   gBitmap;
//		return &gBitmap;
//	}
//
//	// static method used for glut callbacks
//	static void mouse_func(int button, int state,
//		int mx, int my) {
//		if (button == GLUT_LEFT_BUTTON) {
//			CPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
//			if (state == GLUT_DOWN) {
//				bitmap->dragStartX = mx;
//				bitmap->dragStartY = my;
//			}
//			else if (state == GLUT_UP) {
//				bitmap->clickDrag(bitmap->dataBlock,
//					bitmap->dragStartX,
//					bitmap->dragStartY,
//					mx, my);
//			}
//		}
//	}
//
//	// static method used for glut callbacks
//	static void idle_func(void) {
//		static int ticks = 1;
//		CPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
//		bitmap->fAnim(bitmap->dataBlock, ticks++);
//		glutPostRedisplay();
//	}
//
//	// static method used for glut callbacks
//	static void Key(unsigned char key, int x, int y) {
//		switch (key) {
//		case 27:
//			CPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
//			bitmap->animExit(bitmap->dataBlock);
//			//delete bitmap;
//			exit(0);
//		}
//	}
//
//	// static method used for glut callbacks
//	static void Draw(void) {
//		CPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
//		glClearColor(0.0, 0.0, 0.0, 1.0);
//		glClear(GL_COLOR_BUFFER_BIT);
//		glDrawPixels(bitmap->width, bitmap->height, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels);
//		glutSwapBuffers();
//	}
//};
//
////ͼƬ������ֵ
//#define DIM 1024
//#define rnd( x ) (x * rand() / RAND_MAX)
//#define INF     2e10f
//
////----------------------------��װ��һ����-------------------------------
//struct Sphere {
//	REAL   r, b, g;
//	REAL   radius;
//	//С���λ��
//	REAL   x, y, z;
//	//ÿһ֡С����ƶ����ٶ�
//	REAL dx, dy, dz;
//	bool isCrash;
//	//������ ox,oy�������صĹ��ߣ��Ƿ����������ཻ��
//	//��������������ཻ����ô���������������������������������ľ��롣
//	//�����Ͷ�������ཻ��ֻ��¼��ӽ����������Żᱻ������
//	__device__ REAL hit(REAL ox, REAL oy, REAL *n) {
//		REAL dx = ox - x;
//		REAL dy = oy - y;
//		//����С������İ뾶��ʱ�򣬲��ܺ������ཻ
//		if (dx*dx + dy*dy < radius*radius) {
//			REAL dz = sqrtf(radius*radius - dx*dx - dy*dy);
//			*n = dz / sqrtf(radius * radius);
//			return dz + z;
//		}
//		//����Զ
//		return -INF;
//	}
//};
//
////------------С����ײ�ĸ���----------
//#define SPHERES 2000
//
//int *d_crashnum, *h_crashnum;
//
//#ifdef USE_CONST_MEM
//__constant__ Sphere d_spheres[SPHERES];
//#else
//Sphere  *d_spheres;
//#endif
//
////------------------------cuda kernel --------------------------
//
//#define STEP_SIZE REAL(20.0)
//
////�����ײ��С��ĸ���
//__global__ void crash(Sphere *s, int num_sphere, int*d_crashnum , int streamId , int streamNum)
//{
//	//�õ�������ײС������
//	int s1 = threadIdx.x + blockIdx.x * blockDim.x;
//	int s2 = threadIdx.y + blockIdx.y * blockDim.y;
//
//	s2 = s2 + 64 / 4 * streamId * 32;
//	//�����Ϊx,y������С�������ײ���,�Գƾ��󣬼���һ��ľ���
//	if (s2 < num_sphere && s1 < num_sphere && s2 < s1)
//	//if (s2 < num_sphere && s1 < num_sphere)
//	{
//		REAL dx = s[s1].x - s[s2].x;
//		REAL dy = s[s1].y - s[s2].y;
//		REAL dz = s[s1].z - s[s2].z;
//		REAL totalRadius = s[s1].radius + s[s2].radius;
//		//�ж��Ƿ���ײ
//		if (dx*dx + dy*dy + dz*dz <= totalRadius * totalRadius)
//		{
//			s[s1].isCrash = true;
//			s[s2].isCrash = true;
//
//			//printf("y: %d  x: %d\n", s2,s1);
//
//			atomicAdd(d_crashnum, 1);
//		}
//	}
//}
//
//__global__ void addKernel(int * num0 , int * num1, int * num2, int * num3,int * res)
//{
//	*res = *num0 + *num1 + *num2 + *num3;
//}
//
////�����������ڵ�λ��
//__global__ void kernelMoving(Sphere *s, int len)
//{
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	//�Ե�x �����壬���������ڵ�λ��
//	while (x < len) {
//		
//		s[x].isCrash = false;
//		s[x].x += s[x].dx;
//		s[x].y += s[x].dy;
//		s[x].z += s[x].dz;
//		x += gridDim.x*blockDim.x;
//	}
//}
//
//#ifdef USE_CONST_MEM
//__global__ void kernel(unsigned char *ptr) {
//#else
//__global__ void kernel(Sphere *d_spheres, unsigned char *ptr) {
//#endif
//	//�õ�pixel �����ص�λ�á�
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	//���ǵڼ�������
//	int offset = x + y * blockDim.x * gridDim.x;
//	REAL   ox = (x - DIM / 2);
//	REAL   oy = (y - DIM / 2);
//
//	REAL   r = 0, g = 0, b = 0;
//	REAL   maxz = -INF;
//	for (int i = 0; i < SPHERES; i++) {
//		REAL   n;
//		REAL   t = d_spheres[i].hit(ox, oy, &n);
//		if (t > maxz) {
//			REAL fscale = n;
//			if (d_spheres[i].isCrash)
//			{
//				r = 1.0f *fscale;
//				g = 0.0f*fscale;
//				b = 0.0f*fscale;
//			}
//			else
//			{
//				r = d_spheres[i].r * fscale;
//				g = d_spheres[i].g * fscale;
//				b = d_spheres[i].b * fscale;
//				maxz = t;
//			}
//		}
//	}
//
//	ptr[offset * 4 + 0] = (int)(r * 255);
//	ptr[offset * 4 + 1] = (int)(g * 255);
//	ptr[offset * 4 + 2] = (int)(b * 255);
//	ptr[offset * 4 + 3] = 255;
//}
//
//
//// globals needed by the update routine
//struct DataBlock {
//	//��� gpu �е�bitmap ������
//	unsigned char   *dev_bitmap;
//	//cpu�д��bitmap ������
//	CPUAnimBitmap   *bitmap;
//};
//
//
//#define streamNum 4
//cudaStream_t  stream0, stream1, stream2, stream3;
//int *crashNum0, *crashNum1, *crashNum2, *crashNum3;
//Sphere *sphere0, *sphere1, *sphere2, *sphere3;
//
//void generate_frame(DataBlock *d, int ticks) {
//	float totalTime = 0.0;
//	//��С�����ײ�ļ�������0
//	HANDLE_ERROR(cudaMemset(d_crashnum, 0, sizeof(int)));
//	//��С��ĸ��� copy��host �У�����ӡ����
//
//	START_GPU
//
//	//------------�ƶ���С��  --2000�� ----------------
//	kernelMoving << <64, 32 >> > (d_spheres, SPHERES);
//	END_GPU
//	totalTime += elapsedTime;
//
//	//----------------------------����stream handle-------------------------
//	//���ĸ���
//	START_GPU
//
//	dim3    crashGrids(64, 64 / streamNum);
//	dim3    crashBlock(32, 32);
//
//	HANDLE_ERROR(cudaMemset(crashNum0, 0, sizeof(int)));
//	HANDLE_ERROR(cudaMemset(crashNum1, 0, sizeof(int)));
//	HANDLE_ERROR(cudaMemset(crashNum2, 0, sizeof(int)));
//	HANDLE_ERROR(cudaMemset(crashNum3, 0, sizeof(int)));
//
//	cudaMemcpyAsync(crashNum0, d_crashnum, sizeof(Sphere) * SPHERES, cudaMemcpyDeviceToDevice, stream0);
//	cudaMemcpyAsync(crashNum1, d_crashnum, sizeof(Sphere) * SPHERES, cudaMemcpyDeviceToDevice, stream1);
//	cudaMemcpyAsync(crashNum2, d_crashnum, sizeof(Sphere) * SPHERES, cudaMemcpyDeviceToDevice, stream2);
//	cudaMemcpyAsync(crashNum3, d_crashnum, sizeof(Sphere) * SPHERES, cudaMemcpyDeviceToDevice, stream3);
//
//	cudaMemcpyAsync(sphere0, d_spheres, sizeof(Sphere) * SPHERES, cudaMemcpyDeviceToDevice, stream0);
//	cudaMemcpyAsync(sphere1, d_spheres, sizeof(Sphere) * SPHERES, cudaMemcpyDeviceToDevice, stream1);
//	cudaMemcpyAsync(sphere2, d_spheres, sizeof(Sphere) * SPHERES, cudaMemcpyDeviceToDevice, stream2);
//	cudaMemcpyAsync(sphere3, d_spheres, sizeof(Sphere) * SPHERES, cudaMemcpyDeviceToDevice, stream3);
//
//	crash << <crashGrids, crashBlock, 0, stream0 >> > (sphere0, SPHERES, crashNum0, 3, streamNum);
//	crash << <crashGrids, crashBlock, 0, stream1 >> > (sphere1, SPHERES, crashNum1, 2, streamNum);
//	crash << <crashGrids, crashBlock, 0, stream2 >> > (sphere2, SPHERES, crashNum2, 1, streamNum);
//	crash << <crashGrids, crashBlock, 0, stream3 >> > (sphere3, SPHERES, crashNum3, 0, streamNum);
//
//	//----------------------ͬ����------------------------------
//	cudaStreamSynchronize(stream0);
//	cudaStreamSynchronize(stream1);
//	cudaStreamSynchronize(stream2);
//	cudaStreamSynchronize(stream3);
//
//	/*thrust::host_vector<int> crashNumList(4);
//	crashNumList[0] = *crashNum0;
//	crashNumList[1] = *crashNum1;
//	crashNumList[2] = *crashNum2;
//	crashNumList[3] = *crashNum3;
//	int sum = thrust::reduce(crashNumList.begin(), crashNumList.end(), (int)0, thrust::plus<int>());*/
//	//printf("num of pair sphere crash:  %d\n", sum);
//
//	addKernel << <1, 1 >> > (crashNum0, crashNum1, crashNum2, crashNum3, d_crashnum);
//	//*d_crashnum = * + *crashNum1 + *crashNum2 + *crashNum3;
//	END_GPU
//
//	totalTime += elapsedTime;
//
//	//-----------��С������������һ�ŵ� bitmap--------
//	START_GPU
//	dim3    grids(DIM / 16, DIM / 16);
//	dim3    threads(16, 16);
//#ifdef USE_CONST_MEM
//	kernel << <grids, threads >> > (d->dev_bitmap);
//#else
//	kernel << <grids, threads >> > (d_spheres, d->dev_bitmap);
//#endif
//
//	END_GPU
//	totalTime += elapsedTime;
//
//	//-----��bitmap �����ݴ� device ������ host ��-----------
//	HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
//		d->bitmap->image_size(), cudaMemcpyDeviceToHost));
//
//	HANDLE_ERROR(cudaMemcpy(h_crashnum, d_crashnum,sizeof(int), cudaMemcpyDeviceToHost));
//	printf("num of pair sphere crash:  %d\n", (*h_crashnum));
//	printf("total time:  %3.1f\n", totalTime);
//	printf("---------------------------------------------\n");
//}	
//
//// clean up memory allocated on the GPU
//void cleanup(DataBlock *d) {
//	HANDLE_ERROR(cudaFree(d->dev_bitmap));
//	//�ͷ�С����ײ�����Ŀռ�
//	HANDLE_ERROR(cudaFree(d_crashnum));
//	free(h_crashnum);
//
//	//----------free stream-----------
//	cudaStreamDestroy(stream0);
//	cudaStreamDestroy(stream1);
//	cudaStreamDestroy(stream2);
//	cudaStreamDestroy(stream3);
//
//}
//
////-------------------------main-------------------------------
//
//int main(void) {
//	//-----------------�����Ƿ������������----------------------
//	cudaDeviceProp  prop;
//	int whichDevice;
//	cudaGetDevice(&whichDevice);
//	cudaGetDeviceProperties(&prop, whichDevice);
//	if (!prop.deviceOverlap) {
//		printf("Device will not handle overlaps, so no speed up from streams\n");
//		return;
//	}
//	else
//	{
//		printf("Device will  handle overlaps, so we can speed up from streams\n");
//	}
//
//	//----------create stream-----------
//	cudaStreamCreate(&stream0);
//	cudaStreamCreate(&stream1);
//	cudaStreamCreate(&stream2);
//	cudaStreamCreate(&stream3);
//
//	//--------------���С�����ײ------------------
//	HANDLE_ERROR(cudaMalloc(&crashNum0, sizeof(int)));
//	HANDLE_ERROR(cudaMalloc(&crashNum1, sizeof(int)));
//	HANDLE_ERROR(cudaMalloc(&crashNum2, sizeof(int)));
//	HANDLE_ERROR(cudaMalloc(&crashNum3, sizeof(int)));
//	HANDLE_ERROR(cudaMalloc(&sphere0, sizeof(Sphere) * SPHERES));
//	HANDLE_ERROR(cudaMalloc(&sphere1, sizeof(Sphere) * SPHERES));
//	HANDLE_ERROR(cudaMalloc(&sphere2, sizeof(Sphere) * SPHERES));
//	HANDLE_ERROR(cudaMalloc(&sphere3, sizeof(Sphere) * SPHERES));
//
//
//	//---------����ͼƬ�Ŀռ�----------
//	DataBlock   data;
//	CPUAnimBitmap bitmap(DIM, DIM, &data);
//	data.bitmap = &bitmap;
//
//	//����С����ײ�ļ������Ŀռ�
//	h_crashnum = (int *)malloc(sizeof(int));
//	*h_crashnum = 0;
//	
//	HANDLE_ERROR(cudaMalloc((void**)&d_crashnum, sizeof(int)));
//	HANDLE_ERROR(cudaMemcpy(d_crashnum, h_crashnum,sizeof(int), cudaMemcpyHostToDevice));
//	//---------����gpu�ռ�-------------
//	HANDLE_ERROR(cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size()));
//
//#ifdef USE_CONST_MEM
//#else
//	HANDLE_ERROR(cudaMalloc((void**)&d_spheres, sizeof(Sphere) * SPHERES));
//#endif
//
//	// allocate temp memory, initialize it, copy to constant
//	// memory on the GPU, then free our temp memory
//	Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
//	for (int i = 0; i < SPHERES; i++) {
//		temp_s[i].r = rnd(1.0f);
//		temp_s[i].g = rnd(1.0f);
//		temp_s[i].b = rnd(1.0f);
//		
//		temp_s[i].x = rnd(1000.0f) - 500;
//		temp_s[i].y = rnd(1000.0f) - 500;
//		temp_s[i].z = rnd(1000.0f) - 500;
//		temp_s[i].radius = rnd(10.0f) + 5;
//
//		//��ʼ�� С���ƶ����ٶ�
//		temp_s[i].dx = STEP_SIZE * ((rand() / (float)RAND_MAX) * 2 - 1);
//		temp_s[i].dy = STEP_SIZE * ((rand() / (float)RAND_MAX) * 2 - 1);
//		temp_s[i].dz = STEP_SIZE * ((rand() / (float)RAND_MAX) * 2 - 1);
//	}
//
//#ifdef USE_CONST_MEM
//	HANDLE_ERROR(cudaMemcpyToSymbol(d_spheres, temp_s, sizeof(Sphere) * SPHERES));
//#else
//	HANDLE_ERROR(cudaMemcpy(d_spheres, temp_s, sizeof(Sphere)*SPHERES, cudaMemcpyHostToDevice));
//#endif
//
//	free(temp_s);
//
//	// display
//	bitmap.anim_and_exit((void(*)(void*, int))generate_frame, (void(*)(void*))cleanup);
//}