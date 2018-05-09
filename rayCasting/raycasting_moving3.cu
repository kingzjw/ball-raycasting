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
//	//		 e��cuda ��������
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
//__global__ void crash_sharedMemory(Sphere *s, int num_sphere, int*d_crashnum)
//{
//	//--------�������ݵ��ڴ��� һ��block����---------
//	__shared__ Sphere rowSphere[32];
//	__shared__ Sphere colSphere[32];
//	int rowId1 = blockIdx.x * blockDim.x;
//	for (int i = rowId1; i < rowId1 + blockDim.x; i++)
//	{
//		if (i >= num_sphere)
//			break;
//		rowSphere[i-rowId1] = s[i];
//	}
//
//	int colId1 = blockIdx.y * blockDim.y;
//	for (int i = colId1; i < colId1 + blockDim.y; i++)
//	{
//		if (i >= num_sphere)
//			break;
//		colSphere[i- colId1] = s[i];
//	}
//	//�ȴ����ݿ��Լ���ok
//	__syncthreads();
//
//	//�õ�������ײС������
//	int s1 = threadIdx.x + blockIdx.x * blockDim.x;
//	int s2 = threadIdx.y + blockIdx.y * blockDim.y;
//
//	//�����Ϊx,y������С�������ײ���,�Գƾ��󣬼���һ��ľ���
//	if (s1 < num_sphere && s2 < num_sphere && s1 < s2)
//	{
//		REAL dx = rowSphere[threadIdx.x].x - colSphere[threadIdx.y].x;
//		REAL dy = rowSphere[threadIdx.x].y - colSphere[threadIdx.y].y;
//		REAL dz = rowSphere[threadIdx.x].z - colSphere[threadIdx.y].z;
//		REAL totalRadius = rowSphere[threadIdx.x].radius + colSphere[threadIdx.y].radius;
//		//�ж��Ƿ���ײ
//		if (dx*dx + dy*dy + dz*dz <= totalRadius * totalRadius)
//		{
//			s[s1].isCrash = true;
//			s[s2].isCrash = true;
//			atomicAdd(d_crashnum, 1);
//		}
//	}
//
//	__syncthreads();
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
//// globals needed by the update routine
//struct DataBlock {
//	//��� gpu �е�bitmap ������
//	unsigned char   *dev_bitmap;
//	//cpu�д��bitmap ������
//	CPUAnimBitmap  *bitmap;
//};
//
//void generate_frame(DataBlock *d, int ticks) {
//	
//	// initialize all integers of a device_vector to 0
//	/*int * d_crashNumList;
//	cudaMalloc(&d_crashNumList, sizeof(int)* SPHERES);
//	cudaMemset(d_crashNumList, 0, sizeof(int)* SPHERES);*/
//	
//	float totalTime = 0.0;
//	//��С�����ײ�ļ�������0
//	HANDLE_ERROR(cudaMemset(d_crashnum, 0, sizeof(int)));
//	//��С��ĸ��� copy��host �У�����ӡ����
//	HANDLE_ERROR(cudaMemcpy(h_crashnum, d_crashnum,
//		sizeof(int), cudaMemcpyDeviceToHost));
//	printf("init num of crash:  %d\n", (*h_crashnum));
//
//	START_GPU
//
//	//------------�ƶ���С��  --2000�� ----------------
//	kernelMoving << <64, 32 >> > (d_spheres, SPHERES);
//	END_GPU
//	totalTime += elapsedTime;
//	START_GPU
//	//--------------���С�����ײ------------------
//	//SPHERES ��2000 ��ʱ�� grid 64 * 64
//	//dim3    crashGrids(64, 64);
//	dim3    crashGrids(64, 64);
//	dim3    crashBlock(32, 32);
//	
//	crash_sharedMemory << <crashGrids, crashBlock >> > (d_spheres, SPHERES, d_crashnum);
//	
//	END_GPU
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
//	HANDLE_ERROR(cudaMemcpy(h_crashnum, d_crashnum,
//		sizeof(int), cudaMemcpyDeviceToHost));
//	printf("num of pair sphere crash:  %d\n", (*h_crashnum));
//	printf("total time:  %3.1f\n", totalTime);
//	printf("---------------------------------------------\n");
//
//}	
//
//// clean up memory allocated on the GPU
//void cleanup(DataBlock *d) {
//	HANDLE_ERROR(cudaFree(d->dev_bitmap));
//	//�ͷ�С����ײ�����Ŀռ�
//	HANDLE_ERROR(cudaFree(d_crashnum));
//	free(h_crashnum);
//}
//
////-------------------------main-------------------------------
//
//int main(void) {
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
//	HANDLE_ERROR(cudaMemcpy(d_crashnum, h_crashnum,
//		sizeof(int), cudaMemcpyHostToDevice));
//	
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