// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
  ray r;

  	glm::vec3 A,B,M,H,V,P,D ;

	// Calculating the unknown angel theta
float angelv = tan(float(fov.y/57.295779));
float EtoM = (resolution.y/2)/angelv;
float Fovx = atan(float ((resolution.x/2)/EtoM) )*57.295779;
float angelh = tan(float((Fovx)/57.295779));

// Finding the unknown parameters that are required for ray march
A = glm::cross(view,up); 
B = glm::cross(A,view);
M = eye + view;
H = (glm::normalize (A))* sqrt(view.x*view.x + view.y*view.y +view.z*view.z )*(angelh) ;
V = (glm::normalize( B))* sqrt(view.x*view.x + view.y*view.y +view.z*view.z )*(angelv) ; 
float sx,sy ;

 sx=(x/float (resolution.x-1));
 sy=(float(resolution.y-1-y)/float(resolution.y-1)); // To make sure easy BMP displays properly the formula is modified.
 P = M + (((2*sx)-1)*H) + (((2*sy)-1)*V); 
 glm::vec3 aa = P - eye;
 D = glm::normalize(aa) ;



  r.origin = eye;
  r.direction = D;
  return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms,glm::vec3* color){          //

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  int p = numberOfGeoms;
  glm::vec3 norms[20],ips[20];
  float rips[20];
  if((x<=resolution.x && y<=resolution.y)){
   ray r = raycastFromCameraKernel(resolution,time,x,y,cam.position,cam.view,cam.up,cam.fov);

  for(int i=0; i < numberOfGeoms ; i++)
  {
		
		if(geoms[i].type == SPHERE)
		{
		  rips[i] = sphereIntersectionTest(geoms[i],r, ips[i], norms[i]);
		  //colors[index] = materials[geoms[i].materialid].color;
		}  
		else if (geoms[i].type == CUBE)
		{
		  rips[i] = boxIntersectionTest(geoms[i],r, ips[i], norms[i]);
		  //colors[index] = materials[geoms[i].materialid].color;
		}
  
  }

  float temp = 123456.0;
  int obno=0,flag=0;
    for(int i=0; i < numberOfGeoms ; i++)
  {
	
     if(rips[i] == -1)
		rips[i] = 123456.0 ;
	     if(rips[i] <= temp )
	 {
		 temp = rips[i];
		 obno = i;      // Storing the index where the least hit value was found so that the corresponding normal could be got back 
		 
	 }
	 if(rips[i] == rips[i+1])
	 {
		flag++;
	 }
	 if(i == numberOfGeoms-1)
	 {
		if(flag == numberOfGeoms)
			temp = 123456.0;
	 
	 }
	
  }
  if( temp == 123456.0)
{
	colors[index] = glm::vec3(0,0,0);
}
else
{
	  glm::vec3 LPOS = glm::vec3(0,7,7);
	  //ipoint = (camera + thit * direc);
	

	// Calculating the specular component
	    
		glm::vec3 lig =  ips[obno] - LPOS ;
		glm::vec3 ref1  =  lig - (2.0f * norms[obno] * (glm::dot(norms[obno],lig)));     // glm::normalize(glm::reflect(-(lig) , glm::normalize(norms[obno])));
		float dt = glm::dot(glm::normalize(cam.position),glm::normalize(ref1));
		if(dt < 0)
			 dt = 0;
		 float sc = pow(dt,30);


	 // Shadows
	   int shadow = 0 ;
	   float sit = 0; ;
	   glm::vec3 neyep = ips[obno] + ref1 * 0.001f ;
	   ray s;
	   s.origin = neyep;
	   s.direction = glm::normalize(LPOS-ips[obno]);
	   float len = glm::length(neyep - LPOS) ;
	  // float sit;
	   glm::vec3 htemp,ntemp;
	    for(int i=0 ; i <numberOfGeoms ; i++)
        {
		if(i != obno)
		{
			if(geoms[i].type == SPHERE)
		{
		  sit = sphereIntersectionTest(geoms[i],s, htemp, ntemp);
		  //colors[index] = materials[geoms[i].materialid].color;
		}  
		else if (geoms[i].type == CUBE)
		{
		  sit = boxIntersectionTest(geoms[i],s,  htemp, ntemp);
		  //colors[index] = materials[geoms[i].materialid].color;
		}
         //sit = rayintersect(allmynodes[j],ipoint, normalize(LPOS-ipoint));
	
	     if(sit != -1)
	       {
			if ( glm::length(htemp - neyep) < len)
				shadow = 1 ;   //  Shadow == 1 means the point of interesection is under a shadow , if 0 then no shadow
			else
				shadow = 0 ;
		 }
		}	 
		}

	// Reflections ////////////////////////////////////////////////////////////////////

		ray ref;
		ref.origin = neyep;
		ref.direction = ref1 ;
		int rbno = 0;
		float rrps[20];
		  for(int i=0; i < numberOfGeoms ; i++)
  {
		
		if(geoms[i].type == SPHERE)
		{
		  rrps[i] = sphereIntersectionTest(geoms[i],ref, htemp, ntemp);
		  //colors[index] = materials[geoms[i].materialid].color;
		}  
		else if (geoms[i].type == CUBE)
		{
		  rrps[i] = boxIntersectionTest(geoms[i],ref, htemp, ntemp);
		  //colors[index] = materials[geoms[i].materialid].color;
		}
  
  }
		  for(int i=0; i < numberOfGeoms ; i++)
  {
	
     if(rrps[i] == -1)
		rrps[i] = 123456.0 ;
	     if(rrps[i] <= temp )
	 {
		 temp = rrps[i];
		 rbno = i;      // Storing the index where the least hit value was found so that the corresponding normal could be got back 
		 
	 }
	 if(rrps[i] == rrps[i+1])
	 {
		flag++;
	 }
	 if(i == numberOfGeoms-1)
	 {
		if(flag == numberOfGeoms)
			temp = 123456.0;
	 
	 }
	
  }
		  glm::vec3 relcolor(0,0,0);
       if(geoms[rbno].type == SPHERE)
			relcolor =  color[geoms[rbno].materialid] * 0.1f;
	   


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //(LCOL * pnod->color * (dot(N,normalize(LPOS - ipoint))))
		if(shadow == 0)
		{
    colors[index] = color[geoms[obno].materialid] *  glm::dot(norms[obno],glm::normalize(LPOS - ips[obno]))   +   glm::vec3(1,1,1) * sc   + relcolor;   //glm::vec3(1,1,1) * 
		}
		else
		{
		colors[index]  = color[geoms[obno].materialid] * 0.3f;
		}
}
  //colors[index] = glm::vec3(fabsf(r.direction.x),fabsf(r.direction.y),fabsf(r.direction.z));
 
  }

  

 //colors[index] = generateRandomNumberFromThread(resolution, time, x, y);
   
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  //package material color
  glm::vec3* color = NULL;
  cudaMalloc((void**)&color, numberOfMaterials*sizeof(glm::vec3));
   for(int i=0;i <  numberOfMaterials ; i++)
   {
   glm::vec3* tcolor = color + i ;
  cudaMemcpy(tcolor , &materials[i].color, sizeof(glm::vec3), cudaMemcpyHostToDevice);
  }
  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms,color);  //

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
