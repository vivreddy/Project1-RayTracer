// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef RAYTRACEKERNEL_H
#define PATHTRACEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include <vector>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void cudaRaytraceCore(uchar4* pos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, std::vector<glm::vec3> mymainpoints);
float __device__ meshIntersectionTest(staticGeom curGeom,ray s,glm::vec3* myvertex, int numVertices, glm::vec3& htemp, glm::vec3& ntemp);
int __device__ checkForShadows(staticGeom* geoms,int numberOfGeoms,ray s, glm::vec3* myvertex, int numVertices,glm::vec3 LPOS,int index);
int __device__ getreflectedcolor(staticGeom* geoms,int numberOfGeoms,ray s, glm::vec3* myvertex, int numVertices,glm::vec3& htemp, glm::vec3& ntemp);
void __device__ calculateColoratPoint(staticGeom* geoms,glm::vec3 dips,glm::vec3 LPOS,glm::vec3 dnorm,glm::vec3 relcolor,int obno, glm::vec3& mycolor,glm::vec3 cpos,material* cudamat);
#endif
