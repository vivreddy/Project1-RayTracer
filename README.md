-------------------------------------------------------------------------------
CIS565: Project 1: CUDA Raytracer
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
Implementation Details :
-------------------------------------------------------------------------------
In this project I have implemented a CUDA based raytracer capable of
generating raytraced rendered images extremely quickly. 
Here rays are sent through each pixel, and every ray that is sent through a pixel 
will first check the geometry that it has hit. The geometries index is stored so
that the color of the geometry can be retrieved while finding the final color.

Once a ray has hit a particular geometry, the intersection point is calculated.
An offset is added to this intersection point so that there is no self collision.
Then a ray is sent from this point towards the light to check for shadows . 
If there is a object between the intersection point and the light then the pixel 
is in a shadow.

Also from the same intersection point a reflected ray is calculated along the 
normal of the geometry surface and is first checked if the geometry is reflective.
If its reflective , then the color of the object that the reflected ray hits is 
contributed to the objects diffuse color.

Also specular highlights are calculated as a cosine funcion distribution by 
calculating the reflected ray along the normal and using the specular component.

In order for the intersections to work, slab method is used while checking for 
the intersection of a cube. And mesh intersection tests are carried out by 
adding using triangle intersection which is based on the parametric 
representation of a plane. Both the intersection tests are an extension of my
460/560 implementation ,but for CUDA.

All the materials and vertices of the mesh data are copied to the global memory( CUDA)
using cudamemcopy. 

Additional implementation : 

1)Reflection 

2)Area lights & Soft Shadows

3)Object mesh loader 


1) Reflection - The reflections implemented here are capable of having a trace depth
   of 1. Here the intersection point is considered as the eye and a reflected ray is 
   calculated. the geometry to which the reflected ray collides is noted and is 
   contributed to the output color.
   
2) Area lights and soft shadows - To implement area lights, a number of point lights
   are uniformly distributed along the surface. Then from each point,a shadow feeler
   ray is sent to each light and the sum of the color contributed is taken and is
   averaged. Since the matrix multiplication using the light geometry was flipping
   the area lights, a hardcoded matrix was created in order to distribute the point
   lights properly.
   
3) Object mesh loader - The object loader here reads a .obj file and saves all the 
   vertices and the faces. Then the vertices are sorted and saved in a vector in the
   order of the saved faces. This object loader is an extension from my 563 cloth 
   simulation object loader. 



-------------------------------------------------------------------------------
Screen Shots :
-------------------------------------------------------------------------------
Here are the output images from my CUDA ray tracer 

1) With Soft shadows and reflections. Area light at the center .


![alt tag](https://raw.github.com/vivreddy/Project1-RayTracer/master/renders/1.png)




2)With Soft shadows, reflections and object loader.
(Area light in poitive z-axis ie out of the box)


![alt tag](https://raw.github.com/vivreddy/Project1-RayTracer/master/renders/3.png)




-------------------------------------------------------------------------------
Video :
-------------------------------------------------------------------------------
Here is a 30 second video of the running CUDA ray tracer : 


[Video Demo](http://youtu.be/7nRIo22uiFM)





-------------------------------------------------------------------------------
Performance Evaluation : 
-------------------------------------------------------------------------------
The performance evaluation was done by changing the number of threads per block,
the parameter tile size was varied to check the cuda event time taken.

The scene here for which the evaluation was conducted consisted of reflections,
soft shadows and uses 400 point lights which are uniformly distributed on the
light geometry. 

The results are tabulated below for various tile sizes : 


As we can see from the table , the fastest time the image was rendered was when 
the I used 8 tiles. The reason why the CUDA event time is not inversely 
proportional as the number of threads per block increase is because :
->Increase in number of threads does not guarantee a speed up
->Usually an ideal number of threads are tested to see the fastest among them.
->Adding more threads helps upto a certain point after that they cause 
  performance degradation
->If there is lot of memory intensive work then having more number of threads
  does not help increase the speed. 




-------------------------------------------------------------------------------
