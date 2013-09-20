-------------------------------------------------------------------------------
CIS565: Project 1: CUDA Raytracer
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
Implementation Details :
-------------------------------------------------------------------------------
In this project I have implemented a CUDA based raytracer capable of
generating raytraced rendered images extremely quickly. 



-------------------------------------------------------------------------------
Screen Shots :
-------------------------------------------------------------------------------
Here are the output images from my CUDA ray tracer 

1) With Soft shadows and reflections. Area light at the center .







2)With Soft shadows, reflections and object loader.







-------------------------------------------------------------------------------
Video :
-------------------------------------------------------------------------------
Here is a 30 second video of the running CUDA ray tracer : 








-------------------------------------------------------------------------------
Performance Evaluation : 
-------------------------------------------------------------------------------
The performance evaluation is where you will investigate how to make your CUDA
programs more efficient using the skills you've learned in class. You must have
perform at least one experiment on your code to investigate the positive or
negative effects on performance. 

One such experiment would be to investigate the performance increase involved 
with adding a spatial data-structure to your scene data.

Another idea could be looking at the change in timing between various block
sizes.

A good metric to track would be number of rays per second, or frames per 
second, or number of objects displayable at 60fps.

We encourage you to get creative with your tweaks. Consider places in your code
that could be considered bottlenecks and try to improve them. 

Each student should provide no more than a one page summary of their
optimizations along with tables and or graphs to visually explain and
performance differences.

-------------------------------------------------------------------------------
