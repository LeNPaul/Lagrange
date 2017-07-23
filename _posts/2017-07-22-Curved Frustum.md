---
layout: post
title: "Curved Frustum"
author: "Roy Berube"
categories: 3D graphics
tags: [3D, graphics]
image:
  feature: cutting.jpg
  teaser: cutting-teaser.jpg
  credit:
  creditlink:
---

### Curved Frustum

There are a few problems with the standard 3d view frustum with planes as edges:
1. Perspective only works for depth. Horizontal and vertical perspective are not applied. Imagine - in the real world - standing in front of and perpendicular to a fence that stretches as far as the eye can see to the left and right. The height of the fence will appear tallest directly in front. The height of the fence will diminish along its length to the left and right. The standard 3d frustum does not apply this perspective, and the fence appears the same height along its length. The fence remains at the same ratio of distance between the near and far frustum planes.
2. Rotating the camera in place reveals that the horizon can change depth. Objects near the edge of the view frustum can pop into view while they will disappear if the view is centered on them.
3. Spinning the camera in place (around the vertical axis is most noticable) reveals distortion along the edges of the frustum. Since the near plane is flat, near objects will be closer to the near frustum when they are near the edge of the view area. The objects will be larger as they approach the edge of the frustum, and will cause unnatural distortion.
4. Parallax distortion. Objects that hit the view plane are distorted if not at 90 degrees. Corners are the worst.
How can these problems be remedied? The answer lies in changing the frustum.

The traditional frustum uses planes, which allows for simple matrix transformations. Planes must be abandoned for the curved frustum, and it follows that the transformations will have to adapt also. More on that later.
Just two basic shapes are used to create the curved frustum: the sphere and cone. The near and far boundaries are sections of a sphere. The other four sides are created with cones. The sphere and cone origins are all at the camera origin. In illustration one the camera origin is the cursor where the green and red lines intersect. Blender was used to create this model. Boolean modifiers were used to cut away the edges of a cube to create the shape. The top edge, for example, started with a cone pointing straight up and then resized and flared out to the desired intersection with the base cube shape. A mirror modifier was then used to make the bottom edge the same.
A vital point to note is that at any distance from the origin, the width and height of the frustum are consistent � along a curve equi-distant to the origin, and not in a straight line. This is why the sphere and cone are used to define the 6 sides. This one point is what fixes the problems I pointed out earlier, but does leave a few more problems to solve. More on this later.
Illustration one has a good representation of the distortion problem with current 3D transformations. Look at the perspective view in the lower left quarter of the image. This angle is from the view origin, and the only part of the frustum showing is the near boundary. Remember, this is a portion of a sphere that centers on the view origin, so all of its points are at equal distance to the camera origin. In reality I would expect the near boundary to appear rectangular, which I will try to explain further. Switching the same view to an orthogonal camera - see Illustration two � has a result with straight edges.

This also does not represent reality, but helps to clarify the problem with the plane based frustum. Objects in orthogonal frustums do not change size with depth. The orthogonal view helps to make an important point: the near and far boundaries have the same height and width along those axis. For example, if you were to measure from the top to the bottom of the far boundary and sweep across its width, the distance would always be the same. Same thing with the near boundary. Note that this only holds true if the measuring points all share the same distance to the camera, which will be true for the near and far planes because of how they were formed from a sphere.
So how does this curved frustum remedy the problems outlined earlier?
Perspective now applies in vertical and horizontal directions, not just depth.  Take the straight fence perspective example from earlier. In this curved frustum, it does not retain the same distance to the near and far boundaries. Imagine if the fence touches the near boundary in the center and falls somewhere between the near and far boundaries at the left and right edges. The fence moves closer to the far frustum boundary as it moves away from the center. The fence will take up less vertical height as a ratio in the frustum, and as a result will appear smaller. See Illustration 2 � the far boundary is much larger than the near boundary. Objects that move away from the near and towards the far boundary will shrink in apparent size.
Spinning of the viewpoint camera will no longer distort an object towards the edges of the frustum. The object will maintain its point between the near and far boundaries, and will therefore maintain its size on screen. In the same way, distant objects no longer pop in and out of view. The far frustum boundary is equi-distant to the camera at all points so this problem is naturally solved.
Before this can be implemented, there are two big problems to solve.
1. Transformation matrix. The shapes used are simple enough, but the transformation matrix will take some thought. Much to do here.

2. Straight lines cannot be curves.
The transformation matrix will take some deep understanding of transformations to solve. I'm not close to solving this yet. The other problem is perhaps best described with an example.
Imagine the fence example used earlier. It's simplest shape would be a quad; just 2 polygons. Currently this is not a problem to render with a frustum using planar boundaries. However, the curved frustum introduces curves into the scene. The fence will be the widest in the center and will gradually taper off  the left and right. The fence edges will look like a curve. To solve this, the polygons for the fence need to be subdivided to represent the curve.  An example of this step is in Blender when rendering a panoramic camera.
The subdivision step should be tunable in the graphics engine. This could potentially add millions of polygons to the render pilepline so the user should be given some control over the quality.
A possible compromise solution could be a composite shader. One possible way: map the algorithm onto a bitmap of slightly larger than screen size. Each pixel of 32 bits indicates the x (RG channels) and y (BA channels) of the source pixel. Each pixel is shifted according to its location. The algorithm map can be generated at program start or on user changing settings. The simple lookup process should equate to fast processing for this shader.
One tricky part would be to find an algorithm that generates acceptable visual output. Maybe a direct input control panel to adjust a sample grid might work. This would be similar to projection TV alignment controls.
Update January 2017
One viable solution is to use a grid of cameras, each with a narrow field of view and using perspective transform.
Perspective transforms are used, so no change is needed here. The overall frustum shape will approach the proposed shape outlined above, and improve with a higher number of cameras.
It solves the problem of subdividing polygons, but also will be a source of visual distortion at the transition between cameras: a straight line will bend at the camera transition. The only way to minimize this is to increase the number of cameras, but this will result in a (estimated) massive performance hit. An ideal number of cameras to achieve visual perfection would be one camera per pixel.
Implementation and testing is ongoing currently with the Godot engine. It is unknown how much of a performance hit will result with the the expected hundreds of cameras. A grid of just 30 by 30 cameras would require 900 total.
