# Welcome to my presentation of the paper "Homological Inference of Embedding Dimensions in Neural Networks"
My name is Luciano Melodia and I work at the Chair of Computer Science 6, a data management chair, at the Friedrich-Alexander-University Erlangen-Nuremberg. I have been dealing with the question of how neural networks should be parameterized for a long time, not least because I was confronted with these problems during my undergraduate studies.
I had to realize that there is a rigorous theory for very few settings that can be made for training neural networks. In this paper, I would like to present a solution for a particular special case. The special case relates to the parameterization of the neural network for data that is structured in a very specific way. We have tried experimentally to apply this special case to arbitrary data, with some success.

# Overview
First, we will motivate our problem. Our work is based on the hypothesis that data lie in principle on a topological space that would ideally describe all possible actual data points. We give the topological space some more structure, since this still allows us to approximate the data space with arbitrary error. This is our data manifold.

We note that a neural network can also be described by a smooth manifold, where the coordinate functions are constructed by the weights and activation functions when chosen as a model of a smooth function. Often this manifold is chosen to be isomorphic to Euclidean space. It operates on the data manifold. We call it a neuromanifold.

The tool to describe the shape of the topological space we are looking for is an algebraic one, homology. For this we need a corresponding structure on a topological space, a triangulation. Within this triangulation, we look for loops. If we have enough good points from our topological space, then we can span such a structure over the points. The result is a simplicial complex. However, to ensure that the features we find are not artifacts, we examine the one-parameter family of the simplicial complex for homology. The final result is persistent homology, which indicates how long the representatives of the corresponding homology classes survive when the filtration parameter is changing.

We count the kth Betti numbers from a summary statistic of the persistence diagrams and use an assumption from Lie theory that the manifold should be a Lie group. Using a stronger constraint, namely that of connectedness, we are able to relate the homology groups directly to the dimension of such a manifold. This condition is also the main restriction for the general case, since it is not clear whether the data lie on a single manifold at all, or on several different ones.

I therefore conclude the talk with an outlook on how this work can be used for more general cases in the future, which special cases are also worth investigating, and in which particular case our theory has direct practical application.

# Manifolds and Lie groups
What is a manifold and what is meant by a Lie group? First we define a topological space X which is Hausdorff and has a countable basis (this is the second countability axiom). Now we want to make sure that the local neighborhoods are homeomorphic to the Euclidean n-dimensional space. This is also called locally Euclidean. Moreover, the coordinate functions that provide this homeomorphism should be smooth and compatible with each other, so that the reciprocal concatenations yield smooth functions again, with smooth inverses. Then one can exchange the coordinate maps and also cover the regions where the coordinate maps intersect. The collection of smooth coordinate maps that completely describe the manifold is called an atlas. The dimension of the manifold is given by the dimension of the Euclidean space to which the maps lead.

If the manifold has such a group structure that an operation of elements is smooth and has a smooth inverse, it is called a Lie group (after Sophus Lie).

# 1. The manifold of the data
Our assumption states that a set of points, such as we initially have or assume in Euclidean n-space, actually lies on a manifold embedded in the same n-space, with possibly much smaller dimension.

Our assumption of a connected Lie group allows us to decompose this space into simple factor spaces, namely actually Euclidean p-spaces where p is much smaller than n, and q-dimensional tori whose dimensions add to the dimension of the data manifold. This isomorphism allows us to draw equally elegant conclusions about the persistent homology groups.

# 2. The manifold of the data
In this example, we see six points. What manifold can we assume? We span a simplicial complex by choosing a parameter r and connecting all points whose Euclidean distance is less than this r with a 1-simplex. If there are more than two points closer than a given r, we form a 2-simplex for three points, a 3-simplex for 4 points, and so on. We see that for a step size of 0.2, a circle is created for two over a total of five filtration steps.

So our idea is to estimate this structure in this way. We relate the invariants we use to infer the data manifold to its dimension.
The neural network manifold is a different one, but it can approximate the data space. How many dimensions do I need for this? That is our key question! Also, can I use other activation functions to obtain just the topological structure of the data space when propagating forward through the layers of a neural network?

# The manifold of a neural network.
By choosing the weights and the corresponding activation functions, a parameterization of certain coordinate maps is chosen, which can change during forward propagation. Normally, the manifold into which the neural network maps is isomorphic to Euclidean space. As long as the rank of the Jacobian matrix does not change from layer to layer, the manifold also remains the same.

The neuromanifold can of course also be modeled by choosing appropriate activation functions, as is the case for spherical neural networks, for example.

# Realizing a good representation
So the key question we ask can be rephrased: How can we adapt the neuromanifold to fit our data?
...

# Building blocks: simplices
Consider a set of points X, from v0 to vn. These points are called affine independent, or in general position, if the vectors v0-vn, ..., vn-1-v-n are linearly independent, which means nothing more than that the points must not lie on a hyperplane of dimension smaller than n. The convex hull of these points is a simplex, which we write down as follows:
...
The dimension of a simplex is n. The i-th side of a simplex is obtained by removing an element i from the simplex.

# Examples
A simplex generalizes the concept of a triangle by including polyhedra and points in the definition. For example, a null simplex a is just a point, a 1-simplex is a segment, a 2-simplex is a triangle, a 3-simplex is a tetrahedron, and so on.
The coefficients for the simplices can be chosen from either a field or a ring. For our purpose, we use Z/(2Z) and can neglect the orientation of the simplices.

# Definition of a simplicial complex
A simplicial complex, let us call it K for example, is a finite union of simplexes which are expected to satisfy certain conditions. For example, every face of a simplex that lies in a simplicial complex must also be part of the simplicial complex. And nonempty intersections of two simplexes in K should in turn be faces of both.
On the left side we see a valid simplicial complex. On the right side in red a counterexample. Here we violate the second condition, because the sides of two 2-simplices are somehow displaced and a 1-simplice lies inside two 2-simplices.

# 1. Filtered simplicial complexes
It would be extremely difficult for us to decide for which parameter we should span a simplicial complex to reconstruct the structure of the topological space underlying the data. Furthermore, the set of points may well not represent every part of the space equally. Therefore, we consider the entire one-parameter family, or a finite part of it, which we call filtration. This is a nested sequence of simplicial complexes together with the inclusion map.
This inclusion simultaneously induces homomorphisms of the homology groups on the simplicial complexes, but more on that later.

# 2. Filtered simplicial complexes
The most commonly spanned simplicial complexes are the Cech complex, which, if spanned finely enough, satisfies theoretical guarantees on the invariants of the sought topological space, and the Vietoris-Rips complex, which can also satisfy these guarantees under mild circumstances but is easier to compute. The theorem which states this is called the Nerve Theorem. It shows that the Cech complex is homotopy equivalent to the intersection of the open balls around each point. These in turn can be viewed as open coverings of a topological space. The nerve is an abstract simplicial complex, a purely set-theoretic description of such a covering. Thus, given enough points, homotopy groups can be accurately captured in this way. Homotopy groups and homology groups have isomorphisms under mild circumstances, which in turn legitimizes what we are trying to do. 
Above, we again see the one-parameter family of a simplicial complex with the corresponding inclusion. For a very small parameter we obtain the initial point set.

If we increase the parameter, we create some 1-simplices and maybe 2-simplices. But we see immediately that the original set of points is contained in the new simplices complex. If we go one step further, we also generate 3-simplices, here in dark yellow. Again, we see that the previous two simplicial complexes are contained in this one. This can be continued like this.

The already mentioned Cech complex is spanned by placing balls with a certain radius around each point. The number of points whose intersection is non-empty forms a k-simplex, which is part of the simplicial complex for that parameter. All such simplices are added to the simplicial complex. Similarly, the Vietoris-Rips complex is spanned by adding all the simplices generated by the points that have a lower Euclidean norm than the selected parameter.

# Isomorphism of homology theories
Homology theories are largely isomorphic, and in a fairly precise way. For a triangulable manifold X and a simplicial complex K, it holds for certain fields that the simplicial homology on the simplicial complex is isomorphic to the singular homology of a triangulable topological space. The singular homology is defined over continuous maps, which can also be chosen smooth. From this, a theorem on smooth homology can be developed which is isomorphic to its singular counterpart. The De-Rham theorem provides an isomorphism of the chain complex of p-differential forms of a smooth manifold to its singular cohomology with real coefficients, which in turn is isomorphic to the homology with real coefficients. For the latter, then, the isomorphisms discussed first also hold.
At this point we are somewhat imprecise, even in the calculations, since this theorem applies to the field of real numbers, but is much more complicated for fields of positive characteristic. However, for the calculations we use fields of positive characteristic, even the simplest one, for arithmetic reasons. We are currently trying to generate a measurable quantity for the generated error through empirical experiments with persistent homology on fields of bodies with different characteristics.

# Persistent homology
Let us revisit persistent homology to make clear what we are measuring on the filtration. To do this, we consider the persistence module, a family of F-vector spaces for a field F body, this time arbitrary, and for real numbers i and j, such that there are F-linear mappings between the vector spaces Vi and Vj. These are called structure maps. For every pair i less than or equal to j, there is a k less than or equal to i less than or equal to j, so that compositing a map from Vk to Vi with a map from Vi to Vj yields the desired map from Vk to Vj.
If we consider an ordered set of simplicial complexes, ordered by increasing parameter, with the simplicial maps fij from the ith to the jth simplicial complex such that again i is less than or equal to j, then we also get a k less than or equal to i less than or equal to j and hence also a map by composition from the kth to the jth simplicial complex. The persistence module is given by the different homology groups of the ith simplicial complexes and the corresponding structure maps.
The persistence diagram shows the Betti numbers of the kth homology group during the filtration step i, so it is a diagram, a discrete finite subset of the one-parameter family that forms the persistence module.

# Persistent landscapes
Persistence landscapes are a functional representation of the aforementioned persistence diagrams, which are stable. They have been invented by Peter Bubenik and got a lot of attention in the topological data analysis community. The functional representation of a persistence diagram lies in a Banach space. One can think of them as a function, or equivalently as a sequence of functions for the kth homology groups.
We transform the persistence diagram into another coordinate system and look at the two lines x minus the birth component and the death component minus x. Now we look at the kth largest value of the expression min(x-bi,di-x) which gives lambda 1, the darkest yellow function, lambda 2, the dashed slightly lighter function and lambda 3, the lightest almost invisible function.

# Commutative abelian Lie groups
We recall and back to our assumption so that we consider the homology group of the data manifold in kth dimension as the homology group of a product space of hyperplanes and tori whose dimension in sum gives the dimension of our data manifold. We would like to infer this dimension. If we choose the dimension well, we can also estimate the structure of the data well. Via Künneth's theorem we learn how to count the homology groups in the persistence diagrams.
It behaves like this: the kth homology group of the product of two topological spaces is isomorphic to the direct sum of all factors whose index sum is k. The factors themselves are the tensor products of the ith with the jth homology group of the individual factor spaces.

# Computing dimensions

# Experimental results on cifar10 & cifar100

# 1. Results for the Betti numbers

# 2. Results for the Betti numbers

# Losses on cifar10 & cifar100

# Outlook