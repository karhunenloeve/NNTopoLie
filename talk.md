# Welcome to my presentation "Homological Inference of Embedding Dimensions in Neural Networks"
My name ist Luciano Melodia and I work at the chair of computer science 6 at the Friedrich-Alexander University in Erlangen. I realized that there is few theory for parameter choice of neural networks. In this paper I would like to present a solution for a particular special case.

# Manifolds and Lie groups
First, we define smooth manifolds, as topological space X which is Hausdorff and has a countable basis. Moreover, the coordinate functions are homeomorphic to Euclidean n-space and should be smooth and compatible with each other. The dimension of the manifold is given by the dimension of the Euclidean space into which the maps lead.

If the manifold has such a group structure that an operation of elements is smooth and has a smooth inverse, it is called a Lie group (after Sophus Lie).

# 1. The manifold of the data
Our assumption states that a set of points lies on a (smooth) manifold, with possibly much smaller dimension than the data set suggests.

Our assumption of a connected Lie group allows us to decompose this space into simple factor spaces, namely Euclidean p-spaces and q-dimensional tori. This isomorphism allows us to draw equally elegant conclusions about persistent homology groups.

# 2. The manifold of the data
In this example, we see six points. What manifold can we assume? We span a simplicial complex by choosing a parameter r and connecting all points whose Euclidean distance is less than this r with a 1-simplex. If there are more than two points closer than a given r, we form a 2-simplex for three points, a 3-simplex for 4 points, and so on. We see that for a step size of 0.2, a circle is created for two over a total of five filtration steps.

So our idea is to estimate the topological structure in this way. We relate the invariants we use to infer the data manifold to its dimension.

# The manifold of a neural network.
Choosing the weights and the corresponding activation functions, a parameterization of certain coordinate maps is chosen, which can change during forward propagation. 

The neuromanifold can of course also be modeled by choosing appropriate activation functions, as is the case for spherical neural networks, for example.

# Realizing a good representation
So the key question we ask can be rephrased: How can we adapt the neuromanifold to fit our data?

First, we take suitable assumptions on the dataset and its manifold.
Second, we measure the persistent homology groups on the filtration.
Third, we infere possible manifolds from these invariants.
Forth, we relate the invariants to the manifolds and infere the embedding dimensions.
And last but not least, we seek for approximate solutions if our assumption seems to fail.

# Isomorphism of homology theories
Homology theories are largely isomorphic, and in a fairly precise way. For a smooth manifold X and a simplicial complex K, it holds for certain fields that the simplicial homology is isomorphic to singular homology. The singular homology is defined over continuous maps, which can also be chosen to be smooth. From this, a theory on smooth homology can be developed which is isomorphic to its singular counterpart. The De-Rham theorem provides an isomorphism of the chain complex of p-differential forms of a smooth manifold to its singular cohomology with real coefficients. For the latter, then, the isomorphisms discussed also hold.
At this point we are somewhat imprecise, even in the calculations, since this theorem applies to the field of real numbers, but is much more complicated for fields of positive characteristic. However, for the calculations we use fields of positive characteristic. We are currently trying to generate a measurable quantity for the generated error through empirical experiments with persistent homology and torsion.

# Persistent homology
We consider the persistence module, a family of F-vector spaces for a field F and for real numbers i and j, such that there are F-linear maps between the vector spaces Vi and Vj.
We also consider an ordered set of simplicial complexes, ordered by increasing parameter, with the simplicial maps fij from the ith to the jth simplicial complex. The persistence module is given by the different homology groups of the ith simplicial complexes and the corresponding structure maps.
The persistence diagram shows the Betti numbers of the kth homology group during the filtration step i, so it is a diagram, a discrete finite subset of the one-parameter family that forms the persistence module.

# Persistent landscapes
Persistence landscapes are a functional representation of the aforementioned persistence diagrams, which are stable. They have been invented by Peter Bubenik and got a lot of attention in the topological data analysis community. The functional representation of a persistence diagram lies in a Banach space. One can think of them as a function, or equivalently as a sequence of functions for the kth Betti numbers.

# Commutative abelian Lie groups
We recall our assumption that we consider the homology group of the data manifold in kth dimension as the homology group of a product space of hyperplanes and tori. We would like to infer its dimension. Via Künneth's theorem we learn how to relate the persistent homology groups to the dimension.
It behaves like this: the kth homology group of the product of two topological spaces is isomorphic to the direct sum of all factors whose index sum is k. The factors themselves are the tensor products of the ith with the jth homology group of the individual factor spaces.
Now by replacing the product space with our assumption, we get the following isomorphism. We get r indices depending on how many components we have. The question remains, which of these summands are left in the direct sum? What tensor product do we end up with and what dimension does it have?

# Computing dimensions
We obtain for the 0-dimensional homology group of any 1-sphere the ring of integers. The same is true for the one-dimensional homology group and all other homology groups are trivial, that is for an i greater than or equal to 2, H of S1 is zero. By applying Künneth's theorem from the previous slide, we obtain an isomorphism of the zeroth homology group in Z for the connected components which are trivially contractible, and for the kth homology group of a torus we are left with only values for the indices ij which give either 0 or 1. Thus the kth homology group is isomorphic to a power of the binomial coefficient q choose k, with q the dimension of the torus.

# Experimental results on cifar10 & cifar100
These are the persistence landscapes computed over the entire dataset for cifar10 on the left and cifar100 on the right. We see differences at the time of occurrence of the different k-dimensional representants of the respective homology group, but count similar numbers. Epsilon i plus epsilon j+1 divided by two is the coordinate transformation applied to each point of the persistence diagram. The upper axis shows the initial persistent feature that never disappears, since there is expected to always be at least one connected component. By convention, this is paired with the value for death to infinity. We will see immediately in the counting of the representatives of the different classes that the pure number of persistent features does not differ much, but the topological space has been stretched and compressed in one direction or the other by adding extra dimensions and a higher resolution (cifar100 compared to cifar10).

# 1. Results for the Betti numbers
Please look first only at the left side of the table, with the counted representatives of the kth homology group. I emphasize again, the calculations here are done over Z2, over a field the torsor disappears. The first observation we make, the higher resolution dataset shows fewer representatives in the higher homology groups. Obviously, components are being 'filled in'. The second observation is that with the knowledge of the homology groups of tori, we immediately realize that this cannot be a torus.
The homology groups of a torus have a certain symmetry. A one point space has the zeroth homology group of 1 and all others are trivial. A 1-torus has a connected component and a single loop (it is a sphere), so H_1=1, thus we get homology groups H_0 = 1 for the 0-torus and H_0 = 1 and H_1 = 1 for the 1-torus. For the 2-torus the pattern is 1-2-1, for the three torus 1-3-3-1, for the 4-torus 1-4-6-4-1 and for the 5-torus 1-5-10-10-5-1. But since it is connected, H_0 is always equal to 1. Our object has several connected components, so we conclude that the manifold cannot be connected. How can we counter this problem? At this point the empirical experiment starts for us. If it is really a torus, the required dimensions can be derived exactly. This is not a theoretical by-product! In practice, there is a tremendous amount of cases where toric varieties or manifolds can be assumed. We will mention related works which do research on this.

# 2. Results for the Betti numbers
But there is a sufficiently good way to use this theory in a general setting. We seek for an n-torus with as large as possible n and consider which representants of the respective homology group can be represented by it and subtract the respective betti numbers from the measured ones. Then, for the remaining representants, we again choose a k-torus, with k as large as possible, and try to cover as many representants as possible with it. We subtract all representants which are also contained in this k-torus and continue until all representants can be covered. Since we obviously do not have a torus, this integer equation according to Künneth does not work out and we get a certain surplus. On the right hand side we see the dimensions we would need to embed tori with the kth homology groups in a given real space. Since the integer equation does not exactly add up, we have written down the smallest and largest possibilities for embedding.

# Losses on cifar10 & cifar100
We trained autoencoders for both datasets, cifar10 and cifar100, by adding noise to the images and mapping them to the noise-free originals. For our experiment, we use invertible neural networks to model the differentiable structure as the same from layer to layer. Due to the structure of the neural network, which makes the upper right triangular matrix trivially invertible with the lower left triangular matrix of a square matrix, we need to double the embedding dimension. For cifar10 our estimation is extremely good, that neural network learns the denoising without large explosions of the gradient and in the best possible way starting from 272 neurons in the hidden layer. The situation is somewhat worse, but only marginally, for cifar100, where we also succeed in this delimitation well, except for individual parameterizations. These are just above our determined threshold. We attribute them to noise, since the persistence landscapes were not measured on the noisy but on the original images. Of course, the noise is also on a topological space that is quite different from the data.

# Outlook
After all, I promised that there is a broader application for this theory, which we are already working on and which will also come into practical use in the further course of our work, namely that of time series. The so-called sliding window embedding, which I'm sure many of you are familiar with, has been studied by Jose Perea for its mathematical properties. Here, the time series are embedded as curves on a torus or densely within a torus. This also allows us to determine the torus exactly, provided there is enough data.
We have already seen that for arbitrary data sets the assumption of connected structure cannot possibly hold, for time series it is trivially satisfied. How can we generalize our approach? What decomposition theorems exist for other manifolds under less stringent assumptions?
Last, due to time constraints, we used ordinary fully connected neural networks. So these are isomorphic to Euclidean n-space and we can embed the data manifold in it. But it is not robust to outliers. Artifacts can be learned along with it. It would clearly be better to tailor the neural network manifold to our data set, e.g., by having the neuromannifold model a commutative Lie group or toric variety.