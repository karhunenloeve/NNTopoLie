# Invertible Resnets up to Homology Equivalence
Neural nets have been used in an elusive number of scientific disciplines. Nevertheless, their parameterization is largely unexplored. Dense nets are the coordinate transformations of a manifold from which the data is sampled. After processing through a layer, the representation of the original manifold may change. This is crucial for the preservation of its topological structure and should therefore be parameterized correctly. We discuss a method to determine the smallest topology preserving layer for an invertible residual net. We consider the data domain as abelian connected matrix group and observe that it is decomposable into the *p*-dimensional Euclidean space and the *q*-torus. Persistent homology allows us to count its *k*-th homology groups. Using Künneth's theorem, we count the *k*-th Betti numbers. Since we know the embedding dimension of Euclidean space and the *1*-sphere, we parameterize the bottleneck layer with the smallest possible matrix group, which can represent a manifold with those homology groups. Resnets guarantee smaller embeddings due to the dimension of their state space representation.

**Keywords**: Embedding Dimension, Parametrization, Persistent Homology, Neural Networks and Manifold Learning.

    @article{DBLP:journals/corr/abs-2004-02881,
     author    = {Luciano Melodia and
                  Richard Lenz},
     title     = {Parametrization of Neural Networks with Connected Abelian Lie Groups
                  as Data Manifold},
     journal   = {CoRR},
     volume    = {abs/2004.02881},
     year      = {2020},
     url       = {https://arxiv.org/abs/2004.02881},
     archivePrefix = {arXiv},
     eprint    = {2004.02881},
     timestamp = {Wed, 08 Apr 2020 17:08:25 +0200},
     biburl    = {https://dblp.org/rec/journals/corr/abs-2004-02881.bib},
     bibsource = {dblp computer science bibliography, https://dblp.org}
   }
