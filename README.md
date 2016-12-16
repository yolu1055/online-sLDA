# online-sLDA

Implementation of [classification sLDA](https://www.cs.princeton.edu/~blei/papers/WangBleiFeiFei2009.pdf)[1] with [SVI](http://www.jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf)[2]. We use stochastic gradient ascent to optimize the logistic regression parameters.

Only the model itself. Omit the input, output, and main function.

Part of code is adapted from Matthew Hoffman's online LDA code.

## References

[1] Chong, Wang, David Blei, and Fei-Fei Li. "Simultaneous image classification and annotation." Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on. IEEE, 2009.

[2] Hoffman, Matthew D., et al. "Stochastic variational inference." Journal of Machine Learning Research 14.1 (2013): 1303-1347.
