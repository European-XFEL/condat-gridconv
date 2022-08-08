## Hexagonal-cartesian grid conversion

This is an implementation of (part of) the algorithm described in
Condat et al. [*Reversible, fast, and high-quality grid conversions*](https://lcondat.github.io/publis/Condat-TIP-2008.pdf),
IEEE Transactions on Image Processing, vol. 17, no. 5, pp. 679-693, May 2008.
It transforms data sampled on a hexagonal grid, such as an X-ray detector with
hexagonal pixels, into a conventional cartesian lattice.

Specifically, it implements what that paper describes as a Type II fractional
delay filter, with N=2.

This package is based on code written by Andreas Scherz and Rafael Gort. 
