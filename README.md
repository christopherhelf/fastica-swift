# FastICA-Swift

This repository provides methods implementing the [FastICA](http://research.ics.aalto.fi/ica/fastica/) algorithm based on the [libICA C library](http://tumic.wz.cz/fel/online/libICA/) in Swift. I borrowed the matrix class from 
[Surge](https://github.com/mattt/Surge) as well as other functionality, such as matrix inversion, transposition, and basic arithmetic operations on matrices. 

Usage is fairly simple, check out the Test.swift file. Below is an example of sample input and output, which should resemble Example 2 in the [CRAN Documentation](http://cran.r-project.org/web/packages/fastICA/fastICA.pdf). 

Here are two signals mixed with two different matrices.

![Input](https://raw.github.com/christopherhelf/fastica-swift/master/input.png)

Here is the resulting output.

![Output](https://raw.github.com/christopherhelf/fastica-swift/master/result.png)



