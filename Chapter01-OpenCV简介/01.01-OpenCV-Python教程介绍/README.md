## OpenCV  

OpenCV于1999年由Gary Bradsky在英特尔创立，并于2000年发布第一个版本。 随后Vadim Pisarevsky加入了Gary Bradsky负责管理英特尔的俄罗斯软件OpenCV团队。 2005年，OpenCV被用于Stanley车型，并赢得2005年DARPA挑战。 后来，它在Willow Garage的支持下由Gary Bradsky和Vadim Pisarevsky继续领导该项目积极发展。 OpenCV现在支持与计算机视觉和机器学习相关的众多算法，并且正在日益扩展。
OpenCV支持各种编程语言，如C ++，Python，Java等，可在不同的平台上使用，包括Windows，Linux，OS X（该系统已经更名为macOS），Android和iOS。 基于CUDA和OpenCL的高速GPU操作接口也在积极开发中。  
OpenCV-Python是OpenCV的Python API，结合了OpenCV C++ API和Python语言的最佳特性。


## OpenCV-Python
OpenCV-Python是基于Python的库，旨在解决计算机视觉问题。
Python是一种由Guido van Rossum开发的编程语言，它非常的流行，主要是因为它的简单性和代码可读性。 它使程序员能够用更少的代码行表达思想，并且不会降低代码的可读性。
与C/C++等语言相比，Python速度较慢。 也就是说，Python可以使用C/C++轻松扩展，这使我们可以在C/C++中编写计算密集型代码，并使用Python进行封装。 这给我们带来了两个好处：首先，代码与原始C/C++代码一样快（因为它在后台调用的实际是C++代码），其次，在Python中编写代码比使用C / C++更容易。 OpenCV-Python是由OpenCV C++实现并封装的Python库。
OpenCV-Python使用Numpy，这是一个高度优化的数据操作库，具有MATLAB风格的语法。 所有OpenCV数组结构都转换为Numpy数组。这也使得与使用Numpy的其他库（如SciPy和Matplotlib）集成更容易。

## OpenCV-Python教程
OpenCV引入了一组新的教程，它将指导您完成OpenCV-Python中提供的各种功能。 本指南主要关注OpenCV 3.x版本（尽管大多数教程也适用于OpenCV 2.x）。
建议事先了解Python和Numpy，因为本指南不涉及它们。 为了使用OpenCV-Python编写优化代码，必须熟练使用Numpy。

## OpenCV需要你!!!
由于OpenCV是一个开源计划，欢迎所有人为这个封装库的文档和教程做出贡献。 如果您在本教程中发现任何错误（从一个小的拼写错误到代码或概念中的一个令人震惊的错误），请通过在GitHub中clone OpenCV并提交pull请求来纠正它。 OpenCV开发人员将检查您的请求，给您重要的反馈，并且（一旦通过审核者的批准），它将合并到OpenCV中。 然后你将成为一个开源贡献者。
随着新模块被添加到OpenCV-Python，本教程将不得不进行扩展。 如果您熟悉特定算法并且可以编写包含算法基本理论和显示示例用法的代码的教程，请执行此操作。
记住，我们在一起可以使这个项目取得圆满成功！