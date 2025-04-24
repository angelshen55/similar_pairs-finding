# similar_pairs-finding
**1.Overview:**

This repository realized the fingerprint method of Minhash, Simhash and Bit Sampling and used Banding method to efficiently decrease the range of possible similar pairs.

**2.Introduction:**

In the era of big data, large-scale text corpora -ranging from web crawls to user-generated content - often contain extensive, nearly duplicate content, such as paraphrased sentences, reordered paragraphs, or slightly modified versions of the same document. These redundancies introduce significant challenges for machine learning pipelines, particularly in natural language processing (NLP). Near-duplicates can distort training signals, amplify overfitting by overrepresenting specific linguistic patterns, and lead to data leakage between training and evaluation splits. Recent studies have shown that rigorous deduplication of training data can substantially improve model generalization and downstream performance. However, traditional pairwise comparison methods, which exhibit prohibitive O($n^2$) time complexity, are infeasible for modern datasets comprising millions of documents.

Our approach implements three key fingerprinting methods—**MinHash**, **SimHash**, and **Bit Sampling**—to generate compact document signatures that approximate textual similarity. These signatures are integrated into LSH-based indexing structures (e.g., banding techniques for MinHash, bucketing for SimHash) to identify candidate near-duplicate pairs with subquadratic complexity. We evaluate this system on the Wiki40B English dataset, focusing on detecting duplicates both within and between validation and test subsets.

The primary contribution of this work is a practical, scalable framework for text deduplication that maintains high accuracy while dramatically reducing computational requirements compared to traditional approaches.

**3.Related Work**

**3.1 MinHash**
Prior work in MinHash and document deduplication includes:
• Original MinHash (Broder, 1997): Introduced the core technique for estimating Jaccard similarity using minimum hash values, providing the theoretical foundation for our implementation.
• LSH Variants (Gionis et al., 1999): Extended MinHash with Locality-Sensitive Hashing (LSH) bands, significantly improving scalability by reducing the number of required comparisons.
• Recent Optimizations: ”MinHash virtually always outperforms SimHash when the data are binary” are provided with a theoretical answer validated by experiments (Li., Shrivastava, 2014)as a helpful tool in deciding which LSH to use.

**3.2 SimHash**
As early as 2002, researchers began exploring various similarity functions for Locality-Sensitive Hashing (LSH). Charikar (2002) introduced random hyperplane-based hash functions for vectors,a technique that facilitates the transformation of high-dimensional data into lower-dimensional representations and allows for the incorporation of weights for different data features. The practical utility of this approach was subsequently validated through large-scale empirical studies at Google by 2007. Google’s implementation of simhash on real-world datasets led to the development
of novel methodologies, such as employing Hamming distance calculations, to efficiently identify differences between documents. The detailed algorithms presented by Google for both online and batch queries highlight the increasing maturity of simhash applications. Nevertheless, areas for further refinement remain. Research conducted by Caitlin and Greg suggests that for realistic datasets, using fewer tags during key computation can enhance performance, and that the 0x00 pattern is a critical factor influencing efficiency. These findings offer potential avenues for improvement and innovation in the design of our LSH scheme.

**3.3 Bit Sampling**
3.3.1 Theoretical Foundations and Early Explorations
The core idea of Bit Sampling originates from studies on adapting Locality-Sensitive Hashing (LSH) to the Hamming distance. Charikar (2002) first proposed using random hyperplane–based hash functions to map high-dimensional vectors into binary signatures, and proved that these hash functions satisfy the locality-sensitive property: when two vectors have a small Hamming distance, the probability that their hash values coincide increases significantly. This theoretical result laid the mathematical foundation for the subsequent design of Bit Sampling algorithms. Early applications focused primarily on text similarity detection—for example, by computing the Hamming distance between document signatures to quickly identify duplicate web pages.
3.3.2 Algorithmic Optimizations and Cross-Domain Extensions
Recent research has progressed along two main directions: precision enhancement and computational efficiency. For high-dimensional dense datasets, hybrid architectures have been proposed, such as combining Bit Sampling with p-stable distribution LSH to support Euclidean distance metrics. Moreover,
neural network–driven approaches (e.g., LLSH) replace traditional hash functions with multi-layer perceptrons, achieving up to 15% higher accuracy in image retrieval tasks compared to conventional Bit Sampling, at the expense of added hardware dependencies due to GPU acceleration.
