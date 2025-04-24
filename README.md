# similar_pairs-finding
Overview:

This repository realized the fingerprint method of Minhash, Simhash and Bit Sampling and used Banding method to efficiently decrease the range of possible similar pairs.

\section{Introduction}

In the era of big data, large-scale text corpora -ranging from web crawls to user-generated content - often contain extensive, nearly duplicate content, such as paraphrased sentences, reordered paragraphs, or slightly modified versions of the same document. These redundancies introduce significant challenges for machine learning pipelines, particularly in natural language processing (NLP). Near-duplicates can distort training signals, amplify overfitting by overrepresenting specific linguistic patterns, and lead to data leakage between training and evaluation splits. Recent studies have shown that rigorous deduplication of training data can substantially improve model generalization and downstream performance. However, traditional pairwise comparison methods, which exhibit prohibitive \(O(n^2)\) time complexity, are infeasible for modern datasets comprising millions of documents.

Our approach implements three key fingerprinting methods—\textbf{MinHash}, \textbf{SimHash}, and \textbf{Bit Sampling}—to generate compact document signatures that approximate textual similarity. These signatures are integrated into LSH-based indexing structures (e.g., banding techniques for MinHash, bucketing for SimHash) to identify candidate near-duplicate pairs with subquadratic complexity. We evaluate this system on the Wiki40B English dataset, focusing on detecting duplicates both within and between validation and test subsets.

The primary contribution of this work is a practical, scalable framework for text deduplication that maintains high accuracy while dramatically reducing computational requirements compared to traditional approaches.
