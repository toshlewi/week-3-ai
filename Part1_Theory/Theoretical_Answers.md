# Part 1: Theoretical Understanding

## 1. Short Answer Questions

### Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?

**Primary Differences:**

*   **Graph Definition:** TensorFlow traditionally used a static computation graph (Define-and-Run), where you define the entire model architecture before running it. This allows for significant optimizations. PyTorch, on the other hand, uses a dynamic computation graph (Define-by-Run), which is more flexible and intuitive, as the graph is built on-the-fly as operations are executed. This makes debugging easier. TensorFlow 2.x adopted "Eager Execution" by default, making it more like PyTorch in this regard, but the underlying philosophies still influence the libraries.
*   **API Level:** PyTorch is often considered more "Pythonic" and has a more object-oriented API that integrates smoothly with the Python ecosystem. TensorFlow's API, especially in version 1.x, was seen as more verbose and complex. TensorFlow 2.x, with the Keras integration, has become much more user-friendly.
*   **Deployment:** TensorFlow has historically been stronger for production deployment, with tools like TensorFlow Serving and TensorFlow Lite for mobile and embedded devices. PyTorch has been catching up with tools like TorchServe.
*   **Visualization:** TensorFlow comes with a powerful visualization tool called TensorBoard, which is excellent for monitoring training, visualizing the model graph, and more. While PyTorch can be integrated with TensorBoard, it's not as tightly coupled.

**When to Choose:**

*   **Choose PyTorch when:**
    *   You are a beginner and want a more intuitive, Pythonic experience.
    *   You need to do rapid prototyping and research, especially in fields like NLP where dynamic graphs are beneficial.
    *   You want easier debugging capabilities.
*   **Choose TensorFlow when:**
    *   You need to deploy models to production, especially on mobile or embedded devices.
    *   You are working in a large-scale distributed environment.
    *   You want to use a well-established, industry-standard tool with extensive documentation and community support.

### Q2: Describe two use cases for Jupyter Notebooks in AI development.

1.  **Exploratory Data Analysis (EDA) and Visualization:** Jupyter Notebooks are ideal for interactively exploring datasets. You can load data, clean it, and visualize it using libraries like Matplotlib, Seaborn, and Plotly. The cell-based structure allows you to see the results of your code immediately, making it easy to iterate and understand the data's characteristics before building a model.
2.  **Model Prototyping and Experimentation:** When developing an AI model, you often need to experiment with different architectures, hyperparameters, and preprocessing steps. Jupyter Notebooks allow you to build and train models in a modular way. You can have separate cells for data loading, model definition, training, and evaluation, making it easy to modify and re-run specific parts of the workflow without re-executing everything.

### Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?

Basic Python string operations (e.g., `.split()`, `.lower()`, `.replace()`) are limited to simple text manipulation. spaCy, on the other hand, is a full-fledged NLP library that provides a wide range of linguistic annotations and pre-trained models.

*   **Tokenization:** spaCy's tokenizer is language-specific and considers punctuation and other linguistic rules, providing more meaningful tokens than a simple `.split()`.
*   **Part-of-Speech (POS) Tagging:** spaCy can identify the grammatical role of each word (e.g., noun, verb, adjective).
*   **Named Entity Recognition (NER):** It can recognize and extract entities like names of people, organizations, locations, etc.
*   **Dependency Parsing:** spaCy can analyze the grammatical structure of a sentence, which is crucial for understanding relationships between words.
*   **Word Vectors:** spaCy comes with pre-trained word embeddings, allowing you to represent words as numerical vectors for use in machine learning models.

In essence, spaCy transforms raw text into structured linguistic information, enabling much more sophisticated NLP tasks than what's possible with basic string methods.

## 2. Comparative Analysis

### Compare Scikit-learn and TensorFlow in terms of:

| Feature                   | Scikit-learn                                                                        | TensorFlow                                                                                                   |
| ------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Target Applications**   | Classical machine learning algorithms (e.g., regression, classification, clustering, dimensionality reduction). It is not primarily for deep learning. | Deep learning and neural networks. It can be used for classical ML, but it's more complex for those tasks. |
| **Ease of Use for Beginners** | Excellent for beginners. It has a consistent and simple API (`fit`, `predict`, `transform`). | Steeper learning curve, especially with its lower-level API. However, the high-level Keras API (integrated into TensorFlow 2.x) makes it much more accessible. |
| **Community Support**     | Large, active community with extensive documentation and many tutorials and examples. | Very large and active community, backed by Google. It has a vast ecosystem of tools, libraries, and resources. |

</rewritten_file> 