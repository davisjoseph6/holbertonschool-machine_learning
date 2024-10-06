Transformer Applications
 Master
 By: Alexa Orrico, Software Engineer at Holberton School
 Weight: 4
 Your score will be updated as you progress.
 Manual QA review must be done (request it when you are done with the project)
Description


Resources
Read or watch:

How machines Read
Sub-word tokenizers
Summary of the tokenizers
Subword Tokenization
Notes on BERT tokenizer and model
What is AutoTokenizer?
Training a new tokenizer from an old one
TFDS Overview
How Transformers Work: A Detailed Exploration of Transformer Architecture
References:

tfds
tfds.load
AutoTokenizer
encode
tf.py_function
TFDS Keras Example
tf.linalg.band_part
Customizing what happens in fit
Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

General
How to use Transformers for Machine Translation
How to write a custom train/test loop in Keras
How to use Tensorflow Datasets
Requirements
General
Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.9)
Your files will be executed with numpy (version 1.25.2) and tensorflow (version 2.15)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
All of your files must be executable
A README.md file, at the root of the folder of the project, is mandatory
Your code should follow the pycodestyle style (version 2.11.1)
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
Unless otherwise stated, you cannot import any module except import transformers , import tensorflow_datasets as tfds and also import tensorflow as tf for some other tasks.
TF Datasets
For machine translation, we will be using the prepared Tensorflow Datasets ted_hrlr_translate/pt_to_en for English to Portuguese translation

To download Tensorflow Datasets, please use:

pip install --user tensorflow-datasets==4.9.2
To use this dataset:

$ cat load_dataset.py
#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_datasets as tfds

pt2en_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
for pt, en in pt2en_train.take(1):
  print(pt.numpy().decode('utf-8'))
  print(en.numpy().decode('utf-8'))
$ ./load_dataset.py
e quando melhoramos a procura , tiramos a única vantagem da impressão , que é a serendipidade .
and when you improve searchability , you actually take away the one advantage of print , which is serendipity .
Transformers
To download transformers library, please use:

pip install --user transformers==4.44.2
