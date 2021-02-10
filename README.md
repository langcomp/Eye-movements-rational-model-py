# Eye-movements-rational-model-py

A rational model of eye movements for identifying a single word.

---
### I. Prerequisites
In order to run the scripts provided for demo, you need to ensure that Python 3 and the following modules are installed:  

- python3.6+
- numpy
- pandas
- scipy

---
### II. Structure
<pre>
├── Eye-movements-rational-model-py
    ├── README.md
    ├── requirements.txt
    ├── EMRM.py
    ├── demo.py
    ├── demoRefixation.py
    ├── demoSkipping.py
    └── [DIR] data
        ├── example_vocab.csv
        ├── example_refix_vocab.csv
        ├── example_skip_vocab.csv
        └── example_skip_human_fix.csv
</pre>

---
### III. Getting started
See `demo.py` for examples.

1. Input data: you always need a file that stores vocabulary information. This file should include two columns: `word` and `logfreq`. If your simulation involves comparison with existing fixation data, please check `demoSkipping.py` as an example.
2. Build your own `OneVirtualReader` that has its own `Vocabulary`, and can perform `OneTrial` and/or `OneBlock` of word identification task.