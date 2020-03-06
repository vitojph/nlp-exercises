# NLP Exercises

## Set up environment

Create a new virtual environment and install packages.

    conda create -n nlp-exercises pandas tqdm
    conda activate nlp-exercises

If using cuda:
    
    conda install pytorch cudatoolkit=10.1 -c pytorch

else:
    
    conda install pytorch cpuonly -c pytorch

Install simpletransformers.

    pip install simpletransformers


