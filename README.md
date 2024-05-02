# torch_neuronx_exploration
Code to explore the implementation of custom architectures in AWS Neuron using Pytorch

## Tests Logs

What have been experimented so far

- analyzing https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/analyze_for_training.html
    - The analyze command checks the support of operations within the training script by checking each operator against neuronx-cc. It is only supported for PyTorch models. The output of the tool will be available as @sushmitha ult.json within the output location.  
`neuron_parallel_compile --command analyze --analyze-verbosity 1 python3 train_singlew.py` Output: result.json All the operations are supported.
- Collecting `neuron_parallel_compile --command collect python3 train_singlew.py` It runs successfully.
- Cleaning the cache `neuron_parallel_compile --command clean , neuron_parallel_compile --command clear-locks`
- Compiling `neuron_parallel_compile --command compile python3 train_singlew.py` 
- Run the actual script (with the normal architecture) using the conda environment that we created: python3 train_singlew.py It fails.
- Run the actual script (with the sequential one) using the conda environment that we created: python3 train_singlew.py. It fails.
- Run the actual script using the docker containers: https://github.com/aws-neuron/deep-learning-containers
    - aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com
    - docker build --build-arg="NEURON_VER=2.18.2” --build-arg="PYTORCH_VER=2.1.2" -f Dockerfile -t neuron-test .
    - docker build --build-arg="NEURON_VER=2.18.2" --build-arg="PYTORCH_VER=1.13.1" -f Dockerfile -t neuron-test .
    - docker run -t -v ${PWD}:/opt/traiunium/ --device=neuron ...
    - Results:
        - It fails with the pytorch 2.1.2
        - It works with pytorch 1.13.1. It fails If stopped suddenly. Once I clear the cache using clean and clear-locks, it works and saves the model. One of the cores is being used. Compilation last for some seconds, then it starts training and finishes within seconds.