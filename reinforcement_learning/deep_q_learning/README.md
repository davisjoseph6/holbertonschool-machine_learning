Deep Q-learning
 Master
 By: Alexa Orrico, Software Engineer at Holberton School
 Weight: 6
 Manual QA review must be done (request it when you are done with the project)
Description


Resources
Read or watch:

Deep Q-Learning - Combining Neural Networks and Reinforcement Learning
Replay Memory Explained - Experience for Deep Q-Network Training
Training a Deep Q-Network - Reinforcement Learning
Training a Deep Q-Network with Fixed Q-targets - Reinforcement Learning
References:

Setting up anaconda for keras-rl
keras-rl
rl.policy
rl.memory
rl.agents.dqn
Playing Atari with Deep Reinforcement Learning
Learning Objectives
What is Deep Q-learning?
What is the policy network?
What is replay memory?
What is the target network?
Why must we utilize two separate networks during training?
What is keras-rl? How do you use it?
Requirements
General
Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.9)
Your files will be executed with numpy (version 1.25.2), gymnasium (version 0.29.1), keras (version 2.15.0), and keras-rl2 (version 1.0.4)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
A README.md file, at the root of the folder of the project, is mandatory
Your code should use the pycodestyle style (version 2.11.1)
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
All your files must be executable
Your code should use the minimum number of operations
Installing Keras-RL
pip install --user keras-rl2==1.0.4
Dependencies
pip install --user gymnasium[atari]==0.29.1
pip install --user tensorflow==2.15.0
pip install --user keras==2.15.0
pip install --user numpy==1.25.2
pip install --user Pillow==10.3.0
pip install --user h5py==3.11.0
pip install autorom[accept-rom-license]
