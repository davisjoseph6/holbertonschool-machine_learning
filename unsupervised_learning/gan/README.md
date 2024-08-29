GANs
 Novice
 By: Gaël Collinet , Malek Mrabti
 Weight: 3
 Migrated to checker v2: 
 Your score will be updated once you launch the project review.
Concepts
For this project, we expect you to look at these concepts:

Introduction to GANs (Generative Adversarial Networks)
Simple GANs
Wasserstein GANs
Wasserstein GANs with gradient penalty
Generating Faces



Resources

Read or watch:
GoodFellow et al. (2014)
Arjovsky et al. (2017)
Gulrajani et al. (2017)
Introduction to the Wasserstein distance
This person does not exist



Basics about GANs

Since their appearance around 2014, GANs have attracted a lot of attention due to their potential to generate fake models that can fool anybody.

For example, you can visit the pageThis person does not exist to see pictures of fake persons. Each time you refresh your browser, a new picture that seems quite realistic appears, but the person on this picture does not exist :


These persons do not exist



The generator is in fact a continuous function from a (relatively small) space of latent vectors to the space of all pictures.

So if a given latent vector 
 furnishes a realistic picture 
, then for a small latent vector 
, the new latent vector 
 will furnish a picture 
 that is very close from 
.

If 
 is wisely chosen, then 
 will be the same as 
, but smiling :



person0 = Generator (l0) on the left, person1 = Generator (l0+epsilon) on the right


With these kinds of perturbations, you can make your model older, or change its sex, or make its hair longer and so on :



Walk in a star shape to and from a picture


You can also walk along circle in the space of latent vectors, and look at the corresponding persons furnished by the generator, thus producing results like


Cyclic walk from and to a picture



Our aim in this project is to explain the basics of GANs, and convince you that you have understood how they work.

Beyond this project: A large family of image generators has grown since then:

Pro-GANs allow to have a far better quality of images
SR-GANs take a blurry picture as input and return the same picture, unblurred
Style-GANs have been popularized by their ability to modify one character of a picture. For example you can give them a picture of a young person and it will return the picture of the same person at age 60.

Tasks

This project about GANs is organized as follows :

The description contains a crash course about GANs. The fundamentals ideas behind GANs are exposed as fast as possible.
Task 0 : the class Simple_GAN : as they were introduced, GANs are a game played by two adversary players.
Task 1 : the class WGAN_clip : later it appeared that in fact the two players can collaborate.
Task 2 : the class WGANGP : a more natural version of the latter, which outperforms SimpleGANs and WGAN_clips in higher dimensions, as will be illustrated in the main program.
Task 3 : convolutional generators and discriminators : to work with pictures we need to use convolutional neural networks.
Task 4 : our own face generator : we will use a WGAN_GP model to produce faces of persons that don’t exist.
Appendix : at the end of task 4 there is a short digression explaining why GANs are superior to the older generators of fake pictures that were based on PCA.



Requirements

You should carefully read all the concept pages attached above to gain a fundamental understanding of GANs and their various types.
All your files will be interpreted/compiled on (still dont know) using python3 (version 3.9)
Your files will be executed with numpy (version 1.25.2) and tensorflow (version 2.15.0)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
A README.md file, at the root of the folder of the project, is mandatory
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
All your files must be executable



Objective: Provide a comprehensive guide to the project.
Content:
Introduction to GANs and the project.
Description of each task.
How to run the code, including dependencies.
Any relevant results or images.
Instructions on how to extend the project.
File Structure:
README.md

