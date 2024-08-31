#!/usr/bin/env python3

from convolutional_GenDiscr import convolutional_GenDiscr

# Initialize the generator and discriminator models
gen, discr = convolutional_GenDiscr()

# Print the summary of both models
print(gen.summary(line_length=100))
print(discr.summary(line_length=100))

