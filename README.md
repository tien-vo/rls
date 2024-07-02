# Relativistic Lorentz Simulation

This package provides utilities for performing test-particle simulations in plasma physics. We make it public it together with an ApJL submission on whistlers collocated at a magnetic gradient, and intend to show its usage with future projects.

The code is organized in such a way that an experienced Python user might know how to navigate. Documentation is very minimal at the moment, but will be improved in coming months. If you have any questions, please direct your inquiries to the project maintainer.

We also intend to integrate the codebase here with larger projects such as PlasmaPy or AstroPy. Please follow for future updates.

## Installation

### Dependencies

`micromamba` is the preferred package manager. The main Python and non-Pythonic dependencies are managed with `micromamba`, while Pythonic dependencies are managed with `python-poetry`. To install the package, run

```
make install
```

## Usage

Please follow `projects/whistler_in_magnetic_gradients` to see how the simulation data and paper figures are generated.
