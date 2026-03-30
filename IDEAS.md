# Ideas

General ideas that we can use as our improvements of our project.

## Hyperparameter tuning (R, ⍴, 𝛾)

There is the trivial method, which is just brute-force check, run it after each time, and then pick the hyperparameter tuple that maximizes accuracy. While this can be a worst case improvement, this is also doable by a high schooler with enough compute power.

We can perhaps use the trivial method as a way to find optimal to benchmark, but should not become our main contribution.

So A, there should be something more complex that we are able to do.

B, we need to ensure that we have enough compute to carry out said experiments. The whole point of coresets is to may be difficult.

### Formula Analysis

Perhaps analyze the dataset in a single pass before to capture metadata. Perhaps, we are able to create a simple trend lines between different metadata for different datasets, and perhaps we can create a simple function to predict the best hyperparameters. I do like this idea a lot.

This formula could be a simple function, such as linear, logarithmic, exponential, quadratic, constant, etc. I don't think there will be a strong correlation that would require a high degree polynomial or complex compound functions to model, so we force ourselves into simplicity.

### Hyperparameter Autotuning

Somehow we can create an algorithm that can adjust these three hyperparameters and do some form of work to get to optimal.

### Differences

It does seem that hyperparameters changed based on the selection ratio and complexity, but we may be able to extract more out of it.

## Showing results

If we successfully develop a hyperparameter-tuning strategy, we should utilize the same testing datasets that the author initially used with his default hyperparameters. It would obviously take more time to actually train a hyperparameter-tuner and converge, but we could show that it is hopefully pretty small, and it leads to a faster selection (unlikely) and higher accuracy (more likely).

We should also be able to generate some graphs to showcase the differences mentioned above.
