# Ideas

General ideas that we can use as our improvements of our project.

## Hyperparameter tuning (R, ⍴, 𝛾)

There is the trivial method, which is just brute-force check, run it after each time, and then pick the hyperparameter tuple that maximizes accuracy. While this can be a worst case improvement, this is also doable by a high schooler with enough compute power.

We can perhaps use the trivial method as a way to find near-optimal to benchmark to a certain specific level, but should not become our main contribution, because it is literally just a bunch of nested for loops and function calls.

So A, there should be something more complex that we are able to do.

B, we need to ensure that we have enough compute to carry out said experiments. The whole point of coresets is that there is a lot of data that we need to decrease, so doing so may be difficult.

### Global Formula Analysis

Perhaps analyze the dataset in a single pass before to capture metadata. This metadata could include number of classes or outputs, among others that we can consider later. However, these metadata should be cheap. We should probably focus some time into picking metadata. This could be after the R epochs initially, though that is already something we want to test.

We would then run different hyperparameters for these datasets. Perhaps, we are able to create a simple trend lines between different metadata for different datasets (for what the optimal hyperparameters are, or accuracy per hyperparameter selection), and perhaps we can create a simple function to predict the best hyperparameters.

This formula could be a simple function, such as linear, logarithmic, exponential, quadratic, constant, etc. I don't think there will be a strong correlation that would require a high degree polynomial or complex compound functions to model, so we force ourselves into simplicity.

We would probably run this on multiple datasets. However, we will be heavily limited by compute for this case. We should find some cheap datasets.

Problem is that this is only really going to be applicable to the datasets that we test it on, and I'm worried there is no real way of showing that this is going to work for any datasets that do not resemble ours.

### Hyperparameter Autotuning

Somehow we can create an algorithm that can adjust these three hyperparameters and do some form of work to get to optimal without trial and error.

This could be adaptive based on R. We can stop R early if we detect some pattern happening. This could just be rank stability.

Essentially, we are diagnosing factors we realize are important during the warm-up epochs.

For ⍴, after selecting the initial high-MRMC subset and training the proxy model, we can look at how the proxy loss distributes over the remaining samples and see if we need to adjust.

For 𝛾, we can compare the scale and spread of the MRMC scores versus the regularization scores and normalize accordingly. If the two scores are on very different scales, 𝛾 needs to compensate, so we set it based on their relative variance.

### Differences

It does seem that hyperparameters changed based on the selection ratio and complexity, but we may be able to extract more out of it.

## Showing results

If we successfully develop a hyperparameter-tuning strategy, we should utilize the same testing datasets that the author initially used with his default hyperparameters.

It would obviously take more time to actually train a hyperparameter-tuner and converge, but we could show that it is hopefully pretty small, and it leads to a faster selection (unlikely) and higher accuracy (more likely).

Otherwise, we can utilize grid search to find a empirical upper bound, then show that we get closer to that than the random hyperparameters with negligible overhead.

We should also be able to generate some graphs to showcase the differences mentioned above.
