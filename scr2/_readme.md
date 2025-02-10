# Graph Predictive Coding scr2

...
Combination of VanZwol and own code;


## Training loop
0. Make choices:
   - Graph type: Fully connection, Hierachy, SBM
   - Restricted / non-restricted: 
      - Sens2sens / Sens2Sup
      - which connection to update
   - optim / grokfast 



1. Create graph topology by creating a 2d mask (Fully_connected, generative, discriminative, SBM, other (see `scr2/graphbuilder.py`) )
2. Initialize the weights of the graph using the mask: 
   - either 2d matrix or 
   - 1d for sparse weights (MessagePassing)
3. Create dataloader with batched graphs 
4. Init the model, params, grads, optimizer
5. Training using clamped img (X) and img_label (Y)
   Test on val_set by removing either img for generation eval or remove img_label for classification task, depedining on the task and topology (FC can do both, hierachical (VanZwol only either one))
6. Eval on eval_set

