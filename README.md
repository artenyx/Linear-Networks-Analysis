# Linear Networks Analysis

Code to run layerwise network training as an initilization method for deep linear networks. Designed for Adam or SGD optimizers.

To run code:
```markdown
python main.py --add_layers 10 --epochs_per_layer 40 --epochs_classif 50
```

Code currently runs on MNIST as linear networks do not perform well on higher dimensional data.

To Do:
- Test classification function
