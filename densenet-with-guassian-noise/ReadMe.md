For training:
Run "python -u main.py --noise 1 --modeldir 'saved_models/test"(some arbitrary directory)
if you dont want visualization using WANDB, run:
WANDB_MODE=offline python -u main.py --noise 1 --modeldir 'saved_models/test

For Testing, if you want to restore my results, first comment the training part(Line 41 - 46) in the code(exactly like HW2) and then run 
"python -u main.py --noise 1 --modeldir 'saved_models/densenet_noisy"
