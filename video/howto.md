# How to use Drive A two C

## intro

This is a short guide to the Drive A two C project. In about three minutes you will see how to set it up, train your own driver, watch it race, and measure its score the honest way.

The car learns to drive the Car Racing game from Gymnasium. It never sees pixels. Every frame it gets forty nine numbers about its speed, its wheels and the road ahead, and it answers with two: how much to steer, and how hard to press the pedal.

## install

Setting up takes three commands. Clone the repository from GitHub. Make a Python virtual environment inside it. Then install the requirements file into that environment.

That brings in Gymnasium with the Box two D physics engine, PyTorch for the networks, and numba for the fast ray casting. No graphics card is needed. Everything here was trained on an eight thread laptop.

## train

To train, run the training script and give your run a name. It starts eight cars on eight random tracks, one process each, and they drive about fifteen hundred frames per second in total.

Every hundred thousand frames it stops and drives sixteen fixed validation tracks without any randomness. That score decides which checkpoint is the best one.

Everything lands in a folder under runs. There is a table of statistics, a log of events, the curriculum, and the checkpoints. If you stop it with control C, it saves first, and the resume option carries on from where it was.

## watch

To watch a model drive, run the play script with the path of a checkpoint. A window opens and the car drives three test tracks. At the end of each track it prints the score, how much of the track it covered, and whether it finished the lap.

The site has the same thing without installing anything. On the Play tab you pick a model and one of ten fixed tracks, and the clip plays once, ending on its result.

## evaluate

To measure a model, run the evaluate script. It drives one hundred tracks the model has never seen, under the game's own rules. That means a thousand frames, plus one thousand divided by the number of tiles for each new tile, and minus a tenth of a point per frame.

It runs in the reference version of the game, with the pixels drawn and the slow ray casting, never the fast training copy. It writes a results file, and the site's goal table reads it from there.

## tests

Finally, run pytest. The tests check three things. The fast ray caster gives exactly the same answers as the plain Python one. The training game is bit for bit the same world as the reference. And the learning maths is right.

That is the whole workflow: install, train, watch, evaluate, test. The site's Log tab follows the current run every fifteen minutes.
