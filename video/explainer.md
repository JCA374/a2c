# How the car learned to race

## task

This is Car Racing, from the Gymnasium library. Every episode the game draws a new random circuit of about three hundred tiles. The car earns one thousand points spread over those tiles, paid the first time a wheel touches each one. Every frame costs a tenth of a point, and after a thousand frames time is up. So a fast, clean lap scores a little over nine hundred. An average of nine hundred is the classic level for calling the game solved.

## blind

The first version of this project, from December, used a method called A two C. It was meant to see the road through thirty two lidar rays. But every ray read zero, on every frame. The code tested physics bodies as if they were polygons, so no point was ever on the road. Its grip sensor always read full grip, and its finish bonus could never be paid. On one hundred unseen tracks it averaged minus forty three.

## eyes

The new car gets forty nine numbers each frame. They describe its speed, how much it slides and spins, and which wheels are on the grass. They also give where it sits on the road and its angle to it. Eight points on the centre line show where the road goes over the next hundred and forty metres. And nineteen rays, the blue lines here, measure the distance to the road edge. The yellow rings are the first centre line points.

## learn

It learns as an actor and a critic. The actor drives. The critic guesses how many points are still to come. Eight cars drive two hundred and fifty six frames each. Then every action is judged: did things go better or worse than the critic expected? The actor does more of what went better. The newer method, P P O, takes ten careful passes over each batch instead of one. And it is paid for exactly one thing, the game's own score.

## progress

Here is the same test track at three moments of one run, at double speed. Early on, the car is cautious and slides off in bends. In the middle it drives the track but loses time. The best model drives fast and keeps to the road. Below is the score on sixteen fixed validation tracks. It passed seven hundred within four hundred thousand frames, and nine hundred after about one million.

## result

The honest measurement uses one hundred tracks that were never used for training or for choosing a model. The best model so far averages close to nine hundred. Its median track scores over nine hundred and thirty, and it finishes most laps. Version one averaged minus forty three. The remaining gap is a handful of tracks where the car loses control in a sharp bend. That is what the next training runs work on.
