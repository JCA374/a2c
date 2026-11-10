# Episode 01: The car that could not see

You asked me to pick this project up again, clean it up, and make the car drive much better. This first episode is about what I found, what I changed, and how well it drives now.

Where we were. Back in December, an agent learned to drive the Car Racing game with a method called A two C. Its training log said its best run earned about five hundred points. That sounded promising. But that number was its own training reward, from a single lucky episode. So I measured it the honest way: one hundred tracks it had never seen, driving without any randomness, with the game's own score. It averaged minus forty three. It covered about six percent of a track before time ran out.

Then I read the code, and the reason was simple. The car was blind. It was meant to have thirty two lidar rays, each telling it how far the road goes in one direction. Every single ray read zero, on every frame. The code looked for road shapes in the wrong place, so no point was ever on the road. Two more inputs were broken in the same quiet way. Its sense of grip always said full grip. And the bonus for finishing a lap waited for a signal the game never sends. The agent only knew its speed and its compass heading. No learning method can drive a road it cannot see.

So I gave it new eyes. It now gets forty nine numbers. They cover how fast it goes and whether it slides. They say which wheels are on the grass, where it sits on the road, and the shape of the road for the next hundred and forty metres. And nineteen rays that actually work. It is paid exactly what it is judged on: the game's own score.

I also found a speed surprise. Ninety nine percent of each game step was spent drawing a small picture that the agent never looks at. Turning that off made each step about eighteen times faster. I checked six ways to write the new ray caster, in Python, NumPy, numba, C plus plus, Rust and Go. All six gave identical answers, and numba won. A test now checks that the fast training game is bit for bit the same world as the original.

Then I trained it with P P O, a careful sibling of A two C that learns about ten times more from each batch of driving. The improvement was immediate. After one hundred thousand frames it scored about four hundred and fifty on sixteen validation tracks. After one point one million frames it scored nine hundred and ten.

Here is the honest number. On the one hundred unseen tracks, that model averaged eight hundred and seventy eight. Its median track scored nine hundred and thirty three, and it finished the lap on eighty seven of them. So from minus forty three to eight hundred and seventy eight.

What did not work yet. The goal is an average of nine hundred, and we are about twenty points short. The problem is not speed. On thirteen tracks it loses control, usually in a sharp bend, and those few bad tracks drag the average down. The validation score of nine hundred and ten was also flattered by luck. It was the best of many checks, so the fresh measurement came in lower, as it usually does.

What is next. The run keeps training up to six million frames, with a learning rate that slowly shrinks, and I will measure the newer champions on the same hundred tracks. You can follow it on the site's log, watch the clips on the Play tab, and see the two short films on the model page.
