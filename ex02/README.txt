20017069
20038024
*****

Question 5:

When writing the evaluation function for betterEvaluationFunction, our idea was to use (as recommended)
a linear combination of features. We then considered which feature should be taken into account,
since here we evaluate states and not actions (thus as a consequence when looking at some world state,
for example the food map, it is its instance *after* the action was performed (i.e. here we can't 
check if pac-mac is standing "over" a food since it doesn't exist in this point in time)).
Following this logic, we took the following features into account:

- The distance to the scared ghosts. (gradeNumGhosts)
	-> Inversely related: As the sum of all distances is greater, its addition to the grade is smaller.

- The total "amount" of times of the scared-times of the ghosts. (gradeScared)
	-> Directly related. If the pacman just ate a capsule, this value will be quite high.
	   This means that pacman will most surely eat a capsule if it is next to it.

- Amount of capsules left. (gradeCapsule)
	-> Inversely related: more capsules equals less score (drives pac-man to eat capsules when next to them).
	   We gave this a relatively high score so unless pac-man will surely get eaten by a ghost,
	   when passing next to a capsule it'll eat it.

- Amount of food left. (gradeFoodNum)
	-> Inversely related: in order to drive pac-man to eat when he can.

- Average distance to all foods. (gradeFoodDist)
	-> Inversely related: in order to drive pac-man to move closer to food, 
	   and in particular to areas with many food pellets.
	

- Distance to the nearest food. (gradeNearFoodDist)
	-> Inversely related. Aims to get closer to food pellets.

- currentGameState.getScore(). (s)
	-> Directly related: We used it in order to give weight to 2 things that we didn't give weight anywhere else - 
	   time elapsed and avoiding ghosts. This value did a very good job at helping pacman
	   avoid the ghosts and not die.

After calculating all the above scores (and many other which we ended up not including),
we found the proper coefficients by trial-and-error, with some intuition of what's more important.
We aimed to normalize all the variables (e.g. sum of distances to all scared ghosts will
usually be an order of magnitudes bigger then distance to a single food). Eventually we handled the case of
a last remaining food piece (the else clause of ‘if food’), in which we added a very high artifact to 
the score (10000) so when passing next to it it’ll eat it no matter what.
