
# Humanoid Walking and Stair Traversal

## Setting up the Code



## Training

mlagents-learn config/bipedalagent_config.yaml --run-id=TestRun --time-scale 10 --force


## Visualising
### Within Unity

### Tensor Board

tensorboard --logdir results

### PCA Visualisation using Blender
## Changes made from example (CODE THAT WAS ACTUALLY WRITTEN)
## Results from Testing

20/01/2023 - BipedalStabilityTest1.

* Physics settings changed for greater physics simulation accuracy. Reason: for this research to be applicable to real world robots a realistic simulation is needed to allow the training of a model with a better chance of working in the real world.
* Scaled down robot to realistic size to make gravity behave correctly. Reason: again to improve realism

20/01/2023 - BipedalStabilityTest2. Result: Leaping to target, 4+ hours to train, 3 hidden layers

* Joint Drive Controller > max joint force limit set from 20,000 to 10,000. Reason: so that the join controller is not allowed to exert unrealistically high forces
* Target cube moved to global origin as this is the center of spawning - Irrelevant
* Target speed "scaled down" due to earlier scale change. Target walking speed changed from 10 m/s to 1.3 m/s. Reason: the target speed needs to suit the scale and ability of the simulated robot agent
	
20/01/2023 - BipedalStabilityTest3. Result: Trains more than twice as quickly.

* Forward and up directions changed (swaped) to match agent model. Hopefully to make the robot walk forwards.

21/01/2023 - BipedalStabilityTest4. Result: Trains more quickly. Higher overall reward reached before settling. However The technique used is tiny stepping with a wide stance.

* Changed the clock input to be relative to the physics simulation using Time.fixedTime()
* Added the velocity and ground contact penalties with Von Mises distributed phase durations, 0.7s each gate with equally weighted A & B Phases
* !Changed the calculation of the weights from multiplicative to addative and weighted! - Next test is not directly comparable
	
02/02/2023 - BipedalStabilityTest5. Result: Trains more quickly but the still shuffling its feet

* Changed the walk cycle time from 0.7s to 1.0s
* Changed the clock input to 2 inputs (sin and cos wave)
* Changed the fps from 60 to 30
* Changed the multiplier for the foot force from -0.01 to -10 as the "force" is actually the bool of ground contact (0 or 1)

05/02/2023 - BipedalStabilityTest6. Result: Still not behaving like human walking

* Added in a small randmised step between the agent and the target (What is randomised and why?)
* Seperated the 10 environments further apart by 10m (to let out of bound targets fall and reset instead of becoming unreachable)
* Increased the learning_rate from 0.0003 to 0.0006

06/02/2023 - BipedalStabilityTest7. Result: Still not behaving like a human walking

* Removed the tilt reward
* Added reward for smoother action changes
* Added reward for using less torque
* Note: The weights of these rewards may need to be higher

09/02/2023 - BipedalStabilityTest8. Result: Still not behaving like a human walking

* Added more realistic normal force calculation and changed scaling appropriate in reqard function to accomodate this
* Using randomise walking speed now
* Set walking gate time back to 0.7s

11/02/2023 - BipedalStabilityTest9. Result: Still not behaving like a human walking

* Added LSTM component with memory_size: 32 and sequence_length: 64

11/02/2023 - BipedalLSTMTest10. Result: Slow training (NOT COMPLETED)

* Reverted back to using non LSTM RNN model. Reason: Previous training was very slow and didn't show signs of improvement.
* Added inverting of agent state and actions for half a phase.
		
11/02/2023 - BipedalMirrorTest11. Result: Still non-asymetric gait 

* Added symetric reward with 0.5 weighting. Reason: This should insentivise symetric gates.

12/02/2023 - BipedalMirrorTest12. Result: Still non-asymetric gait 

* Added realistic COG location. Reason: As this (may be / is) important to walking "like" a human
* Changed total mass from 10 to 44 kg with 38 kg for the torso. Reason: As this (may be / is) important to walking "like" a human
* Reduced ground contact penalty from -1 to -0.2. Reason: so that the model cares less about falling over and more about achieving the correct motion intially.

18/02/2023 - BipedalMirrorRealisticCOGTest13. Result: Still non-asymetric gait 
	
* Reduced observations from 110 to 79 by removing joint position
* Fixed incorrect location of joint origins in the agent prefab to prevent dislocation of joints
* Changed the hip weight from 1 to 0.5
* Reduced weighting of symetry reaward from 0.5 to 0.2

18/02/2023 - BipedalMirrorRealisticCOGFixSimplifiedObsTest14. Result: Still non-asymetric gait. Slow training

* Gave the robot feet
* Went back to having 111 observations
* Ground contact penalty back to -1

20/02/2023 - BipedalMirrorFeetTest15. Result: Still non-asymetric gait.

* Duplicated the clock sensor inputs and added getPhase as a clock input as well as inverse of conditionalReverse
* Reduced the exponent of the force reqard terms from 0.01 to 0.002 to encourage reducing force for a larger range of forces initially

24/02/2023 - BipedalMirrorFeetTest16. Result: faster training, little improvement to walking like a human.
	
* Removed the ground contact reward and replaced it with a upper leg anlge reward based on phase time
	
26/02/2023 - BipedalMirrorFeetTest17. Result: Human like walking but walking backwards and quite slow without swinging its legs very much.
	
* Changed lookAtTargetReward calculation so that it walks forwards instead of backwards (made it exponential which should also improve this reward when getting close to the target value).
* Changed the upper leg angle reward to use sigmoid function as this is symetrical. Still a function of EI. Increased its midPoint from 320 to 330 degrees. Sensitivity changed to 0.2 to match previous sensitivity a bit better.
* Changed target walking speed from 1.3 to 1.8
* Removed reward for touching the target
* Removed the step that was between the agent and the target

27/02/2023 - BipedalMirrorFeetTest18. Result: Walks much more human like although legs are far apart.

* Replaced the target with a velocity vector (Target rotation & Target speed)
* Removed target cube position sensor input
* Removed conditionalReverse(float)
* Reduced target velocity from 1.8 to 1.4
* Removed direction indicator (was just used for asthetics)

28/02/2023 - BipedalFeetTest19. Result: Same walking as before but trained more quickly (probably due to removing conditional reverse)
	
* Removed all conditionalReverses
* Reduced matching speed reward (factor of 0.5) and increased energy efficiency (torque) reward by 2.5
* Added a physics material to all the objects
* Added ground domain randomisation
* Changed foot slerp drive to have springs, dampers and max force 10 times less than before (x0.1 compared with the rest of the joints)

29/02/2023 - BipedalImprovingRealism1. Result: Much less human like walking
	
* Add back conditional reverses.
* Removed joint power output.

04/03/2023 - BipedalImprovingRealism2. Result: Same human walking as BipedalFeetTest19.

* Got rid of walls and made training area 2 times larger
* Not duplicating clock input sensors
* Changed joint strength param from 1 to 2 to avoid outputs from the ANN reaching their limits

06/03/2023 - BipedalImprovingRealism3. Result: Human like walking. Marginally gaining higher rewards

* Changed slerp drive to position spring (10000 to 100000), position damper (100 to 1000) and maximum Force (25000-1000)

08/03/2023 - BipedalImprovingRealism4. Result: Human like walking. Marginally gaining higher rewards

19/04/2023 - FYP_Feet.

19/04/2023 - FYP_NoFeet. [Removed feet for this run]

19/04/2023 - FYP_2x128
20/04/2023 - FYP_3x128

FYP_NoFeet\BipedalAgent
610.1

FYP_3x64\BipedalAgent
241.6

FYP_3x512\BipedalAgent
615.1

FYP_3x128\BipedalAgent
632

FYP_2x512\BipedalAgent
588.7

FYP_2x256\BipedalAgent
547.9

FYP_2x128\BipedalAgent
201.6



	Reduced maximum joint force from 1000 to 100 and maximum joint strength setting in joint driver from 2 to 1.
	Increased lower bound on domain randomisation of friction from 0.1 to 0.6.

23/04/2023 - FYP_3x128 Results: faster training than 3x128 & higher overall reward

	DONE: Do not randomise start orientation!
	TODO: Double check movement direction and orientation rewards are in the correct directions!!!! ---- Not Needed
	DONE: Realistic Rigid Body Masses (Hip mass was 0.5) m1 was 38, m2 was 2, m3 was 0.5
	

	Rewards for foot force were added back in place of upper leg forward / backward reward
	Reward for velocity changed to z and x directions
	Reward for minimising body angular velocity and acceleration added
	Some reward weights changed, force weight from 0.002 to 0.005, at weight from -1 to -2
	Added random start orientation back in to hopefully improve symmetry of walking gait


------------- Folder Format -------------

Run (Tuning the Reward Function) Due: 27/02/2023
	Experimenting to get human like walking
	[X] Remove step in the way of goal
	[X] Replace the target with a velocity vector (Target rotation & Target speed)
	[X] Remove target cube position sensor input
	[X] Remove conditionalReverse() ???
	
---- Human Like Walking Agent Done ----

Realism Due: 06/03/2023
	[X] Improving the realism of the physics engine
	[X] Improving the realism of the environment by adding friction forces
	[X] Adding domain randomisation to the ground

	1.1 [X] Create stairs and steps generation code

	2.1 [-] Create a script to evaluate reliability of the model
		[X] External perturbation code created

	2.2 [ ] Write code which outputs foot movement data that can be plotted

	3.1 [#] Adding a PD joint controller
	3.2 [ ] Adding domain randomisation to the joint actuators

	4.1 Removing Sensors

	If this works:
	[ ] Tidy up code

---- Realistic Human Like Walking Agent Done ----

Experimenting with the setup:
	[ ] TEST: With vs without feet
	[ ] TEST: Changing the number of parrallel agents with: 1, 2, 4, 8, 16, (32)
	[ ] TEST: Changing the number of time steps the agent can take (episode length increased) from 500 to 5000

AblationStudy Due: 13/03/2023
	[ ] TEST: Removing sensors so that only the minimal and realistic sensors are required whilst still being able to walk like a human

---- Sim-To-Real Applicable Agent ----

[X] ADD STEPS AND STAIRS

ANNshape Due: 22/04/2023
	Use 9 Different ANN shapes and investigate the differences in the learned behaviour


Idea: Allow it to vary the stepping time as an output so that it can hone in on the correct gait step rate
Idea: foot off ground height specific reward

Idea: do not peanalise contact with the ground when it falls [x]
Idea: peanalise multiple steps (ground contacts) per cycle


Auto Startup:
"C:\Program Files\Unity\Hub\Editor\2021.3.9f1\Editor\Unity.exe" -projectPath "C:\Users\Thomas Groom\Desktop\Blind-Bipedal-Locomotion\Bipedal Stair Traversal" -batchmode


DEPRECATED --------

1st Phase - Get it to take steps like a human
TODO:
	[/] - Replicate simulation results from report by Oregon State University
	[/] - Improve physics simulation realism - what does this mean?
		[ ] - Realistic max joint force parameters etc...
		[X] - Realisic COG and mass similar to a human
	[-] - Get LSTM to work?
	[X] - Remove target object as the goal and instead use velocities of the robot and perhaps rotation of the robot?

2nd Phase - Abstract away sensors and actuator control to make it realistic to a robot
TODO:
	[ ] - Implement PD controller to go between the ANN and the actuators
	[ ] - Reduce inputs to be realistic to a real life robot

3rd Phase - make it traverse steps
TODO:
	[/] - Step generation algorithm
		[X] - Single step between origin and target with some randomisation
		[ ] - More advanced random single steps (calling a seperate script)
		[ ] - Multiple steps together to form stairs
		[ ] - Research suitable randomisation parameters


###Joint Configuration Due: 27/03/2023
###	Use 5 different joint configurations to determin the most important joints to walking reliably






