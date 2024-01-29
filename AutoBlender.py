import signal
from subprocess import Popen
import os
import time

if(os.path.isfile("PCA1.png")):
    os.remove("PCA1.png")
#os.startfile('C:/Users/Thomas Groom/Desktop/Bipedal-Locomotion/test.py')
#p = subprocess.run(['blender','GeneratePCAImages.blend', '-P', 'test.py'])
pro = Popen(['blender', 'GeneratePCAImages.blend', '-P', 'test.py', '-p', '2000', '0', '0', '0'])

while(not os.path.isfile("PCA1.png")):
    pass

time.sleep(2)
os.kill(pro.pid, signal.SIGTERM)  # Send the signal to all the process groups
print("DONE")