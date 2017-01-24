from world import World
import numpy as np
import zmq
import sys

		

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect ("tcp://localhost:5678" + str(1))
world = World()
socket.send("Gotcha!!!\nLet's see.")
# inp = socket.recv()
# print inp
while True:
	inp = socket.recv()
	inpSplit = inp.split(' ')
	print inpSplit[0]
	ans = ""
	if inpSplit[0] == 'look':
		ans = world.desc()
		print(ans)
		socket.send(ans)
	elif inpSplit[0] == 'tel':
		world.currRoom = int(inpSplit[1])
		ans = "Teleported to " + inpSplit[1]
		socket.send(ans)
	else:
		world.move(inpSplit[0], inpSplit[1])
		socket.send("Command executed.")
