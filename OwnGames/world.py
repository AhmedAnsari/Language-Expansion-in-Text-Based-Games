from room import Room
from random import randint

class World:


	def __init__(self):

		self.currRoom = 0

		self.roomNames = ['living', 'garden', 'kitchen', 'bedroom']
		self.rooms = []
		desc = ["This room has a couch, chairs and TV.",
				"You have entered the living room. You can watch TV here",
				"This room has two sofas, chairs and a chandelier."]
		nei = [-1, 1, -1, 3]
		obj = "tv"
		act = "watch"
		self.rooms.append(Room(0, self.roomNames[0], desc, nei, obj, act))

		desc = ["This space has a swing, flowers and trees.",
				"You have arrived at the garden. You can exercise here",
				"This area has plants, grass and rabbits."]
		nei = [-1, -1, 0, 2]
		obj = "bike"
		act = "exercise"
		self.rooms.append(Room(1, self.roomNames[1], desc, nei, obj, act))

		desc = ["This room has a fridge, oven, and a sink.",
				"You have arrived in the kitchen. You can find food and drinks here.",
				"This living area has pizza, coke, and icecream."]
		nei = [1, -1, 3, -1]
		obj = "apple"
		act = "consume"
		self.rooms.append(Room(2, self.roomNames[2], desc, nei, obj, act))

		desc = ["This area has a bed, desk and a dresser.",
				"You have arrived in the bedroom. You can rest here.",
				"You see a wooden cot and a mattress on top of it."]
		nei = [0, 2, -1, -1]
		obj = "bed"
		act = "sleep"
		self.rooms.append(Room(3, self.roomNames[3], desc, nei, obj, act))

	def move(self, act, obj):
		if act == 'go':
			ind = 0
			if obj == 'north':
				ind = 0
			elif obj == 'east':
				ind = 1
			elif obj == 'west':
				ind = 2
			elif obj == 'south':
				ind = 3
			else:
				return 'not available'

			nextRoom = self.rooms[self.currRoom].neighbors[ind]
			if nextRoom == -1:
				return "not available"
			else:
				self.currRoom = nextRoom
			return self.desc()

		elif act == self.rooms[self.currRoom].action:
			if obj == self.rooms[self.currRoom].object:
				return act + '\nREWARD_' + self.rooms[self.currRoom].action + ' 1.'
			else:
				return 'not available'
		else:
			return 'not available'

	def desc(self):
		ind = randint(0,2)
		return self.rooms[self.currRoom].desc[ind]

# world = World()

# while True:
# 	act = raw_input("Action: ")
# 	obj = raw_input("Object: ")
# 	print(world.move(act, obj))
# 	print(world.desc())
