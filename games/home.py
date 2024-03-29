import os
import time
import string
import numpy as np
from .game import Game
import re

def clean_words(words):
  return [word.lower().translate(None, string.punctuation) for word in words]

class HomeGame(Game):
  def __init__(self, num_rooms=4, default_reward=-0.01,
               junk_cmd_reward=-0.1, quest_levels=1,
               seq_length=35, max_step=100, debug=False,
               username="root", password="root",
               game_dir="../text-world"):
    super(HomeGame, self).__init__(num_rooms, default_reward,
                                   junk_cmd_reward, quest_levels, seq_length,
                                   max_step, debug, username, password, game_dir)

    self.name = "home"
    self.rooms = ["Living", "Garden", "Kitchen", "Bedroom"]

    self.actions = ["eat", "sleep", "watch", "exercise", "go"]
    self.objects = ["north", "south", "east", "west"]
    self.goals = ["kitchen", "bedroom", "living", "garden"]
    self.quests = ["You are hungry", "You are sleepy", \
                   "You are bored", "You are getting fat"]
    self.quests_mislead = ["You are not hungry", "You are not sleepy", \
                           "You are not bored", "You are not getting fat"]

    self.idx2word = ["not", "but", "now"]

    self.make_vocab(os.path.join(self.game_dir, "evennia/contrib/text_sims/build.ev"))
    self.new_game()

  def make_vocab(self, fname):
    # try:
    #   f = open(fname)
    #   check = f.read()
    #   f.close()
    #   print "I am in " * 100
    # except:
    #   print "*" * 100

    with open(fname) as f:
      for line in f:
        words = line.split()
        if words:
          if words[0] == '@detail' or words[0] == '@desc':
            self.idx2word.extend(words[3:])
          elif words[0] == '@create/drop':
            self.objects.append(words[1].split(":")[0])

    for quest in self.quests:
      quest = quest.translate(None, string.punctuation)
      self.idx2word.extend(quest.split())

    self.idx2word = list(set(clean_words(self.idx2word)))
    self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}

  def new_game(self):
    self.quest_checklist = []
    self.misliad_quest_checklist = []
    self.goal_room = []
    self.step = 0
    self.random_teleport()
    self.random_quest()

    return self.get_state(timeout=True)

  def random_teleport(self):
    room_idx = np.random.randint(self.num_rooms)

    self.client.send('@tel tut#0%s' % room_idx)
    time.sleep(0.1)
    self.client.get()
    self.client.send('l')
    self.client.get()
    if self.debug:
      print(" [*] Start Room : %s %s" % (room_idx, self.rooms[room_idx]))

  def random_quest(self):
    idxs = np.random.permutation(len(self.quests))
    for idx in xrange(len(self.quests)):
      self.quest_checklist.append(idxs[idx])
      self.goal_room.append(self.goals[idxs[idx]])

    self.mislead_quest_checklist = [idxs[-1]]
    for idx in xrange(len(self.quest_checklist) - 1):
      self.mislead_quest_checklist.append(idxs[idx])

    if self.debug:
      print(" [*] Start Quest : %s %s." % (self.get_quest_text(self.quest_checklist[0]), \
                                          self.actions[self.quest_checklist[0]]))

  def get_state(self, action=None, object_=None, timeout=False):
    is_finished = self.step > self.max_step
    result = self.client.get(timeout=timeout)

    self.client.send('look')
    room_description = self.client.get()

    texts, reward = self.parse_game_output(result, room_description)
    
    if self.debug:
      # log = " [@] get_state(\n\tdescription\t= %s \n\tquest\t\t= %s " % (texts[0], texts[1])
      # if action != None and object_ != None:
        # log += "\n\taction\t\t= %s %s " % (action, object_)
        # log += "\n\tresult\t\t= %s)" % (result)
      # log += "\n\treward\t\t= %s)" % (reward)
      print(texts, reward)
      time.sleep(0.1)
      if reward > 0:
        print(texts, reward)
        time.sleep(2)
    # print reward 
    # remove completed quest and refresh new quest
    if reward >= 1:
      self.quest_checklist = self.quest_checklist[1:]
      self.mislead_quest_checklist = self.mislead_quest_checklist[1:]
      self.goal_room = self.goal_room[1:]

      if len(self.quest_checklist) == 0:
        is_finished = True
      else:
        texts.append(self.get_quest_text(self.quest_checklist[0]))

    vector = self.vectorize(texts)
    # percentage = 0.0
    # if is_finished:
    percentage = (1.0 - (len(self.quest_checklist)/(len(self.quests) * 1.0)))
    return vector, reward, is_finished, percentage

  def vectorize(self, texts, reverse=True):
    null_idx = (len(self.word2idx) + 1)
    vector = np.ones(self.seq_length) * null_idx

    cnt = 0
    for text in texts:
      for word in clean_words(text.split()):
        try:
          vector[cnt] = self.word2idx[word]
          cnt += 1
        except:
          pass
          # print(" [!] %s not in vocab" % word)

    if reverse:
      return vector[::-1]
    else:
      return vector

  def parse_game_output(self, text, room_description):
    reward = None
    text_to_agent = [room_description, self.get_quest_text(self.quest_checklist[0])]
    for i in xrange(len(text)):
      if i<len(text) and '<EOM>' in text[i]:
        text_to_agent = [room_description, self.get_quest_text(self.quest_checklist[0])]
      elif 'REWARD' in text[i]:
        if self.actions(self.quest_checklist[0]) in text[i]:
          reward = int(re.findall(r'\d+',text[i]))
      elif 'not available' in text[i] or 'not find' in text[i]:
        reward = self.junk_cmd_reward
    if reward == None:
      reward = self.default_reward
    return text_to_agent, reward


  def goalComplete(self, text):
    self.client.send('look')
    desc = self.client.get()
    if ("REWARD" in text) and (self.goals[0] in desc):
      # print "Text: ",
      # print text
      # print "Desc: ",
      # print desc
      # print "Goal: ",
      # print self.goal_room[0]
      # print self.objects, self.actions    
      return True
    else:
      return False

  def get_quest_text(self, quest_num):
    return self.quests_mislead[self.mislead_quest_checklist[0]] + " now but " + self.quests[quest_num] + " now."

  def do(self, action_idx, object_idx):
    command = "%s %s" % (self.actions[action_idx], self.objects[object_idx])
    self.client.send(command)

    return self.get_state(action=self.actions[action_idx], object_=self.objects[object_idx])
