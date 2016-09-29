framework = require 'framework'
require 'client'
require 'utils'
local port = 4001
print(port)
client_connect(port)
login('root', 'root')

framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims/build.ev")

framework.interact()


