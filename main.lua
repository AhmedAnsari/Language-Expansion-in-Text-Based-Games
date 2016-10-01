framework = require 'framework'
require 'client'
require 'utils'



local port = 4000 + tonumber(arg[1])
print(port)
client_connect(port)
login('root', 'root')
counter = arg[1]
framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims/build.ev")
print("#symbols", #symbols)
framework.interact(counter)


