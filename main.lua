framework = require 'framework'
require 'client'
require 'utils'



local port = 4000 + tonumber(arg[1])
print(port)
client_connect(port)
login('root', 'root')
counter = arg[1]
if arg[1]==1 then
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims/build.ev")
elseif arg[1]==2 then
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims3/build.ev")
elseif arg[1]==3 then
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims4/build.ev")
elseif arg[1]==4 then
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims/build.ev")
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims3/build.ev")
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims4/build.ev")
    counter = 1
end
framework.writeSymbolMapping(arg[1])
print("#symbols", #symbols)
framework.interact(counter)


