require 'ndtypes'
require 'xnd'

require 'etc'

begin
  require 'ruby_gumath.so'
rescue LoadError
  require 'ruby_gumath/ruby_gumath.so'
end
require 'gumath/version'
