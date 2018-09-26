require 'mkmf'

def windows?
  (/cygwin|mswin|mingw|bccwin|wince|emx/ =~ RUBY_PLATFORM) != nil
end

def mac?
  (/darwin/ =~ RUBY_PLATFORM) != nil
end

def unix?
  !windows?
end

# ndtypes config

ndtypes_version = ">= 0.2.0dev5"
ndtypes_spec = Gem::Specification.find_by_name("ndtypes", ndtypes_version)
ndtypes_extdir = File.join(ndtypes_spec.gem_dir, 'ext', 'ruby_ndtypes')
ndtypes_includedir = File.join(ndtypes_extdir, 'include')
ndtypes_libdir = File.join(ndtypes_extdir, 'lib')

find_header("ruby_ndtypes.h", ndtypes_includedir)
raise "cannot find ruby_ndtypes.h in path #{ndtypes_includedir}." unless have_header("ruby_ndtypes.h")

find_header("ndtypes.h", ndtypes_includedir)
find_library("ndtypes", nil, ndtypes_libdir)

dir_config("ndtypes", [ndtypes_includedir], [ndtypes_libdir])

# xnd config

xnd_version = ">= 0.2.0dev5"
xnd_spec = Gem::Specification.find_by_name("xnd", xnd_version)
xnd_extdir = File.join(xnd_spec.gem_dir, 'ext', 'ruby_xnd')
xnd_includedir = File.join(xnd_extdir, 'include')
xnd_libdir = File.join(xnd_extdir, 'lib')

find_header("ruby_xnd.h", xnd_includedir)
raise "cannot find ruby_xnd.h in path #{xnd_includedir}." unless have_header("ruby_xnd.h")

find_header("xnd.h", xnd_includedir)
find_library("xnd", nil, xnd_libdir)

dir_config("xnd", [xnd_includedir], [xnd_libdir])

# gumath config

puts "compiling libgumath on your machine..."
Dir.chdir(File.join(File.dirname(__FILE__) + "/gumath")) do
  if unix?
    ENV['CFLAGS'] = " -I #{xnd_includedir} -I #{ndtypes_includedir} "   
    system("./configure --prefix=#{File.expand_path("../")} --with-docs=no ")
    system("make")
    system("make install")
  elsif windows?
    raise NotImplementedError, "need to specify build instructions for windows."
  end
end

$INSTALLFILES = [
  ["ruby_gumath.h", "$(archdir)"],
  ["gumath.h", "$(archdir)"]
]

binaries = File.expand_path(File.join(File.dirname(__FILE__) + "/lib/"))
headers = File.expand_path(File.join(File.dirname(__FILE__) + "/include/"))
$LOAD_PATH << File.expand_path(binaries)
append_ldflags("-Wl,-rpath #{binaries}")

find_library("gumath", nil, binaries)

find_header("gumath.h", headers)
have_header("gumath.h")

FileUtils.copy_file File.expand_path(File.join(File.dirname(__FILE__) +
                                               "/ruby_gumath.h")),
                    "#{headers}/ruby_gumath.h"

dir_config("gumath", [headers], [binaries])

# for macOS
append_ldflags("-Wl,-rpath #{binaries}")

basenames = %w{util gufunc_object examples functions ruby_gumath}
$objs = basenames.map { |b| "#{b}.o"   }
$srcs = basenames.map { |b| "#{b}.c" }

$CFLAGS += " -Wall -O0 -fPIC -g "
# FIXME: This is jugaad. Remove on deploy.
$libs += " -lndtypes -lxnd -lgumath "
create_makefile("ruby_gumath/ruby_gumath")
