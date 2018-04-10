from xnd import xnd
from ndtypes import ndt
import gumath as gm


# ==============================================================================
#                          Experimental syntax sugar
# ==============================================================================


class Graph(xnd):
   def __new__(cls, *args, **kwargs):
       return super().__new__(cls, *args, typedef="graph")

   def shortest_paths(self, start):
       return gm.single_source_shortest_paths(self, start)


class bfloat16(xnd):
   def __new__(cls, *args, **kwargs):
       return super().__new__(cls, *args, dtypedef="bfloat16")

   def __repr__(self):
       return "bfloat16(%s)" % str(self)
