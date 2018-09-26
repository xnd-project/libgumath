$LOAD_PATH.unshift File.expand_path('../../lib', __FILE__)
require 'gumath'

require 'minitest/autorun'

Fn = Gumath::Functions
Ex = Gumath::Examples

TEST_CASES = [
  [2000.times.map { |i| Float(i)/100.0 }, "2000 * float64", "float64"],
  
  [[1000.times.map { |i| Float(i)/100.0 }, 1000.times.map { |i| Float(i+1) }],
   "2 * 1000 * float64", "float64"],
  
  [[2.times.map { |i| Float(i+1) }] * 1000, "1000 * 2 * float64", "float64"],
  
  [2000.times.map { |i| Float(i)/10.0 }, "2000 * float32", "float32"],
  
  [[1000.times.map { |i| Float(i)/10.0 }, 1000.times.map { |i| Float(i+1) }],
   "2 * 1000 * float32", "float32"],

  [[2.times.map { |i| Float(i+1) }] * 1000, "1000 * 2 * float32", "float32"]
]

class Graph < XND
  def initialize *args
    super(*args, typedef: "graph")
  end

  def shortest_paths start
    Ex.single_source_shortest_paths start
  end
end

def assert_array_in_delta arr1, arr2, delta
  assert_equal arr1.size, arr2.size

  flat1 = arr1.flatten
  flat2 = arr2.flatten
  
  flat1.each_with_index do |d, i|
    assert_in_delta flat1[i], flat2[i], delta
  end
end


def assert_array arr1, arr2
  assert_equal arr1.size, arr2.size

  arr1.size.times do |i|
    assert_equal arr1[i], arr2[i]
  end
end

def compute func, data
  if data.is_a? Array
    data = data.map do |d|
      compute func, d
    end
  else
    Math.send(func, data)
  end
end
