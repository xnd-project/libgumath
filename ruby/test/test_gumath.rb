require 'test_helper'

class TestFunctionHash < Minitest::Test
  def test_hash_contents
    hash = Fn.instance_variable_get(:@gumath_functions)

    assert_instance_of Gumath::GufuncObject, hash[:sin]
  end
end

class TestCall < Minitest::Test
  def test_sin_scalar
    x1 = XND.new(1.2, type: "float64")
    y1 = Fn.sin(x1)
    
    x2 = XND.new(1.23e1, type: "float32")
    y2 = Fn.sin(x2)

    assert_in_delta y1.value, Math.sin(1.2), 0.00001
    assert_in_delta y2.value, Math.sin(1.23e1), 0.00001
  end

  def test_sin
    TEST_CASES.each do |data, t, dtype|
      x = XND.new data, type: t
      y = Fn.sin x
      
      sinned = compute :sin, data

      assert_array_in_delta y.value, sinned, 0.00001
    end
  end

  def test_sin_slice
    TEST_CASES.each do |data, t, dtype|
      x = XND.new data, type: t
      next if x.type.ndim < 2

      y = x[0..10, 0..10]
      z = Fn.sin y
    end
  end

  def test_copy
    TEST_CASES.each do |data, t, dtype|
      x = XND.new data, type: t
      y = Fn.copy x

      assert_array y.value, x.value
    end
  end

  def test_quaternion
    data = [
      [[1+2i, 4+3i],
       [-4+3i, 1-2i]],
      [[4+2i, 1+10i],
       [-1+10i, 4-2i]],
      [[-4+2i, 3+10i],
       [-3+10i, -4-2i]]
    ]

    x = XND.new data, type: "3 * quaternion64"
    y = Ex.multiply x, x

    # FIXME: how to do einsum in Ruby?

    x = XND.new data, type: "3 * quaternion128"
    y = Ex.multiply x, x

    x = XND.new "xyz"
    assert_raises(TypeError) { Ex.multiply x, x }
  end

  def test_quaternion_error
    data = [
      [[1+2i, 4+3i],
       [-4+3i, 1-2i]],
      [[4+2i, 1+10i],
       [-1+10i, 4-2i]],
      [[-4+2i, 3+10i],
       [-3+10i, -4-2i]]
    ]
    x = XND.new data, type: "3 * Foo(2 * 2 * complex64)"
    assert_raises(TypeError) { Ex.multiply(x, x) }
  end

  def test_void
    x = Ex.randint

    assert_equal x.type, NDT.new("int32")
  end

  def test_multiple_return
    x, y = Ex.randtuple

    assert_equal x.type, NDT.new("int32")
    assert_equal y.type, NDT.new("int32")

    x, y = Ex.divmod10 XND.new(233)

    assert_equal x.value, 23
    assert_equal y.value, 3
  end
end

class TestMissingValues < Minitest::Test
  def test_missing_values
    x = [{'index'=> 0, 'name'=> 'brazil', 'value'=> 10},
         {'index'=> 1, 'name'=> 'france', 'value'=> nil},
         {'index'=> 1, 'name'=> 'russia', 'value'=> 2}]

    y = [{'index'=> 0, 'name'=> 'iceland', 'value'=> 5},
         {'index'=> 1, 'name'=> 'norway', 'value'=> nil},
         {'index'=> 1, 'name'=> 'italy', 'value'=> nil}]

    z = XND.new [x, y], type: "2 * 3 * {index: int64, name: string, value: ?int64}"
    ans = Ex.count_valid_missing z

    assert_equal ans.value, [{'valid' => 2, 'missing' => 1}, {'valid' => 1, 'missing' => 2}]
  end
end

class TestRaggedArrays < Minitest::Test
  def test_sin
    data = [
      [[1.0],
       [2.0, 3.0],
       [4.0, 5.0, 6.0]],
      [[7.0],
       [8.0, 9.0],
       [10.0, 11.0, 12.0]]
    ]

    ans = [[[Math.sin(1.0)],
            [Math.sin(2.0), Math.sin(3.0)],
            [Math.sin(4.0), Math.sin(5.0), Math.sin(6.0)]],
           [[Math.sin(7.0)],
            [Math.sin(8.0), Math.sin(9.0)],
            [Math.sin(10.0), Math.sin(11.0), Math.sin(12.0)]]]
    
    x = XND.new data
    y = Fn.sin x

    assert_array y.value, ans
  end
end

class TestGraphs < Minitest::Test
  def test_shortest_path
    graphs = [
      [[[1, 1.2],  [2, 4.4]],
       [[2, 2.2]],
       [[1, 2.3]]],

      [[[1, 1.2], [2, 4.4]],
       [[2, 2.2]],
       [[1, 2.3]],
       [[2, 1.1]]]
    ]

    ans = [[[[0], [0, 1], [0, 1, 2]],      # graph1, start 0
            [[], [1], [1, 2]],             # graph1, start 1
            [[], [2, 1], [2]]],            # graph1, start 2

           [[[0], [0, 1], [0, 1, 2], []],  # graph2, start 0
            [[], [1], [1, 2], []],         # graph2, start 1
            [[], [2, 1], [2], []],         # graph2, start 2
            [[], [3, 2, 1], [3, 2], [3]]]] # graph2, start 3
    
    graphs.each do |i, arr|
      n = arr.size
      graph = Graph.new arr

      n.times do |start|
        node = XND.new start, type: "node"
        x = graph.shortest_paths node

        assert_equal x.value, ans[i][start]
      end
    end
  end

  def test_constraint
    data = [
      [[0, 1.2]],
      [[2, 2.2], [1, 0.1]]
    ]

    assert_raises(ValueError) { Graph.new(data) }
  end
end

class TestPdist < Minitest::Test
  def test_exceptions
    x = XND.new [], dtype: "float64"
    assert_raises(TypeError) { Ex.euclidian_pdist x }

    x = XND.new [[]], dtype: "float64"
    assert_raises(TypeError) { Ex.euclidian_pdist x }

    x = XND.new [[], []], dtype: "float64"
    assert_raises(TypeError) { Ex.euclidian_pdist x }

    x = XND.new [[1], [1]], dtype: "int64"
    assert_raises(TypeError) { Ex.euclidian_pdist x }
  end

  def test_pdist
    x = XND.new [[1]], dtype: "float64"
    y = Ex.euclidian_pdist x
    assert_equal y.value, []

    x = XND.new [[1,2,3]], dtype: "float64"
    y = Ex.euclidian_pdist x
    assert_equal y.value, []

    x = XND.new([[-1.2200, -100.5000,   20.1250,  30.1230],
                 [ 2.2200,    2.2720, -122.8400, 122.3330],
                 [ 2.1000,  -25.0000,  100.2000, -99.5000]], dtype: "float64")
    y = Ex.euclidian_pdist x
    assert_equal y.value, [198.78529349275314, 170.0746899276903, 315.75385646576035]
  end
end
