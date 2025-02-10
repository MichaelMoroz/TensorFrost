from . import TensorFrost as tf

#in-place bitonic sort
def bitonic(keys, values = None):
    tf.region_begin('Bitonic sort')
    keys = tf.copy(keys)
    if values is not None:
        values = tf.copy(values)
    element_count = keys.shape[0]
    log2_count = tf.int(tf.ceil(tf.log2(tf.float(element_count))))
    count_round = 1 << log2_count
    idx = tf.indices([count_round / 2])[0]
    with tf.loop(log2_count) as k:
        with tf.loop(k+1) as j:
            s = 1 << (k-j)
            m_inner = s - 1
            m_outer = ~m_inner
            m_xor = s + tf.select(j == 0, m_inner, 0)

            id1 = (2 * (idx & m_outer) + (idx & m_inner))
            id2 = id1 ^ m_xor
            key1, key2 = keys[id1], keys[id2]
            with tf.if_cond((key1 >= key2) & (id1 < element_count) & (id2 < element_count)):
                if values is not None:
                    val1, val2 = values[id1], values[id2]
                    values[id1] = val2
                    values[id2] = val1
                keys[id1] = key2
                keys[id2] = key1

    tf.region_end('Bitonic sort')
    if values is not None:
        return keys, values
    else:
        return keys

#histogram radix sort
def radix(keys, values = None, bits_per_pass = 6, max_bits = 32):
    def prefix_sum_grouped(A, axis = -1):
        axis = len(A.shape) + axis if axis < 0 else axis
        group_size = 64
        grouped = tf.split_dim(A, group_size, axis)
        group_scan = tf.prefix_sum(tf.sum(grouped, axis = axis + 1), axis = axis)
        ids = grouped.indices
        gid, eid = ids[axis], ids[axis + 1]
        ids = [ids[i] for i in range(len(ids)) if i != axis + 1]
        ids[axis] = gid - 1
        group_scan = tf.prefix_sum(grouped + tf.select((gid == 0) | (eid != 0), tf.uint(0), group_scan[tuple(ids)]), axis = axis + 1)
        full_scan = tf.merge_dim(group_scan, target_size = A.shape[axis], axis = axis + 1)
        return full_scan

    sign_bit = ~tf.uint(0x7FFFFFFF)

    def map_float_to_uint(x):
        # Convert float to uint representation
        ux = tf.asuint(x)
        # Compute mask
        mask = tf.select((ux >> 31) == 1, ~tf.uint(0), sign_bit)
        # Apply XOR
        return ux ^ mask

    def map_uint_to_float(x):
        # Compute mask
        mask = tf.select((x >> 31) == 0, ~tf.uint(0), sign_bit)
        # Apply XOR and convert back to float
        return tf.asfloat(x ^ mask)

    def map_int_to_uint(x):
        return tf.asuint(x) ^ sign_bit

    def map_uint_to_int(x):
        return tf.asint(x ^ sign_bit)

    tf.region_begin('Radix sort')

    has_values = values is not None

    keys = tf.copy(keys)
    if has_values:
        values = tf.copy(values)

    original_type = keys.type
    if(original_type == tf.float32):
        keys = map_float_to_uint(keys)

    if(original_type == tf.int32):
        keys = map_int_to_uint(keys)

    iters = (max_bits + bits_per_pass - 1) // bits_per_pass
    group_size = 128
    histogram_size = 2 ** bits_per_pass

    def GetBits(A, i):
        return (A >> (i * bits_per_pass)) & tf.uint(histogram_size - 1)

    keys1 = tf.buffer(keys.shape, keys.type)
    values1 = None

    if has_values:
        values1 = tf.buffer(values.shape, values.type)

    with tf.loop(iters // 2) as iter:
        def SortIteration(keys_in, keys_out, values_in, values_out, iter):
            tf.region_begin('Radix sort iteration')
            grouped = tf.split_dim(GetBits(keys_in, iter), group_size)

            # Do a packed histogram, since we sum 128 elements at a time, we can pack 4 values into a single uint32
            g, e, i = tf.indices([grouped.shape[0], grouped.shape[1], tf.int(histogram_size/4)])
            this_key = grouped[g, e]
            packed_is_bit = (tf.uint(this_key == tf.uint(4*i))) + (tf.uint(this_key == tf.uint(4*i+1)) << 8) + (tf.uint(this_key == tf.uint(4*i+2)) << 16) + (tf.uint(this_key == tf.uint(4*i+3)) << 24)
            packed_is_bit = tf.select((g*group_size + e) < keys_in.shape[0], packed_is_bit, tf.uint(0))
            group_histogram_packed = tf.sum(packed_is_bit, axis = 1)

            g, i = tf.indices([grouped.shape[0], histogram_size])
            group_histogram = tf.uint((group_histogram_packed[g, i / 4] >> (8*(i % 4))) & tf.uint(0xFF))

            group_histogram_scan = prefix_sum_grouped(group_histogram, axis = 0)
            i, = tf.indices([histogram_size])
            total_bit_histogram = tf.prefix_sum(group_histogram_scan[group_histogram_scan.shape[0] - 1, i])

            with tf.kernel(grouped.shape, group_size=[group_size]) as (g, e):
                if(tf.current_backend() == tf.cpu): #dont use group barriers on CPU - doesn't work
                    element = g * group_size + e
                    with tf.if_cond(element < keys_in.shape[0]):
                        old_key = keys_in[element]
                        old_val = values_in[element]
                        bit = GetBits(old_key, iter)
                        total_offset = tf.select(g == 0, tf.uint(0), group_histogram_scan[g - 1, bit]) + tf.select(bit == tf.uint(0), tf.uint(0), total_bit_histogram[bit - tf.uint(1)])
                        with tf.loop(e) as j:
                            total_offset.val += tf.uint(grouped[g, j] == bit)
                        keys_out[total_offset] = old_key
                        values_out[total_offset] = old_val
                else:
                    temp = tf.group_buffer(group_size, tf.uint32)
                    half_count = tf.group_buffer(histogram_size, tf.uint32)
                    gtid = g.block_thread_index(0)

                    #initialize counters
                    for i in range((histogram_size + group_size - 1) // group_size):
                        index = gtid + i * group_size
                        with tf.if_cond(index < histogram_size):
                            half_count[index] = 0
                    tf.group_barrier()

                    element = g * group_size + e
                    with tf.if_cond(element < keys_in.shape[0]):
                        old_key = keys_in[element]
                        bit = GetBits(old_key, iter)
                        temp[gtid] = bit

                        #count number of bits set in previous sub groups
                        quarter_index = e / (group_size // 4)
                        with tf.if_cond(quarter_index < 3):
                            tf.scatterAdd(half_count[bit], tf.uint(quarter_index < 1) | (tf.uint(quarter_index < 2) << 8) | (tf.uint(quarter_index < 3) << 16))

                        tf.group_barrier()

                        if has_values:
                            old_val = values_in[element]

                        total_offset = tf.select(g == 0, tf.uint(0), group_histogram_scan[g - 1, tf.int(bit)]) + tf.select(tf.int(bit) == 0, tf.uint(0), total_bit_histogram[tf.int(bit) - 1])
                        total_offset += tf.select(quarter_index > 0, (half_count[bit] >> (8*(quarter_index-1))) & tf.uint(0xFF), tf.uint(0))
                        begin_index = quarter_index * (group_size // 4)
                        with tf.loop(begin_index, e) as j:
                            total_offset.val += tf.uint(temp[j] == bit)
                        keys_out[total_offset] = old_key

                        if has_values:
                            values_out[total_offset] = old_val

            tf.region_end('Radix sort iteration')

        SortIteration(keys, keys1, values, values1, 2 * iter)
        SortIteration(keys1, keys, values1, values, 2 * iter + 1)

    tf.region_end('Radix sort')

    if(original_type == tf.float32):
        keys = map_uint_to_float(keys)

    if(original_type == tf.int32):
        keys = map_uint_to_int(keys)

    if has_values:
        return keys, values
    else:
        return keys
