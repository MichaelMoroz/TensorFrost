import TensorFrost as tf

eps = 1e-8

def safe_divide(a, b):
    return a / tf.select(tf.abs(b) < eps, eps * tf.sign(b), b)

def safe_log(x):
    return tf.log(tf.max(x, eps))

def safe_log2(x):
    return tf.log2(tf.max(x, eps))

def safe_exp(x):
    return tf.exp(tf.clamp(x, -50.0, 50.0))

def safe_exp2(x):
    return tf.exp2(tf.clamp(x, -50.0, 50.0))

def Sort(keys, values, element_count):
    tf.region_begin('Sort')
    log2N = tf.ceil(tf.log2(tf.float(element_count)))
    Nround = tf.int(tf.exp2(log2N))
    sort_id = tf.indices([Nround/2])[0]
    steps = tf.int(log2N*(log2N + 1.0)/2.0)

    with tf.loop(steps) as step:
        def getBitonicElementPair(id, step):
            j = tf.floor(tf.sqrt(tf.float(2*step) + 1.0) - 0.5)
            n = tf.round(tf.float(step) - 0.5*j*(j+1.0))
            B = tf.int(tf.round(tf.exp2(j-n)))
            mask = tf.select(n < 0.5, 2*B - 1, B)
            e1 = id%B + 2*B*(id/B)
            e2 = e1 ^ mask
            return e1, e2
        e1, e2 = getBitonicElementPair(sort_id, step)

        with tf.if_cond((e1 < element_count) & (e2 < element_count)):
            key1, key2 = keys[e1], keys[e2]

            #sort by descending order
            with tf.if_cond(key1 < key2):
                val1, val2 = values[e1], values[e2]
                keys[e1] = key2
                keys[e2] = key1
                values[e1] = val2
                values[e2] = val1

    tf.region_end('Sort')
    return keys, values

def sqr(x):
    return x * x

def aslog(x):
    return lognum(safe_log2(tf.abs(x)), tf.sign(x))

class lognum():
    def __init__(self, value = 0.0, sign = 1.0):
        self.value = value
        self.sign = sign
    
    def asfloat(self):
        return self.sign * tf.exp2(self.value)
    
    def __neg__(self):
        return lognum(self.value, -self.sign)
    
    def add(self, other):
        maxv, minv = tf.max(self.value, other.value), tf.min(self.value, other.value)
        diff = maxv - minv
        value = maxv + tf.select(diff > 24.0, 0.0, safe_log2(1.0 + self.sign * other.sign * safe_exp2(-diff)))
        sign = tf.select(self.value > other.value, self.sign, other.sign)
        return lognum(value, sign)

    def __add__(self, other):
        return self.add(other)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        return lognum(self.value + other.value, self.sign * other.sign)
    
    def __div__(self, other):
        return lognum(self.value - other.value, self.sign * other.sign)
    
    def __pow__(self, other):
        return lognum(self.value * other, self.sign)
    
    def log(self):
        return self.value * 0.69314718056
    
def GELU(x):
    return x / (1.0 + tf.exp(-1.702 * x))

def ELU(x):
    return tf.select(x > 0.0, x, tf.exp(x) - 1.0)

def GLIN(X):
    return 0.5*X + 0.5*X/(1.0 + tf.exp(-X))

def GELU2(x):
    return x*(1.0 + 0.95*tf.tanh(x - 0.779))

def arcsinh(x):
    return tf.log(x + tf.sqrt(x*x + 1.0))

def arcsinh_vjp(g, x):
    return g / tf.sqrt(x*x + 1.0)

def arcsinh_op(inputs, tensor, axes):
    return [arcsinh(inputs[0])]

def arcsinh_op_vjp(inputs, gradient, tensor):
    return [arcsinh_vjp(gradient, inputs[0])]

def register_arcsinh():
    tf.register_custom_operation("arcsinh", ["f_f"], arcsinh_op, arcsinh_op_vjp)

def tf_arcsinh(x):
    return tf.custom("arcsinh", [x])

def packfloatsign(x, sign):
    fbits = tf.asuint(x)
    #put sign bit as least significant bit
    fbits = (fbits & ~tf.uint(1)) | tf.uint(sign >= 0.0)
    return tf.asfloat(fbits)

def getsign(x):
    x = tf.asuint(x)
    return tf.select(x & tf.uint(1) == tf.uint(0), -1.0, 1.0)