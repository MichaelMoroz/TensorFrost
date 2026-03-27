import TensorFrost as tf

class vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def zero(shape):
        return vec3(tf.zeros(shape, tf.float32), tf.zeros(shape, tf.float32), tf.zeros(shape, tf.float32))
    
    def zero_like(val):
        return vec3.zero(val.x.shape)
    
    def const(val, shape):
        return vec3(tf.const(val, shape), tf.const(val, shape), tf.const(val, shape))
    
    def copy(val):
        vec = vec3.zero(val.x.shape)
        vec.set(val)
        return vec
    
    def set(self, other):
        self.x.val = other.x
        self.y.val = other.y
        self.z.val = other.z
    
    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __radd__(self, other):
        return vec3(other.x + self.x, other.y + self.y, other.z + self.z)
    
    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __rsub__(self, other):
        return vec3(other.x - self.x, other.y - self.y, other.z - self.z)
    
    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)
    
    def __rmul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)
    
    def __div__(self, other):
        return vec3(self.x / other, self.y / other, self.z / other)
    
    def __rdiv__(self, other):
        return vec3(other / self.x, other / self.y, other / self.z)
    
    def __truediv__(self, other):
        return vec3(self.x / other, self.y / other, self.z / other)
    
    def __neg__(self):
        return vec3(-self.x, -self.y, -self.z)
    
    def __abs__(self):
        return vec3(tf.abs(self.x), tf.abs(self.y), tf.abs(self.z))
    
    def __pow__(self, other):
        return vec3(self.x ** other, self.y ** other, self.z ** other)
    
    def __rpow__(self, other):
        return vec3(other ** self.x, other ** self.y, other ** self.z)

def mul(a, b):
    return vec3(a.x * b.x, a.y * b.y, a.z * b.z)

def dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z

def cross(a, b):
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)

def length(a):
    return tf.sqrt(dot(a, a))

def distance(a, b):
    return length(a - b)

def normalize(a):
    return a / (length(a) + 1e-6)

def vmin(a, b):
    return vec3(tf.min(a.x, b.x), tf.min(a.y, b.y), tf.min(a.z, b.z))

def vmax(a, b):
    return vec3(tf.max(a.x, b.x), tf.max(a.y, b.y), tf.max(a.z, b.z))

def clamp(a, low, high):
    return vec3(tf.clamp(a.x, low, high), tf.clamp(a.y, low, high), tf.clamp(a.z, low, high))

def exp(a):
    return vec3(tf.exp(a.x), tf.exp(a.y), tf.exp(a.z))

def lerp(a, b, t):
    return a + (b - a) * t

def abs(a):
    return vec3(tf.abs(a.x), tf.abs(a.y), tf.abs(a.z))

def reflect(i, n):
    return i - n * 2.0 * dot(n, i)

def sdBox(p, b):
    d = abs(p) - b
    return tf.min(tf.max(d.x, tf.max(d.y, d.z)), 0.0) + length(max(d, vec3(0.0, 0.0, 0.0)))

def mod(a, b):
    return vec3(tf.modf(a.x, b.x), tf.modf(a.y, b.y), tf.modf(a.z, b.z))