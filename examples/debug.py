import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt
import time

tf.initialize(tf.opengl)

ImageW = 1000
ImageH = 1000
RN = 64

def cubic_hermit(x):
    x2 = x * x
    x3 = x2 * x
    return [-0.5 * x3 + x2 - 0.5 * x, 1.5 * x3 - 2.5 * x2 + 1.0, -1.5 * x3 + 2.0 * x2 + 0.5 * x, 0.5 * x3 - 0.5 * x2]

def bicubic(tex, x, y, ch):
    xi, yi = tf.floor(x), tf.floor(y)
    xf, yf = x-xi, y-yi
    xi, yi = tf.int(xi), tf.int(yi)

    wx = cubic_hermit(xf)
    wy = cubic_hermit(yf)

    valueY = 0.0
    for j in range(-1, 3):
        valueX = 0.0
        for i in range(-1, 3):
            valueX = valueX + tex[xi + i, yi + j, ch] * wx[i + 1]
        valueY = valueY + valueX * wy[j + 1]

    return valueY

def mul_bias(X, W):
    ids = tf.indices(list(X.shape[:-1]) + [W.shape[-2]])
    return tf.select(ids[-1] == X.shape[-1], 0.01, X[ids]) @ W

def GELU(x):
    return 0.5*x*(1.0+tf.tanh(0.7978845608*(x+0.044715*x*x*x)))

class TextureEmbedder(tf.Module):
    def __init__(self, embedding_size = RN, embedding_channels = 16, channel_count = 3):
        super().__init__()
        self.embedding_size = embedding_size
        self.channel_count = channel_count
        self.embedding_channels = embedding_channels
        self.hidden_size = embedding_channels
        #self.dequant_scale = tf.Parameter([embedding_channels], tf.float32, random_scale = 2.0)
        #self.dequant_bias = tf.Parameter([embedding_channels], tf.float32, random_scale = 0.0)
        self.tex = tf.Parameter([embedding_size, embedding_size, embedding_channels], tf.float32, random_scale = 1.0)
        self.x_to_embed = tf.Parameter([embedding_channels, embedding_channels], tf.float32, random_scale = 1.0)
        self.fc1 = tf.Parameter([embedding_channels, self.hidden_size], tf.float32)
        self.fc2 = tf.Parameter([self.hidden_size+1, self.hidden_size], tf.float32)
        self.phase_to_embed = tf.Parameter([embedding_channels, embedding_channels], tf.float32, random_scale = 1.0)

        self.fc3 = tf.Parameter([self.hidden_size+1, channel_count], tf.float32)
        self.y_to_embed = tf.Parameter([embedding_channels, embedding_channels], tf.float32, random_scale = 1.0)

    def sample(self, i, j, k):
        res = self.tex[i, j, k]
        #res = tf.clamp(tf.round(127.0*self.tex[i, j, k]).pass_grad(), -127.0, 127.0) / 127.0
        return res

    def neural_sample(self, i, j, x, y, ch):
        embed = self.sample(i, j, ch)
        embed = embed * tf.sin(x * (embed @ self.x_to_embed) + y * (embed @ self.y_to_embed) + (embed @ self.phase_to_embed))
        embed *= tf.smoothstep(1.5, 0.0, tf.abs(x - tf.float(i))) * tf.smoothstep(1.5, 0.0, tf.abs(y - tf.float(j)))
        return embed

    def learned_interp(self, x, y):
        x = tf.repeat(tf.unsqueeze(x), 9)
        y = y[x.indices[:-1]]
        x = x.T
        y = y.T

        i, j = tf.round(x), tf.round(y)
        ii, jj = tf.int(i), tf.int(j)

        ids = x.indices
        ch = ids[-1]
        it = ids[-2]
        xi = it / 3
        yi = it % 3

        sample = self.neural_sample(ii+xi, jj+yi, x, y, ch)

        return tf.sum(sample, axis = -2)

    def forward(self, x, y):
        embed = self.learned_interp(x, y)
        embed = tf.sin(mul_bias(embed, self.fc1))
        embed = tf.sin(mul_bias(embed, self.fc2))
        return (mul_bias(embed, self.fc3))

    def loss(self, X, Y):
        i, j = tf.indices([X.shape[0], self.embedding_channels])
        Yhat = self.forward(X[i, 0], X[i, 1])
        diff = Yhat - Y
        return tf.mean(tf.mean(diff*diff))

def NeuralEmbed():
    pos = tf.input([-1, 2], tf.float32)
    N = pos.shape[0]
    val = tf.input([N, 3], tf.float32)

    info = tf.input([-1], tf.float32)
    learning_rate = info[0]

    model = TextureEmbedder()
    opt = tf.optimizers.adam(model, learning_rate = learning_rate)
    #opt.set_clipping_type(tf.clipping.norm)
    opt.initialize_input()

    loss = opt.step(pos, val)

    params = opt.parameters()
    params.append(loss)
    return params

reconstruct = tf.compile(NeuralEmbed)

def RenderImage():
    model = TextureEmbedder()
    model.initialize_input()

    i, j, e = tf.indices([ImageH, ImageW, model.embedding_channels])
    x, y = tf.float(i*(model.embedding_size-1))/float(ImageH), tf.float(j*(model.embedding_size-1))/float(ImageW)
    x = tf.float(model.embedding_size-1) - x

    return model.forward(x, y)

render = tf.compile(RenderImage)