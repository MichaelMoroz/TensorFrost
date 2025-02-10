from . import TensorFrost as tf

class ModuleOptimizer(tf.Module):
    class OptimizerType:
        ADAM = 0
        SGD = 1
        RMSProp = 2

    class RegularizerType:
        None_ = 0
        L1 = 1
        L2 = 2

    class ClippingType:
        Clamp = 0
        Norm = 1
        None_ = 2

    def __init__(self, optimizer_type, regularizer_type, net, params):
        super().__init__()
        self.optimizer_type = optimizer_type
        self.regularizer_type = regularizer_type
        self.clipping_type = self.ClippingType.Clamp
        self.epsilon = 1e-8

        # Set passed parameters as attributes
        self.net = net
        for k, v in params.items():
            setattr(self, k, v)

        # Initialize t
        t = tf.Parameter([1], tf.float32, False)  # mimic Parameter({1}, TFType::Float, false)
        self.t = t

        self.initializeOptimizer(net)

    def set_clipping_type(self, ctype):
        self.clipping_type = ctype

    def initializeOptimizer(self, net):
        net_params = net.parameters()
        requires_grads = net.requires_grads_list()

        if self.optimizer_type == self.OptimizerType.ADAM:
            self.initializeParameterArray("m", net_params, requires_grads)
            self.initializeParameterArray("v", net_params, requires_grads)
        elif self.optimizer_type == self.OptimizerType.SGD:
            # No additional parameters needed
            pass
        elif self.optimizer_type == self.OptimizerType.RMSProp:
            self.initializeParameterArray("v", net_params, requires_grads)

    def initializeParameterArray(self, name, net_params, requires_grads):
        arr = tf.ParameterArray()

        for i, param in enumerate(net_params):
            if not requires_grads[i]:
                continue

            new_param = tf.Parameter(param.shape, tf.float32, False)
            arr[i] = new_param

        setattr(self, name, arr)

    def assert_parameters(self):
        net_params = self.net.parameters()
        requires_grads = self.net.requires_grads_list()
        self.assertParameterArray("m", net_params, requires_grads)
        self.assertParameterArray("v", net_params, requires_grads)

    def gradient_norm(self, grad):
        # sum of squares
        g = grad * grad
        shape = grad.shape
        num_dims = len(shape)
        for i in range(num_dims):
            g = tf.sum(g)
        return tf.sqrt(g)

    def assertParameterArray(self, name, net_params, requires_grads):
        if hasattr(self, name):
            arr = getattr(self, name)
            for i, param in enumerate(net_params):
                if not requires_grads[i]:
                    continue
                arr_item = arr[i]
                arr_item = tf.assert_tensor(arr_item, param.shape, param.type)
                arr[i] = arr_item

    def step(self, *args):
        # Overloaded step:
        # step(X, Y) or step(loss)
        if len(args) == 2:
            X, Y = args
            loss = self.net.loss(X, Y)
            self._step(loss)
            return loss
        elif len(args) == 1:
            (loss,) = args
            self._step(loss)
        else:
            raise ValueError("Invalid arguments to step")

    def _step(self, loss):
        # Increment t by 1
        self.t = self.t + 1.0

        net = self.net
        net_params = net.parameters()
        requires_grads = net.requires_grads_list()

        learning_rate = self.learning_rate
        grad_clip = self.grad_clip
        has_clip = isinstance(grad_clip, float) and grad_clip > 0.0

        for i, param in enumerate(net_params):
            if not requires_grads[i]:
                continue

            grad = tf.grad(loss, param)
            if has_clip:
                if self.clipping_type == self.ClippingType.Clamp:
                    grad = tf.clamp(grad, -grad_clip, grad_clip)
                elif self.clipping_type == self.ClippingType.Norm:
                    grad_norm = tf.max(1e-6, self.gradient_norm(grad))
                    grad = grad * tf.min(1.0, grad_clip / grad_norm)

            if self.optimizer_type == self.OptimizerType.ADAM:
                update = self.adam_update(i, param, grad, self.t, learning_rate)
            elif self.optimizer_type == self.OptimizerType.SGD:
                update = self.sgd_update(param, grad, learning_rate)
            elif self.optimizer_type == self.OptimizerType.RMSProp:
                update = self.rmsprop_update(i, param, grad, learning_rate)
            else:
                raise RuntimeError("Unknown optimizer type")

            # Apply regularization if needed
            if self.regularizer_type == self.RegularizerType.L1:
                param = param - learning_rate * self.reg * tf.sign(param)
            elif self.regularizer_type == self.RegularizerType.L2:
                param = param - learning_rate * self.reg * param

            # Update parameter with computed update
            param = param - update
            net_params[i] = param

        net.update_parameters(net_params)

    def adam_update(self, i, param, grad, t, learning_rate):
        beta1 = tf.float(self.beta1)
        beta2 = tf.float(self.beta2)

        m = self.m[i]
        v = self.v[i]

        m = tf.lerp(grad, m, beta1)
        v = tf.lerp(grad * grad, v, beta2)

        # t is a Parameter with shape [1]; get the scalar
        t_val = self.t[0]
        mhat = m / (1.0 - tf.pow(beta1, t_val))
        vhat = v / (1.0 - tf.pow(beta2, t_val))

        self.m[i] = m
        self.v[i] = v

        return learning_rate * mhat / (tf.sqrt(vhat) + self.epsilon)

    def sgd_update(self, param, grad, learning_rate):
        return learning_rate * grad

    def rmsprop_update(self, i, param, grad, learning_rate):
        decay = tf.float(self.decay)

        v = self.v[i]
        v = tf.lerp(grad * grad, v, decay)
        self.v[i] = v

        return (grad * learning_rate) / (tf.sqrt(v) + self.epsilon)


def adam(net, reg_type=ModuleOptimizer.RegularizerType.None_, learning_rate=0.001, beta1=0.9, beta2=0.999, clip=0.0, reg=0.0):
    return ModuleOptimizer(
        ModuleOptimizer.OptimizerType.ADAM,
        reg_type,
        net,
        {
            "learning_rate": learning_rate,
            "beta1": beta1,
            "beta2": beta2,
            "grad_clip": clip,
            "reg": reg,
        }
    )

def sgd(net, reg_type=ModuleOptimizer.RegularizerType.None_, learning_rate=0.001, clip=0.0, reg=0.0):
    return ModuleOptimizer(
        ModuleOptimizer.OptimizerType.SGD,
        reg_type,
        net,
        {
            "learning_rate": learning_rate,
            "grad_clip": clip,
            "reg": reg,
        }
    )

def rmsprop(net, reg_type=ModuleOptimizer.RegularizerType.None_, learning_rate=0.001, decay=0.9, clip=0.0, reg=0.0):
    return ModuleOptimizer(
        ModuleOptimizer.OptimizerType.RMSProp,
        reg_type,
        net,
        {
            "learning_rate": learning_rate,
            "decay": decay,
            "grad_clip": clip,
            "reg": reg,
        }
    )