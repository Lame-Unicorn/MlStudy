import gzip
import time
import numpy as np

'''
FUNCTIONS:
'''
def bytes2int(bytes_):
    res = 0
    for byte in bytes_:
        res *= 256
        res += byte
    return res

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def parse_package(filename, type_):
    signal = ['\\', '-', '/', '|']
    if type_ == "labels":
        content = gzip.open(filename, "rb").read()
        num = bytes2int(content[4:8])
        labels = []
        counter = 0
        for byte in content[8:]:
            temp = np.zeros(10)
            temp[byte] = 1
            labels.append(temp)
            counter += 1
            if counter == num:
                break
        return (num, labels)
    if type_ == "images":
        content = gzip.open(filename, "rb").read()
        num = bytes2int(content[4:8])
        h = bytes2int(content[8:12])
        w = bytes2int(content[12:16])
        images = []
        num_counter = 0
        num_record = 0
        pixel_counter = 0
        image = []
        for byte in content[16:]:
            image.append(byte)
            pixel_counter += 1
            if pixel_counter == w*h:
                images.append(np.array(image)/255)
                pixel_counter = 0
                num_counter += 1
                image = []
            if num_counter > num_record and not num_counter % (num // 100):
                print("\rLoading images: %2d%%...%c"%(num_counter // (num // 100), signal[num_counter // (num // 100)% 4] if num_counter < num else ' '), end='')
                #print("Loading images: %d%%..." % (num_counter // (num // 100), signal[num_counter%4]))
                num_record = num_counter
            if num_counter == num:
                break
        print('\n')
        return (num, images)
'''
def show_image(image, w, h):
    print("Show Image:")
    for i in range(h):
        for j in range(w):
            tmp = image[i*w+j]
            if tmp < 128:
                print(' ', end='')
            else:
                print('#', end='')
        print('\n')
    print("End")
'''

'''
CLASSES:
'''

class Layer():
    def __init__(self, size, activate_function = sigmoid):
        self.size = size
        self.activate_function  = activate_function
    
    def initial(self, former_size):
        weights_size = (self.size, former_size)
        bias_size = (self.size, 1)
        self.weights = np.random.normal(0.0, self.size ** -0.5, weights_size)
        self.bias = np.random.normal(0.0, self.size ** -0.5, bias_size)
        self.weights_gradient = np.zeros(weights_size)
        self.bias_gradient = np.zeros(self.size)

    def calc(self, inputs):
        if len(inputs.shape) != 2 or inputs.shape[1] != 1:
            inputs = np.array(inputs, ndmin = 2).T
        self.inputs = inputs
        self.output = self.activate_function(np.dot(self.weights, inputs) + self.bias)

    def derivative_function(self):
        #Rewrite this function when using a different activate function
        return self.output * (1 - self.output)

class Model_Sequential():
    def __init__(self, input_size, *layers):
        self.input_size = input_size
        self.layers = list(layers)
        self.n = len(layers)
        self.layers[0].initial(input_size)
        for i in range(1, self.n):
            self.layers[i].initial(self.layers[i-1].size)

    def spread_forward(self, inputs):
        if len(inputs.shape) != 2 or inputs.shape[1] != 1:
            inputs = np.array(inputs, ndmin = 2).T
        self.inputs = inputs
        self.layers[0].calc(inputs)
        for i in range(1, self.n):
            self.layers[i].calc(self.layers[i-1].output)
        self.output = self.layers[-1].output

    def spread_backward(self, label, alpha = 0.001):
        if len(label.shape) != 2 or label.shape[1] != 1:
            label = np.array(label, ndmin = 2).T
        self.layers[-1].bias_gradient = (self.layers[-1].output - label) * self.layers[-1].derivative_function()
        self.layers[-1].weights_gradient = np.dot(self.layers[-1].bias_gradient, self.layers[-2].output.T)
        for i in range(self.n-2, 0, -1):
            self.layers[i].bias_gradient = np.dot(self.layers[i+1].weights.T, self.layers[i+1].bias_gradient) * self.layers[i].derivative_function()
            self.layers[i].weights_gradient = np.dot(self.layers[i].bias_gradient, self.layers[i-1].output.T)
        self.layers[0].bias_gradient = np.dot(self.layers[1].weights.T, self.layers[1].bias_gradient) * self.layers[0].derivative_function()
        self.layers[0].weights_gradient = np.dot(self.layers[0].bias_gradient, self.inputs.T)

        for i in range(self.n):
            self.layers[i].weights -= alpha * self.layers[i].weights_gradient
            self.layers[i].bias -= alpha * self.layers[i].bias_gradient

    def get_loss(self, label):
        if len(label.shape) != 2 or label.shape[1] != 1:
            label = np.array(label, ndmin = 2).T
        return 1/2 * ((self.layers[-1].output - label)**2).sum()

    def fit(self, train_inputs, train_labels, sample_size, training_iters = 10, show_gap = 1000, write_to_log = False):
        signal = ['\\', '-', '/', '|']
        t0 = time.localtime()
        print("[%02d:%02d:%02d] Start training..."%(t0.tm_hour, t0.tm_min, t0.tm_sec))
        if write_to_log:
            with open("MlLog.txt", "w") as f:
                f.write(' '.join(("Model size:", str(self.input_size), *[str(layer.size) for layer in self.layers], ".\n")))
                f.write("Train data size:%d. Epoch:%d.\n" % (sample_size, training_iters))
                f.write("Train started at %02d:%02d:%02d.\n" % (t0.tm_hour, t0.tm_min, t0.tm_sec))
        loss_sum = 0
        tip0_format = "\rIn epoch %%%dd pic %%%dd...%%c" % (int(np.log10(training_iters) + 1), int(np.log10(sample_size) + 1))
        tip1_format = "\n%%%dd train finished.Current average loss is %%f" % int(np.log10(sample_size * training_iters) + 1)
        for epoch in range(training_iters):
            for i in range(sample_size):
                self.spread_forward(train_images[i])
                loss = self.get_loss(train_labels[i])
                loss_sum += loss
                self.spread_backward(train_labels[i])
                if not (i + 1) % show_gap:
                    print(tip0_format % (epoch + 1, i + 1, signal[(i + 1)//show_gap % 4] if i + 1 < sample_size else ' '), end = '')
            print(tip1_format % ((epoch + 1) * sample_size, loss_sum / (epoch + 1) / sample_size))
        t1 = time.localtime()
        print("\n[%02d:%02d:%02d] Learning program is finished.\nProgram runs %d times, started at %02d:%02d:%02d\nCurrent loss is %f" % \
              (t1.tm_hour, t1.tm_min, t1.tm_sec, (epoch + 1) * sample_size, t0.tm_hour, t0.tm_min, t0.tm_sec, loss_sum / (epoch + 1) / sample_size))
        if write_to_log:
            with open("mlLog.txt", "a") as f:
                f.write("\n[%02d:%02d:%02d] Learning program is finished.\nProgram runs %d times, started at %02d:%02d:%02d\nCurrent loss is %f" % \
                (t1.tm_hour, t1.tm_min, t1.tm_sec, training_iters * sample_size, t0.tm_hour, t0.tm_min, t0.tm_sec, loss_sum / training_iters / sample_size))

    def evaluate(self, test_inputs, test_labels, test_size, write_to_log = False):
        signal = ['\\', '-', '/', '|']
        tip_format = "\r %%%dd pic tested...%%c" % int(np.log10(test_size) + 1)
        count = 0
        print("Test started.")
        for i in range(test_size):
            self.spread_forward(test_inputs[i])
            if self.output.argmax() == test_labels[i].argmax():
                count += 1
            if not (i + 1) % (test_size//100):
                print(tip_format % (i + 1, signal[(i+1)//50 % 4] if i + 1 < test_size else ' '), end = '')
        print("\n[Test ended]\nTest samples: %d\nCorrect samples: %d\nWrong samples: %d\nAccuracy is %.2f%%\n" %\
             (test_size, count, test_size-count, count / test_size * 100))
        if write_to_log:
            with open("MlLog.txt", 'a') as f:
                f.write("[Test Result]\nTest samples: %d\nCorrect samples: %d\nWrong samples: %d\nAccuracy is %.2f%%\n"%\
                       (test_size, count, test_size-count, count / test_size * 100))
            

train_images_filename = "train-images-idx3-ubyte.gz"
train_labels_filename = "train-labels-idx1-ubyte.gz"
test_images_filename = "t10k-images-idx3-ubyte.gz"
test_labels_filename = "t10k-labels-idx1-ubyte.gz"

print("\nCreating model...")
MPL = Model_Sequential(28*28, Layer(80), Layer(10))

print("Loading train data...")
sample_size, train_images = parse_package(train_images_filename, "images")
train_labels = parse_package(train_labels_filename, "labels")[1]

if __name__ == "__main__":
    MPL.fit(train_images, train_labels, sample_size, write_to_log = True)
    print("Loading test data...")
    test_images = parse_package(test_images_filename, "images")[1]
    test_size, test_labels = parse_package(test_labels_filename, "labels")
    MPL.evaluate(test_images, test_labels, test_size, write_to_log = True)