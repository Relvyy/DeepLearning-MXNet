from tools.tools import *
from model.model_zoo import *


def test(model_string, r):
    models = Model()
    tool = Tools()
    tool.try_gpu()

    model = models.model(model_string),
    model.initialize()
    model.load_parameters(f'.\\runs\\{model_string}_best.params')
    _, test_iter = tool.load_data_fashion_mnist(resize=r)
    tool.test(model, test_iter, r)


if __name__ == '__main__':
    test('ResNet', 224)
